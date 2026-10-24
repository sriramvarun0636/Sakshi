"""CPU contracts for the multimodal network, focal loss and final hierarchy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sakshi.data import labels
from sakshi.models.fusion import HEAD_CLASSES, HUMANITARIAN_CONDITIONAL, FusionClassifier, FusionConfig
from sakshi.models.losses import IGNORE, class_weights, focal_loss, mask_targets_for_dropout, multitask_loss
from sakshi.models.predict import composite_humanitarian, finalize


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(50, 12)
        self.seen_batch_sizes = []

    def forward(self, *, input_ids, attention_mask):
        self.seen_batch_sizes.append(len(input_ids))
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_model():
    encoder = TinyTextEncoder()
    config = FusionConfig(
        text_hidden=12,
        image_hidden=8,
        image_tokens=3,
        d_model=16,
        cross_attention_layers=2,
        attention_heads=4,
        ffn_dim=32,
        head_hidden=10,
        dropout=0,
        head_dropout=0,
    )
    return FusionClassifier(encoder, config), encoder


def batch():
    return {
        "input_ids": torch.tensor([[1, 2, 0], [0, 0, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 0], [0, 0, 0], [1, 1, 1]]),
        "image_tokens": torch.randn(3, 3, 8),
        "has_text": torch.tensor([True, False, True]),
        "has_image": torch.tensor([True, True, False]),
    }


def test_fusion_shapes_missing_modality_and_gradient():
    torch.manual_seed(0)
    model, encoder = tiny_model()
    model.eval()
    data = batch()
    output = model(**data)
    assert encoder.seen_batch_sizes == [2]  # no BERTweet call for the image-only row
    for name, classes in HEAD_CLASSES.items():
        assert output[name].shape == (3, len(classes))
        assert torch.isfinite(output[name]).all()
    assert torch.allclose(output["alpha"].sum(dim=1), torch.ones(3))
    assert output["alpha"][1, 0] == 0
    assert output["alpha"][2, 1] == 0

    changed = {**data, "image_tokens": data["image_tokens"].clone()}
    changed["image_tokens"][2] = 1000  # ignored on the text-only row
    changed["input_ids"] = data["input_ids"].clone()
    changed["input_ids"][1] = 49  # ignored on the image-only row
    changed_output = model(**changed)
    for name in HEAD_CLASSES:
        assert torch.allclose(changed_output[name], output[name], atol=1e-6)

    output["informative"].sum().backward()
    assert encoder.embedding.weight.grad is not None
    assert model.no_text.grad is not None
    assert model.no_image.grad is not None


def test_fusion_rejects_empty_rows_and_can_disable_auxiliary_heads():
    model, _ = tiny_model()
    data = batch()
    data["has_image"][1] = False
    with pytest.raises(ValueError, match="text or image"):
        model(**data)
    config = FusionConfig(text_hidden=12, image_hidden=8, image_tokens=3, d_model=16, attention_heads=4, auxiliary_heads=False)
    output = FusionClassifier(TinyTextEncoder(), config)(**batch())
    assert set(output) == {"alpha", "informative", "humanitarian", "severity", "disaster_type"}


def test_fusion_accepts_autocast_encoder_states_without_scatter_dtype_mismatch(monkeypatch):
    model, encoder = tiny_model()
    original = encoder.forward

    def reduced_precision(**kwargs):
        return SimpleNamespace(last_hidden_state=original(**kwargs).last_hidden_state.to(torch.bfloat16))

    monkeypatch.setattr(encoder, "forward", reduced_precision)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**batch())
        loss = sum(value.float().square().mean() for key, value in output.items() if key != "alpha")
    assert torch.isfinite(loss)
    loss.backward()
    assert encoder.embedding.weight.grad is not None and torch.isfinite(encoder.embedding.weight.grad).all()


def test_focal_weights_masks_and_empty_heads():
    weights = class_weights(torch.tensor([100, 25, 0]))
    assert weights[1] > weights[0] and weights[2] == 0
    assert torch.isclose((weights * torch.tensor([100, 25, 0])).sum() / 125, torch.tensor(1.0))
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    target = torch.tensor([0, IGNORE])
    loss = focal_loss(logits, target, torch.ones(2))
    assert loss is not None and loss > 0
    assert focal_loss(logits, torch.full_like(target, IGNORE), torch.ones(2)) is None
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))

    targets = {name: torch.tensor([0, 0]) for name in HEAD_CLASSES}
    masked = mask_targets_for_dropout(
        targets, dropped_text=torch.tensor([True, False]), dropped_image=torch.tensor([False, True])
    )
    assert masked["informative"].tolist() == [0, 0]
    assert masked["humanitarian"].tolist() == [0, 0]
    assert masked["severity"].tolist() == [0, IGNORE]
    assert masked["informative_text"].tolist() == [IGNORE, 0]
    output = {"informative": torch.zeros(2, 2, requires_grad=True), "alpha": torch.ones(2, 2)}
    total, parts = multitask_loss(output, targets, {"informative": torch.ones(2)})
    assert set(parts) == {"informative"} and total == parts["informative"]


def test_one_optimizer_step_with_all_modality_patterns_and_heads():
    torch.manual_seed(7)
    model, _ = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.heads["informative"][0].weight.detach().clone()
    outputs = model(**batch())
    targets = {name: torch.tensor([0, IGNORE, IGNORE]) for name in HEAD_CLASSES}
    targets["disaster_type"] = torch.tensor([IGNORE, 2, IGNORE])
    targets["informative_text"] = torch.tensor([0, IGNORE, 1])
    targets["humanitarian_text"] = torch.tensor([1, IGNORE, 2])
    targets["informative_image"] = torch.tensor([0, 1, IGNORE])
    targets["humanitarian_image"] = torch.tensor([1, 2, IGNORE])
    weights = {name: torch.ones(len(classes)) for name, classes in HEAD_CLASSES.items()}
    loss, parts = multitask_loss(outputs, targets, weights)
    assert torch.isfinite(loss) and len(parts) == len(HEAD_CLASSES)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.heads["informative"][0].weight)


def _probs(classes, winner, high):
    return {name: high if name == winner else (1 - high) / (len(classes) - 1) for name in classes}


def test_hierarchy_conditional_decision_and_abstention():
    p_inf = {"not_informative": 0.49, "informative": 0.51}
    q_hum = _probs(HUMANITARIAN_CONDITIONAL, "affected_individuals", 0.4)
    composite = composite_humanitarian(p_inf, q_hum)
    assert sum(composite.values()) == pytest.approx(1)
    assert max(composite, key=composite.get) == "not_humanitarian"
    decision = finalize(
        post_id="p",
        model_id="m",
        p_inf=p_inf,
        q_hum=q_hum,
        p_sev=_probs(labels.SEVERITY, "mild_damage", 0.8),
        p_dt=_probs(labels.DISASTER_TYPE, "fire", 0.9),
        has_text=True,
        has_image=True,
        alpha_image=0.6,
        tau_inf=0.5,
        tau_hum=0.6,
    )
    assert decision.humanitarian.label == "affected_individuals"  # conditional rule, not composite argmax
    assert decision.abstain_reason == "low_confidence:humanitarian"
    assert decision.severity.label == "mild_damage"
    assert decision.disaster_type.label == "fire"

    gated = finalize(
        post_id="p",
        model_id="m",
        p_inf={"not_informative": 0.8, "informative": 0.2},
        q_hum=q_hum,
        p_sev={},
        p_dt={},
        has_text=True,
        has_image=False,
        alpha_image=0,
    )
    assert gated.humanitarian.label == "not_humanitarian" and gated.humanitarian.status == "gated"
    assert gated.severity.status == "unavailable" and gated.disaster_type.status == "unavailable"
