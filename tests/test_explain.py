"""Offline explanation contracts, real attribution gradients and perturbation semantics."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from sakshi.data.features import ClipFeaturizer
from sakshi.explain.common import Example
from sakshi.explain.faithfulness import mask_patches, mask_text, ranking_curve
from sakshi.explain.gradcam import LiveClipFeaturizer, crop_to_original, extract_regions, grad_cam, pool_live_tokens
from sakshi.explain.ig import extract_spans, integrated_gradients, merge_subwords, word_groups
from sakshi.explain.packets import make_packet, validate_packet
from sakshi.explain.render import blur_faces, export_overlay
from sakshi.explain.sanity import cascade, correlation
from sakshi.explain.workflow import _request, select_rows, stratified_ids
from sakshi.models.fusion import FusionClassifier, FusionConfig
from sakshi.models.predict import finalize
from sakshi.types import Explanation, ImageRegion, ProvenanceResult, ProvStatus, TextSpan


class Tokenizer:
    all_special_ids = [0, 1, 2, 3, 9]
    pad_token_id, unk_token_id, mask_token_id = 0, 9, 3
    vocab = {"A": [4], "bridge": [5], "washed": [6, 7], "away": [8]}

    def __call__(self, word, **kwargs):
        return {"input_ids": self.vocab.get(word, [9])}

    def convert_ids_to_tokens(self, ids):
        lookup = {6: "wash@@", 7: "ed"}
        return [lookup.get(i, "<unk>") for i in ids]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens).replace("@@ ", "")


def prediction(post_id="example", *, informative=True, has_text=True, has_image=True):
    return finalize(
        post_id=post_id,
        model_id="fixture",
        p_inf={"not_informative": 0.1 if informative else 0.9, "informative": 0.9 if informative else 0.1},
        q_hum={
            "affected_individuals": 0.1,
            "infrastructure_and_utility_damage": 0.7,
            "rescue_volunteering_or_donation_effort": 0.1,
            "other_relevant_information": 0.1,
        },
        p_sev={"little_or_no_damage": 0.1, "mild_damage": 0.2, "severe_damage": 0.7},
        p_dt={
            "earthquake": 0.1,
            "fire": 0.1,
            "flood": 0.1,
            "hurricane": 0.4,
            "landslide": 0.1,
            "not_disaster": 0.1,
            "other_disaster": 0.1,
        },
        has_text=has_text,
        has_image=has_image,
        alpha_image=0.5 if has_text and has_image else float(has_image),
    )


def example():
    return Example(
        "example",
        "A bridge washed away",
        "A bridge washed away",
        {
            "input_ids": torch.tensor([[1, 4, 5, 6, 7, 8, 2]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "image_tokens": torch.randn(1, 50, 8),
            "has_text": torch.tensor([True]),
            "has_image": torch.tensor([True]),
        },
        prediction(),
        torch.randn(1, 3, 224, 224),
        (448, 224),
    )


class LinearText(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Module()
        self.text_encoder.embedding = nn.Embedding(10, 2)
        self.text_encoder.get_input_embeddings = lambda: self.text_encoder.embedding
        with torch.no_grad():
            self.text_encoder.embedding.weight.copy_(torch.arange(20).reshape(10, 2) / 20)
            self.text_encoder.embedding.weight[0] = 0

    def forward(self, input_ids, **kwargs):
        score = self.text_encoder.embedding(input_ids).sum((1, 2))
        return {"informative": torch.stack((-score, score), dim=-1)}


def test_bpe_merging_unknowns_and_truncation():
    assert merge_subwords(["<s>", "wash@@", "ed", "away", "</s>"], [0, 1, 2, -1, 0], special_tokens={"<s>", "</s>"}) == [
        ("washed", 3),
        ("away", -1),
    ]
    assert merge_subwords(["wash@@"], [2]) == [("wash", 2)]
    assert word_groups(Tokenizer(), "A unknown washed", [1, 4, 9, 6, 7, 2]) == [("A", [1]), ("unknown", [2]), ("washed", [3, 4])]
    assert word_groups(Tokenizer(), "A washed away", [1, 4, 6, 2])[-1] == ("wash@@", [2])
    with pytest.raises(ValueError, match="alignment"):
        word_groups(Tokenizer(), "A", [1, 5, 2])


def test_spans_respect_privacy_contiguity_window_and_global_positive_mass():
    words = [("start", 1), ("@USER", 5), *[(f"w{i}", float(i + 1)) for i in range(12)], ("HTTPURL", 4), ("last", -1)]
    spans = extract_spans(words, "informative")
    assert len(spans) == 1
    assert spans[0].text == " ".join(f"w{i}" for i in range(4, 12))
    assert spans[0].score == pytest.approx(sum(range(5, 13)) / (1 + 5 + sum(range(1, 13)) + 4))
    assert not extract_spans([("none", -1), ("zero", 0)], "informative")
    assert not extract_spans([("photo.jpg", 3), ("HTTPURL", 2), ("RT", 1)], "informative")


def test_actual_captum_ig_matches_linear_difference_and_restores_model():
    model = LinearText().train()
    model.text_encoder.embedding.weight.grad = torch.ones_like(model.text_encoder.embedding.weight)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = integrated_gradients(model, Tokenizer(), example(), "informative")
    assert result.steps == 50 and not result.flags
    assert sum(score for _, score in result.words) == pytest.approx(result.output_difference, abs=1e-5)
    assert abs(result.delta) < 1e-5
    assert model.training and model.text_encoder.embedding.weight.requires_grad
    assert torch.all(model.text_encoder.embedding.weight.grad == 1)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    gated = replace(example(), prediction=prediction(informative=False))
    assert integrated_gradients(model, Tokenizer(), gated, "humanitarian") is None
    assert integrated_gradients(model, Tokenizer(), replace(example(), model_text=None), "informative") is None


def test_ig_retries_and_records_unconverged_alignment_fallback(monkeypatch):
    calls = []

    class LIG:
        def __init__(self, *args):
            pass

        def attribute(self, ids, *, n_steps, **kwargs):
            calls.append(n_steps)
            return torch.ones(1, ids.shape[1], 2), torch.tensor([100.0])

    monkeypatch.setattr("captum.attr.LayerIntegratedGradients", LIG)
    result = integrated_gradients(LinearText(), Tokenizer(), replace(example(), text_clean="different count"), "informative")
    assert calls == [50, 150]
    assert result.flags == ["word_alignment_fallback", "ig_unconverged"]


def test_ig_word_baseline_does_not_move_roberta_positions():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.word_embeddings = nn.Embedding(10, 2, padding_idx=0)
            self.embeddings.position_embeddings = nn.Embedding(20, 2, padding_idx=0)

        def get_input_embeddings(self):
            return self.embeddings.word_embeddings

        def forward(self, input_ids, position_ids=None):
            if position_ids is None:
                mask = input_ids.ne(0).long()
                position_ids = mask.cumsum(1) * mask
            return self.embeddings.word_embeddings(input_ids) + self.embeddings.position_embeddings(position_ids)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = Encoder()

        def forward(self, input_ids, **kwargs):
            score = self.text_encoder(input_ids=input_ids).sum((1, 2))
            return {"informative": torch.stack((-score, score), dim=-1)}

    torch.manual_seed(2)
    model = Model()
    result = integrated_gradients(model, Tokenizer(), example(), "informative")
    assert abs(result.delta) < 1e-5 and "ig_unconverged" not in result.flags
    assert not model.text_encoder._forward_pre_hooks


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(768)

    def forward(self, tokens):
        values = self.layer_norm1(tokens)
        return values + values.mean(1, keepdim=True)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.randn(1, 1, 768))
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([VisionBlock()])
        self.post_layernorm = nn.LayerNorm(768)

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        tokens = self.encoder.layers[0](torch.cat((self.cls.expand(len(patches), -1, -1), patches), 1))
        return SimpleNamespace(last_hidden_state=tokens, pooler_output=self.post_layernorm(tokens[:, 0]))


def tiny_live():
    class Processor:
        def __call__(self, images, **kwargs):
            arrays = [np.asarray(image.resize((224, 224)), dtype=np.float32).transpose(2, 0, 1) / 255 for image in images]
            return {"pixel_values": torch.from_numpy(np.stack(arrays))}

    cached = ClipFeaturizer.__new__(ClipFeaturizer)
    cached.torch, cached.device, cached.processor = torch, "cpu", Processor()
    cached.model = nn.Module()
    cached.model.vision_model, cached.model.visual_projection = Vision(), nn.Linear(768, 512)
    return LiveClipFeaturizer(cached), cached


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_live_matches_cache_on_five_images_and_real_gradcam_flows_through_frozen_tower():
    torch.manual_seed(7)
    live, cached = tiny_live()
    config = FusionConfig(
        text_hidden=8,
        image_hidden=768,
        image_tokens=50,
        d_model=8,
        attention_heads=2,
        ffn_dim=16,
        head_hidden=8,
        cross_attention_layers=1,
        dropout=0,
        head_dropout=0,
    )
    model = FusionClassifier(TinyEncoder(), config).eval()
    rng = np.random.default_rng(0)
    for _ in range(5):
        image = Image.fromarray(rng.integers(0, 256, (224, 448, 3), dtype=np.uint8))
        tokens, _ = cached.encode([image])
        pixels = live.pixels(image)
        assert live.assert_cached(pixels, torch.from_numpy(tokens)) < 1e-5
        image.close()
    item = example()
    item.inputs["image_tokens"] = live(item.pixels).detach()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    attributed = grad_cam(model, live, item, "severity")
    assert attributed.heatmap.shape == (14, 14)
    assert np.isfinite(attributed.heatmap).all() and 0 <= attributed.heatmap.min() <= attributed.heatmap.max() <= 1
    assert attributed.heatmap.max() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in live.parameters())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not live.target_layer._forward_hooks
    with pytest.raises(ValueError, match="drift"):
        live.assert_cached(item.pixels, item.inputs["image_tokens"] + 1)
    assert pool_live_tokens(torch.zeros(1, 197, 8)).shape == (1, 50, 8)


def test_region_components_and_original_coordinates():
    assert crop_to_original((0, 0, 224, 224), (448, 224)) == (112, 0, 336, 224)
    assert crop_to_original((0, 0, 224, 224), (64, 128)) == (0, 32, 64, 96)
    assert not extract_regions(np.zeros((14, 14)), (448, 224), "severity")
    assert not extract_regions(np.full((14, 14), np.nan), (448, 224), "severity")
    heat = np.zeros((14, 14), dtype=np.float32)
    heat[1:5, 1:5], heat[9:13, 9:13] = 1, 0.8
    regions = extract_regions(heat, (448, 224), "severity")
    assert len(regions) == 2 and regions[0].score > regions[1].score
    assert all(112 <= r.bbox_xyxy[0] < r.bbox_xyxy[2] <= 336 for r in regions)
    assert sum(r.score for r in regions) <= 1.00001


def test_masks_keep_special_tokens_and_allow_fully_masked_inputs():
    ids = torch.tensor([[1, 4, 5, 2]])
    assert mask_text(ids, [[1], [2]], set(), keep_only=True, mask_token_id=3).tolist() == [[1, 3, 3, 2]]
    assert mask_text(ids, [[1], [2]], {0}, keep_only=False, mask_token_id=3).tolist() == [[1, 3, 5, 2]]
    pixels = torch.ones(1, 3, 224, 224)
    removed = mask_patches(pixels, {0}, keep_only=False)
    assert removed[:, :, :16, :16].sum() == 0 and removed[:, :, 16:, :].min() == 1
    assert mask_patches(pixels, set(), keep_only=True).sum() == 0
    assert pixels.min() == 1 and torch.equal(ids, torch.tensor([[1, 4, 5, 2]]))


def test_ranking_curve_exact_probabilities_and_random_baseline():
    def score(selected, keep):
        return 0.9 if (0 in selected) == keep else 0.1

    actual = ranking_curve(np.r_[1, np.zeros(99)], score, 0.9)
    assert all(row["comprehensiveness"] == pytest.approx(0.8) and row["sufficiency"] == 0 for row in actual)
    assert [row["units"] for row in actual] == [1, 5, 10, 20, 50]
    random = ranking_curve(np.random.default_rng(0).random(100), score, 0.9)
    assert np.mean([row["comprehensiveness"] for row in actual]) > np.mean([row["comprehensiveness"] for row in random])


def test_sanity_restores_rng_and_weights_even_on_failure():
    torch.manual_seed(17)
    model = FusionClassifier(
        TinyEncoder(),
        FusionConfig(text_hidden=8, image_hidden=8, image_tokens=50, d_model=8, attention_heads=2, ffn_dim=16, head_hidden=8),
    )
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="intentional"), cascade(model, "informative") as stages:
        assert list(stages) == ["head", "cross_attention", "gate"]
        assert not torch.equal(model.blocks[-1].text_to_image.in_proj_weight, saved["blocks.1.text_to_image.in_proj_weight"])
        raise RuntimeError("intentional")
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(value, saved[key]) for key, value in model.state_dict().items())
    assert correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert correlation([0, 0], [1, 2]) is None


def test_packets_schema_privacy_model_and_gated_heads():
    item = example()
    explanations = [
        Explanation(
            item.post_id,
            head,
            getattr(item.prediction, head).label,
            [],
            [TextSpan("T9", head, "bridge washed away", 1, 4, 0.7)] if head == "humanitarian" else [],
            None,
            [ImageRegion("I9", head, (120, 20, 200, 100), 0.6)] if head == "severity" else [],
            0.0,
            model_id="fixture",
        )
        for head in ("humanitarian", "severity")
    ]
    provenance = ProvenanceResult(
        item.post_id,
        ProvStatus.RECYCLED,
        "clip",
        0.955,
        None,
        "source",
        "prior_event",
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        "source",
        None,
        False,
        10,
    )
    packet = make_packet(item, event=None, posted_at=None, provenance=provenance, explanations=explanations)
    assert [e["id"] for e in packet["evidence"]] == ["T1", "I1"]
    assert packet["predictions"]["humanitarian"]["p"] == 0.63
    assert packet["provenance"]["origin_first_seen"] == "2016-01-01"
    assert packet["image_caption"] is None
    for unsafe in ("@someone", "https://example.com", "photo.png"):
        packet["text_clean"] = unsafe
        with pytest.raises(ValueError, match="handle, URL"):
            validate_packet(packet)
    with pytest.raises(ValueError, match="explanation differs"):
        make_packet(
            item, event=None, posted_at=None, provenance=provenance, explanations=[replace(explanations[0], model_id="other")]
        )
    negative = replace(item, prediction=prediction(informative=False))
    with pytest.raises(ValueError, match="explanation differs"):
        make_packet(negative, event=None, posted_at=None, provenance=provenance, explanations=explanations)
    packet = make_packet(negative, event=None, posted_at=None, provenance=provenance, explanations=[])
    assert packet["predictions"]["severity"] == {"label": None, "p": None, "status": "gated"}


def test_selectors_sample_strata_and_request_reuse(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "split": ["dev", "dev", "test"],
            "informative_label": ["one", "one", "rare"],
            "text": ["canonical"] * 3,
        }
    )
    assert set(stratified_ids(frame, "informative", 2)) & {"c"}
    assert len(set(stratified_ids(frame, "informative", 99))) == 3
    assert select_rows(frame, "dev").sample_id.tolist() == ["a", "b"]
    selector = tmp_path / "rows.parquet"
    pd.DataFrame({"sample_id": ["b"], "text": ["tampered"]}).to_parquet(selector)
    assert select_rows(frame, str(selector)).text.iloc[0] == "canonical"
    _request(tmp_path / "audit", {"checkpoint": "one"})
    _request(tmp_path / "audit", {"checkpoint": "one"})
    with pytest.raises(ValueError, match="different"):
        _request(tmp_path / "audit", {"checkpoint": "two"})


def test_overlay_blurs_detected_face_and_preserves_outside_crop(tmp_path):
    class Detector:
        def detectMultiScale(self, *args, **kwargs):
            return [(10, 10, 30, 30)]

    pixels = np.random.default_rng(0).integers(0, 256, (64, 128, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    blurred = blur_faces(image, detector=Detector())
    assert np.asarray(blurred)[10:40, 10:40].var() < pixels[10:40, 10:40].var() / 5
    output = export_overlay(image, np.ones((14, 14), dtype=np.float32), tmp_path / "overlay.png", detector=Detector())
    with Image.open(output) as result:
        assert result.size == image.size
        assert np.array_equal(np.asarray(result)[50:, :20], pixels[50:, :20])
    blurred.close()
    image.close()


def test_default_face_detector_is_available_and_missing_detector_fails_closed(monkeypatch):
    import cv2

    image = Image.new("RGB", (64, 64), (80, 80, 80))
    blurred = blur_faces(image)
    assert blurred.size == image.size
    monkeypatch.delattr(cv2, "CascadeClassifier")
    with pytest.raises(RuntimeError, match="refusing figure export"):
        blur_faces(image)
    blurred.close()
    image.close()
