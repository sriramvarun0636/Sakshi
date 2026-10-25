import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sakshi.config import load
from sakshi.models import predict as prediction
from sakshi.models import train as training
from sakshi.models.calibrate_run import calibrate_run
from sakshi.models.fusion import HEAD_CLASSES, FusionClassifier, FusionConfig
from sakshi.models.train import OptimSettings, TrainConfig, evaluate_dev, optimizer_groups, train_one_epoch


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def fixture():
    model_config = FusionConfig(
        text_hidden=8,
        image_hidden=8,
        image_tokens=3,
        d_model=16,
        attention_heads=4,
        cross_attention_layers=1,
        ffn_dim=32,
        head_hidden=10,
        dropout=0,
        head_dropout=0,
    )
    model = FusionClassifier(TinyEncoder(), model_config)
    targets = {name: torch.tensor([0, -100, -100]) for name in HEAD_CLASSES}
    targets["disaster_type"] = torch.tensor([-100, 1, -100])
    batch = {
        "inputs": {
            "input_ids": torch.tensor([[1, 2], [0, 0], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [0, 0], [1, 1]]),
            "image_tokens": torch.randn(3, 3, 8),
            "has_text": torch.tensor([True, False, True]),
            "has_image": torch.tensor([True, True, False]),
        },
        "targets": targets,
        "humanitarian_gold": torch.tensor([0, -100, 2]),
    }
    return model, model_config, batch


def test_full_config_loads_and_one_training_epoch_updates_weights():
    config = load(Path("configs/classifier/full.yaml"), TrainConfig)
    assert config.model.d_model == 512 and config.data.samples_per_epoch == 16000
    model, model_config, batch = fixture()
    settings = OptimSettings(batch_size=3, max_epochs=1)
    config = TrainConfig(model=model_config, optim=settings)
    groups = optimizer_groups(model, settings)
    assert {group["lr"] for group in groups} == {settings.lr_text, settings.lr_other}
    assert {group["weight_decay"] for group in groups} == {0.0, settings.weight_decay}
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    weights = {name: torch.ones(len(classes)) for name, classes in HEAD_CLASSES.items()}
    before = model.heads["informative"][0].weight.detach().clone()
    metrics = train_one_epoch(model, [batch], optimizer, scheduler, weights, config, torch.device("cpu"))
    assert metrics["nonfinite_batches"] == 0 and metrics["n_informative"] == 1
    assert not torch.equal(before, model.heads["informative"][0].weight)
    dev = evaluate_dev(model, [batch], torch.device("cpu"))
    assert set(dev) == {"informative", "humanitarian", "severity", "disaster_type", "selection_score"}
    assert 0 <= dev["selection_score"] <= 1


def test_partial_accumulation_group_has_full_weight():
    torch.manual_seed(8)
    reference, model_config, batch = fixture()
    accumulated, _, _ = fixture()
    accumulated.load_state_dict(reference.state_dict())
    weights = {name: torch.ones(len(classes)) for name, classes in HEAD_CLASSES.items()}

    def one_step(model, grad_accum):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        config = TrainConfig(model=model_config, optim=OptimSettings(grad_accum=grad_accum, grad_clip=1e6))
        train_one_epoch(model, [batch], optimizer, scheduler, weights, config, torch.device("cpu"))
        return scheduler.last_epoch

    assert one_step(reference, 1) == one_step(accumulated, 2)
    for left, right in zip(reference.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(left, right)


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "select_device", lambda requested="auto": torch.device("cpu"))

    class Store:
        def has_image(self, sample_id):
            return True

        def tokens(self, sample_id):
            return np.zeros((3, 8), dtype=np.float32)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
            }

    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "dataset": ["crisismmd", "cib", "crisismmd"],
            "split": ["train", "train", "dev"],
            "text": ["Bridge damaged", pd.NA, "Road clear"],
            "y_informative": ["informative", pd.NA, "not_informative"],
            "y_humanitarian": ["infrastructure_and_utility_damage", pd.NA, "not_humanitarian"],
            "y_severity": ["mild_damage", pd.NA, pd.NA],
            "y_disaster_type": [pd.NA, "fire", pd.NA],
            "informative_text": ["informative", pd.NA, "not_informative"],
            "humanitarian_text": ["infrastructure_and_utility_damage", pd.NA, "not_humanitarian"],
            "informative_image": ["informative", pd.NA, "not_informative"],
            "humanitarian_image": ["infrastructure_and_utility_damage", pd.NA, "not_humanitarian"],
        }
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    frame.to_parquet(processed / "manifest.parquet")
    monkeypatch.setattr(training.paths, "PROCESSED", processed)
    monkeypatch.setattr(training, "FeatureStore", Store)
    monkeypatch.setattr(training, "load_tokenizer", lambda name, **kwargs: Tokenizer())
    monkeypatch.setattr(training.FusionClassifier, "from_pretrained", classmethod(lambda cls, cfg: cls(TinyEncoder(), cfg)))
    _, model_config, _ = fixture()
    config = TrainConfig(
        model=model_config,
        data=training.DataSettings(samples_per_epoch=4),
        optim=OptimSettings(batch_size=2, max_epochs=1, patience=1),
    )
    return config, frame, Store


def test_training_run_writes_checkpoint_and_metadata_without_model_download(monkeypatch, tmp_path, tiny_training):
    config, frame, Store = tiny_training
    run_dir = training.run_training(config, output_dir=tmp_path / "runs")
    for filename in (
        "config.yaml",
        "env.json",
        "checkpoint.pt",
        "label_maps.json",
        "class_weights.json",
        "train_log.jsonl",
        "metrics_dev.json",
    ):
        assert (run_dir / filename).is_file()
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    monkeypatch.setattr(prediction, "load_tokenizer", training.load_tokenizer)
    predictor = prediction.Predictor.from_run(run_dir, device="cpu")
    predictor.features = Store()
    predictions = predictor.predict_rows(frame.iloc[:2])
    assert predictions.sample_id.tolist() == ["a", "b"]
    calibration = calibrate_run(run_dir, features=Store(), rows=frame)
    assert calibration["thresholds"]["reason"] == "too_few_rows"
    assert (run_dir / "temperatures.json").is_file()
    assert (run_dir / "calibration_metrics.json").is_file()
    reloaded = prediction.Predictor.from_run(run_dir, device="cpu")
    assert all(value == 1.0 for value in reloaded.temperatures.values())


@pytest.mark.parametrize("modality", ["text", "image"])
def test_unimodal_run_trains_calibrates_and_predicts(monkeypatch, tmp_path, tiny_training, modality):
    config, frame, Store = tiny_training
    config = replace(
        config,
        name=f"baseline-{modality}",
        model=replace(config.model, modality=modality),
        data=replace(config.data, modality_dropout=0.0),
    )
    run_dir = training.run_training(config, output_dir=tmp_path / "runs")
    monkeypatch.setattr(prediction, "load_tokenizer", training.load_tokenizer)
    calibration = calibrate_run(run_dir, features=Store(), rows=frame)
    assert set(calibration["temperatures"]) == (
        {"informative", "humanitarian"} if modality == "text" else {"informative", "humanitarian", "severity", "disaster_type"}
    )
    predictor = prediction.Predictor.from_run(run_dir, device="cpu")
    predictor.features = Store()
    selected = frame.iloc[:1] if modality == "text" else frame.iloc[:2]
    predictions = predictor.predict_rows(selected)
    assert len(predictions) == len(selected)
    assert predictions.has_text.all() == (modality == "text")
    assert predictions.has_image.all() == (modality == "image")
    if modality == "text":
        assert predictions.severity_status.eq("unavailable").all()
        assert predictions.disaster_type_status.eq("unavailable").all()


def test_interrupted_resume_matches_uninterrupted_training(monkeypatch, tmp_path, tiny_training):
    config, _, _ = tiny_training
    config = replace(
        config, model=replace(config.model, dropout=0.2, head_dropout=0.2), optim=replace(config.optim, max_epochs=3, patience=3)
    )
    uninterrupted = training.run_training(config, output_dir=tmp_path / "reference")
    original = training.train_one_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("simulated session interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "train_one_epoch", interrupted)
    with pytest.raises(InterruptedError):
        training.run_training(config, output_dir=tmp_path / "interrupted")
    resumed = next((tmp_path / "interrupted").iterdir())
    assert torch.load(resumed / "last.pt", weights_only=True)["epoch"] == 1
    # Smaller artifacts can be absent or ahead of the last atomic epoch state.
    (resumed / "train_log.jsonl").write_text("stale partial log")
    (resumed / "metrics_dev.json").write_text("{}")
    monkeypatch.setattr(training, "train_one_epoch", original)
    training.run_training(config, resume_run=resumed)
    reference = torch.load(uninterrupted / "last.pt", weights_only=True)
    actual = torch.load(resumed / "last.pt", weights_only=True)

    def same(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                same(a, b)
        else:
            assert left == right

    for key in ("state_dict", "optimizer", "scheduler", "scaler", "rng", "best_state", "best", "stale"):
        same(reference[key], actual[key])
    for ref, observed in zip(reference["records"], actual["records"], strict=True):
        same(
            {k: v for k, v in ref.items() if k != "epoch_compute_seconds"},
            {k: v for k, v in observed.items() if k != "epoch_compute_seconds"},
        )
    assert len((resumed / "train_log.jsonl").read_text().splitlines()) == 3
    assert json.loads((resumed / "metrics_dev.json").read_text()) == reference["best_metrics"]
    # A completed resume restores artifacts without another training epoch.
    training.run_training(config, resume_run=resumed)
    same(actual["state_dict"], torch.load(resumed / "last.pt", weights_only=True)["state_dict"])


@pytest.mark.parametrize("freeze_name", ["evaluation_request.json", "evaluation_loeo_request.json"])
def test_resume_rejects_changed_config_inputs_and_test_freeze(tmp_path, tiny_training, freeze_name):
    config, frame, _ = tiny_training
    run = training.run_training(config, output_dir=tmp_path / "runs")
    with pytest.raises(ValueError, match="resume config differs"):
        training.run_training(replace(config, seed=9), resume_run=run)
    manifest = training.paths.PROCESSED / "manifest.parquet"
    frame.assign(text="changed").to_parquet(manifest)
    with pytest.raises(ValueError, match="resume inputs differ"):
        training.run_training(config, resume_run=run)
    (run / freeze_name).write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        training.run_training(config, resume_run=run)
    with pytest.raises(ValueError, match="frozen"):
        calibrate_run(run)


def test_overflow_skips_optimizer_and_scheduler_update():
    model, model_config, batch = fixture()
    weights = {name: torch.ones(len(classes)) for name, classes in HEAD_CLASSES.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    class ControlledScaler:
        """Inject one unscaled overflow, then emulate GradScaler's skipped step."""

        value = 2.0
        overflow = True

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            if self.overflow:
                next(model.parameters()).grad.fill_(float("inf"))

        def get_scale(self):
            return self.value

        def step(self, optimizer):
            if not self.overflow:
                optimizer.step()

        def update(self):
            if self.overflow:
                self.value /= 2
                self.overflow = False

    before = [p.detach().clone() for p in model.parameters()]
    config = TrainConfig(model=model_config)
    scaler = ControlledScaler()
    stats = train_one_epoch(model, [batch], optimizer, scheduler, weights, config, torch.device("cpu"), scaler=scaler)
    assert stats["overflow_updates"] == 1 and stats["optimizer_updates"] == 0
    assert scheduler.last_epoch == 0
    for a, b in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    stats = train_one_epoch(model, [batch], optimizer, scheduler, weights, config, torch.device("cpu"), scaler=scaler)
    assert stats["optimizer_updates"] == 1 and scheduler.last_epoch == 1


def test_cuda_precision_uses_native_bf16_support_and_enables_fp16_scaling(monkeypatch):
    enabled = []
    native_bf16 = False

    def supported(*, including_emulation):
        assert including_emulation is False
        return native_bf16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda device, **kwargs: enabled.append(kwargs["enabled"]))
    assert training.precision_for(torch.device("cpu")) == torch.float32
    assert training.precision_for(torch.device("cuda")) == torch.float16
    training.make_scaler(torch.device("cuda"))
    native_bf16 = True
    assert training.precision_for(torch.device("cuda")) == torch.bfloat16
    training.make_scaler(torch.device("cuda"))
    training.make_scaler(torch.device("cpu"))
    assert enabled == [True, False, False]
