from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_training_run_writes_checkpoint_and_metadata_without_model_download(monkeypatch, tmp_path):
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
    monkeypatch.setattr(training, "load_tokenizer", lambda name: Tokenizer())
    monkeypatch.setattr(training.FusionClassifier, "from_pretrained", classmethod(lambda cls, cfg: cls(TinyEncoder(), cfg)))
    _, model_config, _ = fixture()
    config = TrainConfig(
        model=model_config,
        data=training.DataSettings(samples_per_epoch=4),
        optim=OptimSettings(batch_size=2, max_epochs=1, patience=1),
    )
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
    monkeypatch.setattr(prediction, "load_tokenizer", lambda name: Tokenizer())
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
