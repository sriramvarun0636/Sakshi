import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sakshi.data import labels
from sakshi.models.fusion import FusionClassifier, FusionConfig
from sakshi.models.predict import Predictor


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Tokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.ones(len(texts), 2, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Store:
    def has_image(self, sample_id):
        return sample_id != "text_only"

    def tokens(self, sample_id):
        return np.ones((3, 8), dtype=np.float32) if self.has_image(sample_id) else np.zeros((3, 8), dtype=np.float32)


def test_manifest_prediction_file_contract_and_invariants(tmp_path):
    config = FusionConfig(
        text_hidden=8,
        image_hidden=8,
        image_tokens=3,
        d_model=16,
        attention_heads=4,
        cross_attention_layers=1,
        ffn_dim=32,
        head_hidden=10,
        dropout=0,
    )
    predictor = Predictor(
        FusionClassifier(TinyEncoder(), config),
        Tokenizer(),
        model_id="tiny",
        device=torch.device("cpu"),
        features=Store(),
    )
    rows = pd.DataFrame({"sample_id": ["both", "image_only", "text_only"], "text": ["Road blocked", pd.NA, "Help needed"]})
    result = predictor.predict_rows(rows, batch_size=2)
    assert result.sample_id.tolist() == rows.sample_id.tolist()
    assert result.has_text.tolist() == [True, False, True]
    assert result.has_image.tolist() == [True, True, False]
    assert result.alpha_image.iloc[1] == pytest.approx(1)
    assert result.alpha_image.iloc[2] == pytest.approx(0)
    assert result.severity_status.iloc[2] == "unavailable"
    assert result.disaster_type_status.iloc[2] == "unavailable"
    assert len([name for name in result if name.startswith("p_humanitarian__")]) == len(labels.HUMANITARIAN)
    for _, row in result.iterrows():
        if row.informative_label == "not_informative":
            assert row.humanitarian_label == "not_humanitarian"
            assert row.severity_status in ("gated", "unavailable")
    output = predictor.write_rows(rows, tmp_path / "predictions.parquet")
    assert pd.read_parquet(output).sample_id.tolist() == rows.sample_id.tolist()
    with torch.no_grad():
        classifier = predictor.model.heads["informative"][-1]
        classifier.weight.zero_()
        classifier.bias.copy_(torch.tensor([5.0, -5.0]))
    gated = predictor.predict_rows(rows.iloc[:1]).iloc[0]
    assert gated.severity_status == "gated"
    assert sum(gated[f"p_severity__{name}"] for name in labels.SEVERITY) == pytest.approx(1)
    with pytest.raises(ValueError, match="duplicate"):
        predictor.predict_rows(pd.concat([rows, rows.iloc[:1]]))


def test_loading_rejects_wrong_label_order_before_model_download(tmp_path):
    (tmp_path / "config.yaml").write_text("name: full\n", encoding="utf-8")
    (tmp_path / "label_maps.json").write_text(json.dumps({"informative": list(reversed(labels.INFORMATIVE))}), encoding="utf-8")
    with pytest.raises(ValueError, match="label maps"):
        Predictor.from_run(tmp_path)
