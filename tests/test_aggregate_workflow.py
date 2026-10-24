import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sakshi.aggregate.core import evidence_groups, rollup_samples
from sakshi.aggregate.packets import load_packets, validate_selected
from sakshi.aggregate.ranking import evaluate_window, gold_relevance, ndcg
from sakshi.aggregate.workflow import AggregateConfig, run_aggregation, select_scope


class Embedder:
    calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


@pytest.fixture
def inputs(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "tweet_id": ["ta", "tb", "tc", "td"],
            "dataset": ["crisismmd"] * 4,
            "event": ["hurricane_maria"] * 4,
            "posted_at": pd.to_datetime(["2017-09-21T00:00:00Z"] * 4),
            "text": ["Bridge collapsed in the affected area"] * 4,
            "component": [1, 1, 3, 4],
            "split": ["train", "dev", "test", "test"],
            "loeo_hurricane_maria": ["test"] * 4,
            "y_informative": ["informative"] * 4,
            "y_humanitarian": ["infrastructure_and_utility_damage", "affected_individuals", "other_relevant_information", None],
            "y_severity": ["severe_damage", None, None, None],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": frame.sample_id,
            "model_id": "fixture",
            "informative_label": "informative",
            "humanitarian_label": "infrastructure_and_utility_damage",
            "severity_label": "severe_damage",
            "disaster_type_label": "hurricane",
            "abstain": False,
            "p_informative__informative": 0.9,
            "p_humanitarian__infrastructure_and_utility_damage": [0.8, 0.7, 0.6, 0.5],
            "p_severity__severe_damage": 0.8,
            "p_disaster_type__hurricane": 0.8,
        }
    )
    provenance = pd.DataFrame({"post_id": frame.sample_id, "status": ["novel", "novel", "novel", "recycled"]})
    manifest_path, prediction_path = tmp_path / "manifest.parquet", tmp_path / "predictions.parquet"
    frame.to_parquet(manifest_path)
    predictions.to_parquet(prediction_path)
    provenance.to_parquet(tmp_path / "replay_hurricane_maria.parquet")
    run = tmp_path / "classifier"
    run.mkdir()
    (run / "env.json").write_text(json.dumps({"model_id": "fixture"}))
    (run / "config.yaml").write_text(yaml.safe_dump({"fold": "loeo_hurricane_maria"}))
    config = AggregateConfig(
        model_id="fixture",
        event="hurricane_maria",
        mode="demo",
        manifest=str(manifest_path),
        predictions=str(prediction_path),
        provenance_dir=str(tmp_path),
        output_root=str(tmp_path / "aggregates"),
        report_dir=str(tmp_path / "reports"),
    )
    return frame, predictions, provenance, config, run


def packet_rows(frame, predictions, provenance):
    packets = []
    for index, row in frame.iterrows():
        pred = predictions.iloc[index]
        heads = {}
        for head in ("informative", "humanitarian", "severity", "disaster_type"):
            label = pred[f"{head}_label"]
            heads[head] = {"label": label, "p": round(float(pred[f"p_{head}__{label}"]), 2), "status": "predicted"}
        packets.append(
            {
                "schema_version": "1.0",
                "post_id": row.sample_id,
                "model_id": "fixture",
                "event": row.event,
                "posted_at": row.posted_at.isoformat(),
                "has_text": True,
                "has_image": True,
                "text_clean": "Bridge collapsed in the affected area",
                "image_caption": None,
                "predictions": heads,
                "evidence": [],
                "provenance": {"status": provenance.iloc[index].status},
                "modality_weights": {"text": 0.5, "image": 0.5},
                "abstain": False,
            }
        )
    return packets


def test_scope_requires_correct_fold_and_holds_out_main_model_rows(inputs):
    frame, _, _, config, run = inputs
    real = replace(config, mode="loeo", classifier_run=str(run))
    assert len(select_scope(frame, real)) == 4
    (run / "config.yaml").write_text("fold: loeo_hurricane_harvey\n")
    with pytest.raises(ValueError, match="fold does not match"):
        select_scope(frame, real)
    (run / "config.yaml").write_text("fold: null\n")
    assert select_scope(frame, replace(real, mode="heldout")).sample_id.tolist() == ["c", "d"]
    with pytest.raises(ValueError, match="classifier_run"):
        select_scope(frame, replace(config, mode="loeo"))


def test_demo_workflow_uses_all_images_excludes_recycled_and_preserves_outputs(inputs):
    _, _, _, config, _ = inputs
    embedder = Embedder()
    marker = run_aggregation(config, embedder=embedder)
    assert json.loads(marker.read_text())["evidence_ready"] is False
    aggregate = json.loads((marker.parent / "20170921T000000Z.json").read_text())
    assert aggregate["counts"]["tweets"] == 4
    assert aggregate["counts"]["excluded_provenance"]["recycled"] == 1
    assert aggregate["counts"]["severity_eligible"]["severe_damage"] == 3
    assert len(aggregate["selected"]) == 2  # all texts are one evidence group
    ranking = json.loads((marker.parent / "ranking.json").read_text())[0]
    assert ranking["candidates"] == 3 and ranking["unlabelled"] == 0
    previous = embedder.calls
    assert run_aggregation(config, embedder=embedder) == marker and embedder.calls == previous
    assert (marker.parent / "env.json").is_file()
    assert (marker.parent / "inputs.json").is_file()
    assert "demonstration" in (marker.parent / "ranking.md").read_text()
    assert (Path(config.report_dir) / "ranking_demo.md").is_file()
    assert not (Path(config.report_dir) / "ranking.md").exists()
    changed = pd.read_parquet(config.predictions)
    changed.loc[0, "p_informative__informative"] = 0.8
    changed.to_parquet(config.predictions)
    with pytest.raises(FileExistsError, match="input artifacts changed"):
        run_aggregation(config, embedder=embedder)


def test_research_requires_valid_same_model_packets_and_detects_stale_content(inputs, tmp_path):
    frame, predictions, provenance, config, run = inputs
    packet_path = tmp_path / "packets.jsonl"
    real = replace(config, mode="loeo", classifier_run=str(run), packets=str(packet_path))
    with pytest.raises(FileNotFoundError):
        run_aggregation(real, embedder=Embedder())
    records = packet_rows(frame, predictions, provenance)
    packet_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    marker = run_aggregation(real, embedder=Embedder())
    assert json.loads(marker.read_text())["evidence_ready"] is True
    packets = load_packets(packet_path, "fixture")
    packets["a"]["text_clean"] = "A different clean claim"
    with pytest.raises(ValueError, match="text differs"):
        validate_selected({"a"}, packets, frame, predictions, provenance)
    packets["a"]["text_clean"] = records[0]["text_clean"]
    packets["a"]["predictions"]["severity"]["p"] = 0.1
    with pytest.raises(ValueError, match="probability differs"):
        validate_selected({"a"}, packets, frame, predictions, provenance)
    records[0]["text_clean"] = "Please call @someone https://example.com"
    packet_path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="handle, URL"):
        load_packets(packet_path, "fixture")


def test_ndcg_hand_calculation_and_unknown_label_accounting(inputs):
    frame, predictions, provenance, _, _ = inputs
    assert ndcg([1, 3], k=2) == pytest.approx((1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3)))
    assert ndcg([0, 0], k=10) is None
    assert ndcg([], k=25) is None
    with pytest.raises(ValueError):
        ndcg([4], k=10)
    gold = gold_relevance(frame)
    assert gold == {"ta": 3, "tb": 3, "tc": 1}
    tweets = rollup_samples(frame, predictions, provenance.assign(status="novel"))
    clusters = {tweet.tweet_id: 0 for tweet in tweets}
    groups = evidence_groups(tweets)
    result = evaluate_window(tweets, clusters, groups, gold)
    assert result["candidates"] == 3 and result["unlabelled"] == 1
    assert result["random_seeds"] == list(range(10))
    assert result == evaluate_window(tweets, clusters, groups, gold)
    assert set(result["metrics"]) == {"priority", "random", "recency", "confidence"}


def test_raw_text_deduplication_ignores_url_presence_and_display_removes_handles(inputs):
    frame, predictions, provenance, _, _ = inputs
    frame.loc[0, "text"] = "Bridge collapsed in the affected area https://example.com @someone"
    frame.loc[1, "text"] = "Bridge collapsed in the affected area"
    frame.loc[0, "component"] = 10
    frame.loc[1, "component"] = 11
    tweets = rollup_samples(frame, predictions, provenance)
    assert "@someone" not in tweets[0].text_clean and "https" not in tweets[0].text_clean
    groups = evidence_groups(tweets)
    assert groups["ta"] == groups["tb"]


def test_packets_reject_nan_and_wrong_model(inputs, tmp_path):
    frame, predictions, provenance, _, _ = inputs
    records = packet_rows(frame, predictions, provenance)
    packet_path = tmp_path / "packets.jsonl"
    packet_path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="model_id differs"):
        load_packets(packet_path, "another_model")
    records[0]["predictions"]["severity"]["p"] = float("nan")
    packet_path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        load_packets(packet_path, "fixture")
