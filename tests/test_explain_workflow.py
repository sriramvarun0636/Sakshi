"""Audit freeze, report generation, resumption and downstream packet integration."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_explain import example, prediction

from sakshi import paths
from sakshi.aggregate.packets import load_packets, validate_selected
from sakshi.compute.files import hash_file
from sakshi.explain.captions import CaptionCache
from sakshi.explain.packets import make_packet, validate_packet
from sakshi.explain.reports import interval, write_report
from sakshi.explain.workflow import prediction_record, protocol
from sakshi.types import ProvenanceResult, ProvStatus


def test_packet_roundtrip_matches_wp4_predictions_context_and_privacy(tmp_path):
    item = example()
    provenance = ProvenanceResult(item.post_id, ProvStatus.NOVEL, None, None, None, None, None, None, None, None, False, 0)
    packet = make_packet(item, event=None, posted_at=None, provenance=provenance, explanations=[])
    packet_path = tmp_path / "packets.jsonl"
    packet_path.write_text(json.dumps(packet) + "\n")
    rows = pd.DataFrame({"sample_id": [item.post_id], "event": [None], "posted_at": [pd.NaT], "text": [item.text_clean]})
    rows["posted_at"] = pd.to_datetime(rows.posted_at, utc=True)
    validate_selected(
        {item.post_id},
        load_packets(packet_path, "fixture"),
        rows,
        pd.DataFrame([prediction_record(item)]),
        pd.DataFrame({"post_id": [item.post_id], "status": ["novel"]}),
    )
    rows.loc[0, "posted_at"] = pd.Timestamp("2017-09-21T00:00:00Z")
    packet["posted_at"] = rows.posted_at.iloc[0].isoformat()
    packet_path.write_text(json.dumps(packet) + "\n")
    validate_selected(
        {item.post_id},
        load_packets(packet_path, "fixture"),
        rows,
        pd.DataFrame([prediction_record(item)]),
        pd.DataFrame({"post_id": [item.post_id], "status": ["novel"]}),
    )
    packet["predictions"]["severity"]["p"] = float("nan")
    with pytest.raises(ValueError):
        validate_packet(packet)
    missing = replace(item, pixels=None, prediction=prediction(has_image=False))
    packet = make_packet(missing, event=None, posted_at=None, provenance=provenance, explanations=[])
    assert packet["predictions"]["severity"]["status"] == "unavailable"


def test_optional_caption_cache_marks_generation_and_rejects_duplicates(tmp_path):
    path = tmp_path / "captions.parquet"
    assert CaptionCache(path).get("sample") is None
    rows = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "caption": ["@someone Houses in floodwater https://example.com"],
            "model": ["Qwen/Qwen2.5-VL-7B-Instruct"],
            "prompt_sha256": ["0" * 64],
            "created_at": [pd.Timestamp("2026-10-24T00:00:00Z")],
        }
    )
    rows.to_parquet(path)
    caption = CaptionCache(path).get("sample")
    assert caption["model_generated"] and "@" not in caption["text"] and "http" not in caption["text"]
    pd.concat([rows, rows]).to_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        CaptionCache(path)


def test_bootstrap_and_reports_keep_negative_results_and_undefined_correlations(tmp_path):
    assert interval([1, 2, 3], 100) == interval([1, 2, 3], 100)
    assert interval([]) is None
    with pytest.raises(ValueError):
        interval([np.nan])
    results = [
        {
            "head": "informative",
            "flags": [],
            "stage_flags": {"head": ["ig_unconverged"]},
            "status": "predicted",
            "measurements": [
                {"modality": "text", "stage": "head", "rho": None},
                {"modality": "text", "stage": "gate", "rho": -0.8},
            ],
        }
    ]
    report = write_report("sanity", results, tmp_path, split="dev", count=1, bootstrap=100, heads=("informative",))
    text = report.read_text()
    assert "Development check only" in text and "n/a" in text and "-0.8000" in text and "head:ig_unconverged" in text
    assert "| informative | text | head | 0 | 1 |" in text


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    (processed / "features").mkdir(parents=True)
    monkeypatch.setattr(paths, "PROCESSED", processed)
    manifest = pd.DataFrame({"sample_id": ["a", "b"], "split": ["dev", "test"]})
    manifest.to_parquet(processed / "manifest.parquet")
    run = tmp_path / "run"
    run.mkdir()
    inputs = {"manifest": hash_file(processed / "manifest.parquet")}
    for name in ("features_index.parquet", "clip_vitb16_tokens.f16.npy", "clip_vitb16_embed.f32.npy"):
        (processed / "features" / name).write_bytes(name.encode())
        inputs[name] = hash_file(processed / "features" / name)
    (run / "inputs.json").write_text(json.dumps(inputs))
    for name in ("checkpoint.pt", "config.yaml", "temperatures.json", "thresholds.json", "label_maps.json"):
        (run / name).write_text("fold: null\n" if name == "config.yaml" else "fixture")
    return run, manifest


def test_test_rows_require_classifier_freeze_and_dev_does_not_freeze(frozen_inputs):
    run, manifest = frozen_inputs
    protocol(run, manifest, manifest.iloc[:1])
    assert not (run / "explanation_protocol.json").exists()
    with pytest.raises(ValueError, match="must be frozen"):
        protocol(run, manifest, manifest.iloc[1:])


def test_protocol_freeze_rejects_changed_classifier_and_cache(frozen_inputs, monkeypatch):
    run, manifest = frozen_inputs
    (run / "evaluation_complete.json").write_text("fixture")
    calls = []
    monkeypatch.setattr("sakshi.compute.runner.final_evaluate", lambda path: calls.append(path))
    protocol(run, manifest, manifest.iloc[1:])
    assert calls == [run]
    protocol(run, manifest, manifest.iloc[1:])
    (run / "checkpoint.pt").write_text("changed")
    with pytest.raises(ValueError, match="protocol was frozen"):
        protocol(run, manifest, manifest.iloc[1:])
    (paths.PROCESSED / "features/clip_vitb16_tokens.f16.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="feature cache differs"):
        protocol(run, manifest, manifest.iloc[1:])


def test_loeo_packet_protocol_accepts_only_frozen_heldout_rows(frozen_inputs, monkeypatch):
    run, manifest = frozen_inputs
    manifest["loeo_hurricane_harvey"] = ["test", "excluded"]
    (run / "config.yaml").write_text("fold: loeo_hurricane_harvey\n")
    with pytest.raises(ValueError, match="must be frozen"):
        protocol(run, manifest, manifest.iloc[:1])
    (run / "evaluation_loeo_complete.json").write_text("fixture")
    calls = []
    monkeypatch.setattr("sakshi.models.evaluate_loeo.evaluate_loeo_run", lambda path: calls.append(path))
    protocol(run, manifest, manifest.iloc[:1])
    assert calls == [run] and (run / "explanation_protocol.json").is_file()
    with pytest.raises(ValueError, match="only frozen held-out-event rows"):
        protocol(run, manifest, manifest.iloc[1:])
