"""The dashboard selects completed captioned research artifacts when available."""

import json

from sakshi.app import data


def _artifact(root, folder, event, *, sitrep):
    directory = root / "runs" / folder / "model" / event
    directory.mkdir(parents=True)
    window = directory / "20170827T000000Z.json"
    window.write_text(
        json.dumps({"event": event, "model_id": "model", "models": {"classifier": "model"}}) + "\n"
    )
    (directory / "complete.json").write_text(
        json.dumps({"windows": 1, "mode": "loeo", "evidence_ready": True}) + "\n"
    )
    if not sitrep:
        (directory / "tweets.parquet").touch()
    return directory


def test_captioned_artifacts_take_priority(tmp_path):
    event = "hurricane_harvey"
    _artifact(tmp_path, "aggregate", event, sitrep=False)
    captioned_aggregate = _artifact(tmp_path, "aggregate_captioned", event, sitrep=False)
    _artifact(tmp_path, "sitreps_hf", event, sitrep=True)
    captioned_sitrep = _artifact(tmp_path, "sitreps_hf_captioned", event, sitrep=True)
    assert data.preferred_aggregates(tmp_path)[event].directory == captioned_aggregate
    assert data.sitreps(tmp_path)[event].directory == captioned_sitrep


def test_live_dashboard_selects_frozen_full_classifier(tmp_path):
    base = tmp_path / "runs" / "classifier"
    full = base / "full-s0-20261024-062724"
    full.mkdir(parents=True)
    (full / "checkpoint.pt").touch()
    (full / "evaluation_complete.json").touch()
    ablation = base / "b4_disaster_type-s2-20261026-103346"
    ablation.mkdir()
    (ablation / "checkpoint.pt").touch()
    incomplete = base / "full-s0-20261027-000000"
    incomplete.mkdir()
    (incomplete / "checkpoint.pt").touch()

    assert data.classifier_run(tmp_path) == full
