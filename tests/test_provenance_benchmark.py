from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sakshi.config import load
from sakshi.provenance.benchmark import (
    BenchmarkConfig,
    Outcome,
    _write_report,
    ambiguous_ids,
    auprc,
    injection_plan,
    positive_pool,
    score_case,
)
from sakshi.types import ProvStatus


class MiniIndex:
    def __init__(self):
        start = pd.Timestamp("2017-09-20T00:00:00Z")
        self.onsets = {"hurricane_maria": start}
        self.frame = pd.DataFrame(
            {
                "sample_id": ["old", "graphic", "same", "future", "degenerate"],
                "first_seen": pd.to_datetime(
                    [
                        "2017-09-01T00:00:00Z",
                        "2017-09-01T00:00:00Z",
                        "2017-09-01T00:00:00Z",
                        "2017-10-01T00:00:00Z",
                        "2017-09-01T00:00:00Z",
                    ]
                ),
                "index_event": ["hurricane_irma", "hurricane_irma", "hurricane_maria", "hurricane_irma", "hurricane_irma"],
                "degenerate": [False, False, False, False, True],
                "is_graphic": [False, True, False, False, False],
            }
        )
        self.manifest = pd.DataFrame(
            {
                "sample_id": ["a", "b", "other"],
                "dataset": ["crisismmd"] * 3,
                "event": ["hurricane_maria", "hurricane_maria", "hurricane_irma"],
                "component": [1, 2, 1],
                "posted_at": pd.to_datetime(["2017-09-21T00:00:00Z", "2017-09-22T00:00:00Z", "2017-09-10T00:00:00Z"]),
            }
        ).set_index("sample_id", drop=False)


def test_injection_plan_respects_pool_time_and_fixed_seed():
    index = MiniIndex()
    assert positive_pool(index, "hurricane_maria") == ["old"]
    assert ambiguous_ids(index) == {"a", "other"}
    first = injection_plan(index, "hurricane_maria", rate=0.2, seed=3)
    assert first == injection_plan(index, "hurricane_maria", rate=0.2, seed=3)
    assert len(first) == 0  # round(0.2 * 2 / 0.8) uses nearest-even rounding
    plan = injection_plan(index, "hurricane_maria", rate=0.34, seed=3)
    assert len(plan) == 1 and plan[0].source_id == "old"
    assert plan[0].query_id == "inj:hurricane_maria:3:0"
    assert datetime(2017, 9, 21, tzinfo=timezone.utc) <= plan[0].time <= datetime(2017, 9, 22, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="pool"):
        injection_plan(index, "hurricane_maria", rate=0.9, seed=3)


def test_scoring_excludes_ambiguous_genuine_and_requires_origin():
    genuine = [
        Outcome(ProvStatus.RECYCLED, 1.0, ambiguous=True),
        Outcome(ProvStatus.RECYCLED, 2.0, hard_negative=True),
        Outcome(ProvStatus.TEMPLATE_MATCH, 3.0, hard_negative=True),
    ]
    injected = [Outcome(ProvStatus.RECYCLED, 4.0, origin_correct=True), Outcome(ProvStatus.NOVEL, 5.0)]
    result = score_case(genuine, injected)
    assert result["tp"] == 1 and result["fp"] == 1
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    assert result["fpr_genuine"] == 0.5 and result["recall_at_origin"] == 0.5
    assert result["hard_negative_accuracy"] == 0.5 and result["review_flag_rate"] == pytest.approx(1 / 3)
    assert result["mean_latency_ms"] == 3.0
    assert np.isfinite(auprc([result]))


def test_benchmark_config_and_report_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    config = load("configs/provenance/benchmark.yaml", BenchmarkConfig)
    assert len(config.tau_grid.values()) == 39
    stats = score_case([Outcome(ProvStatus.NOVEL, 1.0)], [Outcome(ProvStatus.RECYCLED, 2.0, origin_correct=True)])
    cases = [{"event": "hurricane_maria", "transform": name, "rate": 0.05, "seed": 0, **stats} for name in config.transforms]
    curves = {name: [{"tau_clip": 0.92, "precision": 1.0, "recall": 1.0}] for name in config.transforms}
    output = tmp_path / "reports" / "provenance"
    _write_report(config, 0.92, {0.92: [1.0]}, cases, curves, output)
    assert (output / "benchmark.json").is_file() and (output / "benchmark.md").is_file()
    figures = tmp_path / "reports" / "figures"
    assert len(list(figures.glob("prov_*.png"))) == 10
