import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from sakshi import config


@dataclass(frozen=True)
class Optim:
    lr: float = 0.1


@dataclass(frozen=True)
class Run:
    name: str = "full"
    optim: Optim = Optim()


def test_config_rejects_unknown_nested_keys(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "run.yaml"
    path.write_text("name: trial\noptim:\n  lr: 0.2\n")
    result = config.load(path, Run)
    assert result.name == "trial" and result.optim.lr == 0.2
    path.write_text("optim:\n  momentum: 0.9\n")
    with pytest.raises(ValueError, match="optim"):
        config.load(path, Run)


def test_seeds_and_run_directory_are_reproducible(tmp_path):
    pytest.importorskip("yaml")
    config.seed_everything(42)
    one = np.random.random()
    config.seed_everything(42)
    assert np.random.random() == one
    path, model_id = config.create_run_dir(
        Run(),
        name="full",
        seed=0,
        fold="loeo_harvey",
        root=tmp_path,
        now=datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc),
    )
    assert model_id == "full-s0-loeo_harvey"
    assert (path / "config.yaml").exists() and (path / "env.json").exists()
    with pytest.raises(FileExistsError):
        config.create_run_dir(
            Run(), name="full", seed=0, fold="loeo_harvey", root=tmp_path, now=datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        )


def test_source_snapshot_preserves_original_identity_without_git(monkeypatch, tmp_path):
    monkeypatch.setattr(config.paths, "REPO", tmp_path)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(config.subprocess, "run", no_git)
    identity = {"git_commit": "original-commit", "git_dirty": True}
    (tmp_path / "source_provenance.json").write_text(json.dumps(identity))
    environment = config.environment(model_id="full-s0", seed=0)
    assert environment["git_commit"] == "original-commit"
    assert environment["git_dirty"] is True and environment["source_snapshot"] == identity
