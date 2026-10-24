import ast
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from sakshi import paths
from sakshi.compute.bootstrap import prepare_workspace, verify_bundle
from sakshi.compute.bundle import prepare_bundle
from sakshi.compute.files import hash_file
from sakshi.compute.preflight import inspect_inputs
from sakshi.compute.runner import export_run, final_evaluate, restore_export, training_smoke
from sakshi.data.features import EMBED_FILE, INDEX_FILE, TOKENS_FILE
from sakshi.models import data as model_data
from sakshi.models import evaluate_run as evaluation
from sakshi.models.fusion import FusionClassifier, FusionConfig
from sakshi.models.train import OptimSettings, TrainConfig


@pytest.fixture
def prepared(tmp_path):
    processed = tmp_path / "processed"
    features = processed / "features"
    features.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "sample_id": ["train-text", "train-image", "dev", "test"],
            "dataset": ["crisismmd", "cib", "crisismmd", "crisismmd"],
            "split": ["train", "train", "dev", "test"],
            "component": range(4),
            "tweet_id": ["1", None, "3", "4"],
            "text": ["Bridge damaged", None, "Road clear", "Held out text"],
            "y_informative": ["informative", None, "not_informative", "informative"],
            "y_humanitarian": ["infrastructure_and_utility_damage", None, "not_humanitarian", None],
            "y_severity": ["mild_damage", None, None, None],
            "y_disaster_type": [None, "fire", None, None],
            "informative_text": ["informative", None, "not_informative", None],
            "humanitarian_text": ["infrastructure_and_utility_damage", None, "not_humanitarian", None],
            "informative_image": ["informative", None, "not_informative", None],
            "humanitarian_image": ["infrastructure_and_utility_damage", None, "not_humanitarian", None],
        }
    )
    rows.to_parquet(processed / "manifest.parquet")
    pd.DataFrame({"sample_id": rows.sample_id, "row": range(4), "ok": True}).to_parquet(features / INDEX_FILE)
    np.save(features / TOKENS_FILE, np.zeros((4, 50, 768), dtype=np.float16))
    embeds = np.zeros((4, 512), dtype=np.float32)
    embeds[:, 0] = 1
    np.save(features / EMBED_FILE, embeds)
    return processed


@pytest.mark.parametrize("problem", ["component", "tweet_id", "nonfinite", "norm", "dtype", "shape", "sample_id"])
def test_preflight_rejects_corrupt_inputs_and_split_leaks(prepared, problem):
    if problem in ("component", "tweet_id", "sample_id"):
        file = prepared / "manifest.parquet"
        rows = pd.read_parquet(file)
        rows.loc[2, problem] = rows.loc[0, problem]
        rows.to_parquet(file)
    else:
        file = prepared / "features" / (TOKENS_FILE if problem != "norm" else EMBED_FILE)
        values = np.load(file)
        if problem == "nonfinite":
            values[0, 0, 0] = np.nan
        elif problem == "norm":
            values[0] = 0
        elif problem == "dtype":
            values = values.astype(np.float32)
        else:
            values = values[:2]
        np.save(file, values)
    with pytest.raises(ValueError):
        inspect_inputs(TrainConfig(), processed=prepared, strict=False)


def test_preflight_aligns_by_id_and_preserves_missing_image_semantics(prepared):
    frame = pd.read_parquet(prepared / "manifest.parquet")
    frame.iloc[::-1].to_parquet(prepared / "manifest.parquet")
    result = inspect_inputs(TrainConfig(), processed=prepared, strict=False)
    assert result["manifest_rows"] == 4 and result["training_rows"] == 2 and result["dev_rows"] == 1
    index = pd.read_parquet(prepared / "features" / INDEX_FILE)
    index.loc[3, "ok"] = False
    index.to_parquet(prepared / "features" / INDEX_FILE)
    file = prepared / "features" / EMBED_FILE
    embeddings = np.load(file)
    embeddings[3] = 0
    np.save(file, embeddings)
    assert inspect_inputs(TrainConfig(), processed=prepared, strict=False)["usable_images"] == 3
    embeddings[3, 0] = 1
    np.save(file, embeddings)
    with pytest.raises(ValueError, match="zero cache rows"):
        inspect_inputs(TrainConfig(), processed=prepared, strict=False)


@pytest.fixture
def bundle(tmp_path, prepared):
    repo = tmp_path / "source-repo"
    (repo / "src/sakshi/compute").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    bootstrap = Path(__file__).parents[1] / "src/sakshi/compute/bootstrap.py"
    (repo / "src/sakshi/compute/bootstrap.py").write_bytes(bootstrap.read_bytes())
    (repo / ".env").write_text("SECRET=exclude-me")
    (repo / "private.jpg").write_bytes(b"private-image")
    return prepare_bundle(tmp_path / "bundle", processed=prepared, repo=repo, strict=False)


def test_private_bundle_verifies_stages_source_and_rejects_overwrite(bundle, tmp_path):
    manifest = verify_bundle(bundle)
    assert not any(".env" in entry["path"] or "private.jpg" in entry["path"] for entry in manifest["files"])
    workspace = tmp_path / "workspace"
    result = prepare_workspace(bundle, workspace)
    assert result["processed_dir"] == str(bundle / "prepared")
    assert (workspace / "pyproject.toml").is_file()
    assert not (workspace / "prepared").exists()
    assert prepare_workspace(bundle, workspace) == result
    unexpected = workspace / "old_module.py"
    unexpected.write_text("stale source")
    with pytest.raises(FileExistsError, match="outside the source snapshot"):
        prepare_workspace(bundle, workspace)
    unexpected.unlink()
    (workspace / "pyproject.toml").write_text("changed")
    with pytest.raises(FileExistsError, match="differs"):
        prepare_workspace(bundle, workspace)
    with pytest.raises(ValueError, match="outside"):
        prepare_workspace(bundle, bundle / "writable")
    (bundle / "prepared/manifest.parquet").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(bundle)


@pytest.mark.parametrize("name", ["source/../../escape", "source/../escape", "/source/escape", "source/evil\\path"])
def test_bundle_rejects_path_traversal(bundle, name):
    path = bundle / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest["files"].append({"path": name, "bytes": 0, "sha256": "unused"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_training_smoke_exercises_real_updates_for_all_modalities(prepared, monkeypatch, tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = nn.Embedding(10, 8)

        def forward(self, *, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {key: torch.ones(len(texts), 2, dtype=torch.long) for key in ("input_ids", "attention_mask")}

    monkeypatch.setattr(model_data, "load_tokenizer", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(FusionClassifier, "from_pretrained", classmethod(lambda cls, cfg: cls(Encoder(), cfg)))
    model = FusionConfig(
        text_hidden=8,
        image_hidden=768,
        image_tokens=50,
        d_model=16,
        attention_heads=4,
        cross_attention_layers=1,
        ffn_dim=32,
        head_hidden=10,
    )
    config = TrainConfig(model=model, optim=OptimSettings(batch_size=2, grad_accum=2))
    result = training_smoke(config, tmp_path / "smoke.json", processed=prepared, require_cuda=False)
    assert result["device"] == "cpu" and result["peak_cuda_bytes"] is None
    assert set(result["patterns"]) == {"both", "text_only", "image_only"}
    assert all(value["metrics"]["optimizer_updates"] == 1 for value in result["patterns"].values())
    assert all(value["padded_text_tokens"] == 128 for value in result["patterns"].values())
    assert result["patterns"]["text_only"]["metrics"].get("n_severity", 0) == 0
    assert result["patterns"]["image_only"]["metrics"].get("n_informative_text", 0) == 0


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "full-s0-fixture"
    run.mkdir()
    config = TrainConfig()
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(config)))
    for name in (
        "checkpoint.pt",
        "env.json",
        "inputs.json",
        "label_maps.json",
        "train_log.jsonl",
        "training_summary.json",
        "temperatures.json",
        "thresholds.json",
        "last.pt",
    ):
        (run / name).write_text(name)
    return run


def test_final_evaluation_scores_once_and_freezes_inputs(completed_run, prepared, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", prepared)
    calls = []

    def evaluate(run, **kwargs):
        calls.append(kwargs)
        (run / "predictions_official.parquet").write_bytes(b"predictions")
        (run / "metrics_test.json").write_text('{"test": true}')

    monkeypatch.setattr(evaluation, "evaluate_run", evaluate)
    metrics = final_evaluate(completed_run, bootstrap=10)
    assert final_evaluate(completed_run, bootstrap=10) == metrics and len(calls) == 1
    # Recover a crash after metrics publication without running inference again.
    (completed_run / "evaluation_complete.json").unlink()
    assert final_evaluate(completed_run, bootstrap=10) == metrics and len(calls) == 1
    with pytest.raises(FileExistsError, match="changed"):
        final_evaluate(completed_run, bootstrap=11)
    (completed_run / "temperatures.json").write_text("changed")
    with pytest.raises(FileExistsError, match="changed"):
        final_evaluate(completed_run, bootstrap=10)


def test_final_evaluation_rejects_loeo_and_unrecorded_metrics(completed_run, prepared, monkeypatch):
    monkeypatch.setattr(paths, "PROCESSED", prepared)
    (completed_run / "config.yaml").write_text(yaml.safe_dump(asdict(replace(TrainConfig(), fold="loeo_harvey"))))
    with pytest.raises(ValueError, match="LOEO"):
        final_evaluate(completed_run)
    (completed_run / "config.yaml").write_text(yaml.safe_dump(asdict(TrainConfig())))
    (completed_run / "metrics_test.json").write_text("{}")
    with pytest.raises(FileExistsError, match="without a freeze"):
        final_evaluate(completed_run)


@pytest.mark.parametrize("include_resume", [False, True])
def test_export_restore_round_trip_checks_parts_and_excludes_unlisted_data(completed_run, tmp_path, include_resume):
    (completed_run / "private.csv").write_text("not a run artifact")
    exported = tmp_path / "download"
    manifest_path = export_run(completed_run, exported, include_resume=include_resume, part_bytes=128)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["parts"]) > 1
    assert max(part["bytes"] for part in manifest["parts"]) <= 128
    assert ("last.pt" in manifest["members"]) == include_resume
    assert "private.csv" not in manifest["members"]
    restored = restore_export(exported, tmp_path / "restored")
    assert hash_file(restored / "checkpoint.pt") == hash_file(completed_run / "checkpoint.pt")
    with pytest.raises(FileExistsError):
        restore_export(exported, restored)
    first_part = exported / manifest["parts"][0]["path"]
    first_part.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        restore_export(exported, tmp_path / "corrupted")
    assert not (tmp_path / "corrupted").exists()


def test_notebook_is_clean_valid_python_and_bootstraps_without_installed_package(bundle, tmp_path):
    path = Path(__file__).parents[1] / "notebooks/train_classifier_kaggle.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for cell in cells:
        assert cell.execution_count is None and cell.outputs == []
        ast.parse(cell.source)
    parameters = {}
    exec(compile(cells[0].source, str(path), "exec"), parameters)
    assert parameters["FINAL_EVALUATION"] is False
    workspace = tmp_path / "notebook-workspace"
    setup = f"from pathlib import Path\nBUNDLE_ROOT = Path({str(bundle)!r})\nWORK_ROOT = Path({str(workspace)!r})\n"
    verified = "\nassert 'sakshi' not in sys.modules\nassert Path(os.environ['SAKSHI_PROCESSED_DIR']).is_dir()\n"
    subprocess.run([sys.executable, "-I", "-c", setup + cells[1].source + verified], check=True, capture_output=True)
    assert (workspace / "source/pyproject.toml").exists()
