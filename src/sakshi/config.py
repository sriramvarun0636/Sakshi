"""Strict experiment configs, deterministic seeds and reproducible run directories."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypeVar, get_type_hints

import numpy as np

from sakshi import paths

T = TypeVar("T")


def _construct(cls: type[T], values: dict, location: str) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass type")
    known = {field.name for field in fields(cls)}
    if extra := set(values) - known:
        raise ValueError(f"unknown config keys at {location}: {sorted(extra)}")
    hints = get_type_hints(cls)
    kwargs = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        field_type = hints[field.name]
        if is_dataclass(field_type):
            if not isinstance(value, dict):
                raise TypeError(f"{location}.{field.name} must be a mapping")
            value = _construct(field_type, value, f"{location}.{field.name}")
        kwargs[field.name] = value
    return cls(**kwargs)


def load(path: str | Path, cls: type[T]) -> T:
    """Load YAML into a dataclass, rejecting unknown keys at every nested level."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configs require the 'ml' extra: pip install -e '.[ml]'") from exc
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return _construct(cls, values, cls.__name__)


def seed_everything(seed: int):
    """Seed Python, NumPy and every available PyTorch backend; return a data-loader generator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(_worker_id: int) -> None:
    """Seed Python and NumPy workers from PyTorch's seeded worker stream."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=paths.REPO, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment(*, model_id: str, seed: int, device: str | None = None, run_id: str | None = None) -> dict:
    """Capture reproducibility metadata for classifier and downstream experiment runs."""
    packages = {}
    for package in ("numpy", "pandas", "torch", "transformers", "sentence-transformers", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    commit = _git(["rev-parse", "HEAD"])
    snapshot_path = paths.REPO / "source_provenance.json"
    snapshot = json.loads(snapshot_path.read_text()) if commit is None and snapshot_path.exists() else None
    return {
        "git_commit": commit if snapshot is None else snapshot["git_commit"],
        "git_dirty": _git(["status", "--porcelain"]) != "" if snapshot is None else snapshot["git_dirty"],
        "source_snapshot": snapshot,
        "python": sys.version.split()[0],
        "packages": packages,
        "device": device,
        "seed": seed,
        "model_id": model_id,
        "run_id": run_id,
    }


def create_run_dir(
    config: object,
    *,
    name: str,
    seed: int,
    fold: str | None = None,
    root: Path = paths.REPO / "runs" / "classifier",
    device: str | None = None,
    now: datetime | None = None,
) -> tuple[Path, str]:
    """Create one immutable run directory with resolved YAML and environment metadata."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    if not name or any(c in name for c in "/\\ "):
        raise ValueError("name must be a nonempty path-safe config name")
    if fold is not None and (not fold or any(c in fold for c in "/\\ ")):
        raise ValueError("fold must be path-safe")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("run metadata requires the 'ml' extra: pip install -e '.[ml]'") from exc
    model_id = f"{name}-s{seed}" + (f"-{fold}" if fold else "")
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{model_id}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
        env = environment(model_id=model_id, seed=seed, device=device, run_id=run_dir.name)
        (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # A failed metadata write leaves an obviously incomplete directory; callers can
        # inspect it, and subsequent runs refuse to silently overwrite it.
        raise
    return run_dir, model_id
