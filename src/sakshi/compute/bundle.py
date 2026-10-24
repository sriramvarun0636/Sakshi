"""Create private, checksum-verified source and cached-feature input bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sakshi import atomic, paths
from sakshi.compute.bootstrap import verify_bundle
from sakshi.compute.files import hash_file
from sakshi.config import environment

PREPARED_FILES = (
    "manifest.parquet",
    "features/features_index.parquet",
    "features/clip_vitb16_tokens.f16.npy",
    "features/clip_vitb16_embed.f32.npy",
)
SOURCE_EXTENSIONS = {".py", ".toml", ".yaml", ".json", ".md", ".ipynb"}


def source_files(repo: Path) -> list[Path]:
    """Use an allowlist so tokens, images, caches, Git internals and runs stay out of code."""
    result = [repo / name for name in ("pyproject.toml", "README.md", "SAKSHI_design_spec_v2.md") if (repo / name).is_file()]
    for folder in ("src/sakshi", "configs", "notebooks", "tests", "docs"):
        result.extend(
            path
            for path in (repo / folder).rglob("*")
            if path.is_file() and path.suffix in SOURCE_EXTENSIONS and "__pycache__" not in path.parts
        )
    for path in result:
        if not path.resolve().is_relative_to(repo.resolve()) or path.is_symlink():
            raise ValueError(f"source snapshot refuses symbolic or external files: {path}")
    return sorted(set(result))


def prepare_bundle(output: Path, *, processed: Path = paths.PROCESSED, repo: Path = paths.REPO, strict: bool = True) -> Path:
    """Snapshot only code and four prepared inputs; publish bundle.json after validation."""
    from sakshi.compute.preflight import inspect_inputs
    from sakshi.models.train import TrainConfig

    output, processed, repo = Path(output), Path(processed), Path(repo)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("bundle output is not empty; choose a new directory to preserve previous snapshots")
    if output.resolve().is_relative_to(processed.resolve()):
        raise ValueError("bundle output must be outside the prepared input directory")
    report = inspect_inputs(TrainConfig(), processed=processed, strict=strict, scan_arrays=True)
    sources = source_files(repo)
    needed = sum((processed / name).stat().st_size for name in PREPARED_FILES) + sum(path.stat().st_size for path in sources)
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < needed + 1024**3:
        raise OSError("bundle preparation needs space for copied inputs plus 1 GiB headroom")
    metadata = environment(model_id="gpu-input-bundle", seed=0)
    source = {key: metadata[key] for key in ("git_commit", "git_dirty")}
    records = []
    for path, relative in [
        *((processed / name, f"prepared/{name}") for name in PREPARED_FILES),
        *((path, "source/" + path.relative_to(repo).as_posix()) for path in sources),
    ]:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        digest = hash_file(path)
        if digest != hash_file(destination):
            raise OSError(f"bundle copy did not reproduce source bytes: {relative}")
        records.append({"path": relative, "bytes": destination.stat().st_size, "sha256": digest})
    provenance = output / "source/source_provenance.json"
    atomic.text(provenance, json.dumps(source, indent=2) + "\n")
    records.append({"path": "source/source_provenance.json", "bytes": provenance.stat().st_size, "sha256": hash_file(provenance)})
    atomic.text(output / "SHA256SUMS", "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in records))
    atomic.text(output / "preflight.json", json.dumps(report, indent=2) + "\n")
    manifest = {
        "format_version": 1,
        "source": source,
        "files": records,
        "usage": "Private team training inputs; keep the mounted dataset and notebook private.",
    }
    atomic.text(output / "bundle.json", json.dumps(manifest, indent=2) + "\n")
    verify_bundle(output)
    return output
