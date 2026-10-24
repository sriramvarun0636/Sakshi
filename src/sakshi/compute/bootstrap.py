"""Standard-library-only verification and source staging before notebook installation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in name or relative.as_posix() != name:
        raise ValueError(f"unsafe bundle member {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"bundle member escapes its root: {name}")
    return path


def verify_bundle(root: Path) -> dict:
    """Verify every listed source/cache byte before imports, pip or model loading."""
    root = Path(root)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported training bundle format")
    members = manifest["files"]
    names = [entry["path"] for entry in members]
    if len(set(names)) != len(names):
        raise ValueError("bundle contains duplicate paths")
    required = {
        "prepared/manifest.parquet",
        "prepared/features/features_index.parquet",
        "prepared/features/clip_vitb16_tokens.f16.npy",
        "prepared/features/clip_vitb16_embed.f32.npy",
        "source/pyproject.toml",
        "source/source_provenance.json",
        "source/src/sakshi/compute/bootstrap.py",
    }
    if required - set(names):
        raise ValueError("bundle is missing required prepared inputs or source files")
    for entry in members:
        if not entry["path"].startswith(("prepared/", "source/")):
            raise ValueError("bundle member is outside prepared/ or source/")
        path = _member(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _hash(path) != entry["sha256"]:
            raise ValueError(f"bundle integrity check failed: {entry['path']}")
    return manifest


def prepare_workspace(bundle_root: Path, source_dir: Path) -> dict:
    """Copy verified source to a writable workspace while leaving prepared inputs mounted."""
    bundle_root, source_dir = Path(bundle_root), Path(source_dir)
    manifest = verify_bundle(bundle_root)
    if source_dir.resolve().is_relative_to(bundle_root.resolve()):
        raise ValueError("writable source directory must be outside the mounted input bundle")
    expected = {entry["path"].removeprefix("source/") for entry in manifest["files"] if entry["path"].startswith("source/")}
    for existing in source_dir.rglob("*"):
        if (
            existing.is_file()
            and existing.relative_to(source_dir).as_posix() not in expected
            and "__pycache__" not in existing.parts
            and not any(part.endswith(".egg-info") for part in existing.parts)
        ):
            raise FileExistsError(f"workspace contains a file outside the source snapshot: {existing}")
    for entry in manifest["files"]:
        if not entry["path"].startswith("source/"):
            continue
        relative = entry["path"].removeprefix("source/")
        destination = _member(source_dir, relative)
        if destination.exists():
            if not destination.is_file() or _hash(destination) != entry["sha256"]:
                raise FileExistsError(f"workspace source differs from the attached bundle: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_member(bundle_root, entry["path"]), destination)
    return {
        "source_dir": str(source_dir.resolve()),
        "processed_dir": str((bundle_root / "prepared").resolve()),
        "source": manifest["source"],
        "bundle_sha256": _hash(bundle_root / "bundle.json"),
    }
