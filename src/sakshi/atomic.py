"""Publish generated files only after their complete replacement has been written."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def write(path: Path, writer: Callable[[Path], None]) -> None:
    """Run a writer on a sibling temporary file, then atomically replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text."""
    write(path, lambda temporary: temporary.write_text(content, encoding="utf-8"))


def parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a Parquet data frame."""
    write(path, lambda temporary: frame.to_parquet(temporary, index=False))


def csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a CSV data frame."""
    write(path, lambda temporary: frame.to_csv(temporary, index=False))
