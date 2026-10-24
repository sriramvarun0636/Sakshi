"""Read optional model-generated captions without inventing or regenerating missing ones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sakshi.aggregate.packets import _privacy
from sakshi.text import clean_for_display


class CaptionCache:
    def __init__(self, path: Path):
        self.rows = {}
        if not path.exists():
            return
        frame = pd.read_parquet(path)
        required = {"sample_id", "caption", "model", "prompt_sha256", "created_at"}
        if not required.issubset(frame) or frame.sample_id.duplicated().any():
            raise ValueError("caption cache needs the §9.4 columns and unique sample IDs")
        for row in frame.itertuples():
            if not isinstance(row.caption, str) or not row.caption.strip() or not isinstance(row.model, str):
                raise ValueError("caption text and model must be nonempty strings")
            if not isinstance(row.prompt_sha256, str) or len(row.prompt_sha256) != 64:
                raise ValueError("caption prompt identity is missing")
            value = {"text": clean_for_display(row.caption), "model": row.model, "model_generated": True}
            _privacy(value)
            self.rows[str(row.sample_id)] = value

    def get(self, sample_id: str) -> dict | None:
        return self.rows.get(sample_id)
