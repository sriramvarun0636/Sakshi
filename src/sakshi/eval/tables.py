"""Small deterministic Markdown tables for machine-generated results reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _field(row: Mapping[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"result is missing table field {key!r}")
        value = value[part]
    return value


def _cell(value: Any, decimals: int | None) -> str:
    if value is None:
        return "—"
    result = f"{value:.{decimals}f}" if isinstance(value, float) and decimals is not None else str(value)
    return result.replace("|", "\\|").replace("\n", " ")


def render(results: list[dict], spec: dict) -> str:
    """Render rows using explicit ordered columns and optional decimal formatting.

    ``spec`` has ``title`` and ``columns``; each column has ``key``, ``header``,
    and optionally ``decimals``. Dotted keys read nested result objects.
    """
    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("table spec must define nonempty columns")
    for column in columns:
        if not isinstance(column, dict) or not {"key", "header"} <= set(column):
            raise ValueError("each table column needs key and header")
    lines = []
    if title := spec.get("title"):
        lines += [f"## {title}", ""]
    lines.append("| " + " | ".join(_cell(c["header"], None) for c in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in results:
        lines.append("| " + " | ".join(_cell(_field(row, c["key"]), c.get("decimals")) for c in columns) + " |")
    return "\n".join(lines) + "\n"
