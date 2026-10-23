import pandas as pd
import pytest

from sakshi import atomic


def test_failed_replacement_preserves_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("old")

    def fail(temporary):
        temporary.write_text("partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic.write(path, fail)
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_atomic_parquet_and_csv_round_trip(tmp_path):
    frame = pd.DataFrame({"x": [1, 2], "name": ["a", "b"]})
    atomic.parquet(tmp_path / "x.parquet", frame)
    atomic.csv(tmp_path / "x.csv", frame)
    assert pd.read_parquet(tmp_path / "x.parquet").equals(frame)
    assert pd.read_csv(tmp_path / "x.csv").equals(frame)
