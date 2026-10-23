import pandas as pd
import pytest

from sakshi.provenance.labelling import _pairs, export_labels, score_labels, select_pairs


def test_replay_decision_takes_priority_and_group_selection_is_stable():
    index = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "component": [1, 1, 2, 3],
            "is_graphic": [False, False, True, True],
        }
    )
    candidates = pd.DataFrame({"a": ["a", "c"], "b": ["b", "d"]})
    replay = pd.DataFrame(
        {
            "post_id": ["b", "d"],
            "origin_post_id": ["a", "c"],
            "status": ["recycled", "template_match"],
            "tier": ["clip", "phash"],
        }
    )
    pairs = _pairs(candidates, replay, index)
    selected = select_pairs(pairs, graphic_sample=1, seed=4)
    assert set(selected.detector_status) == {
        "recycled",
        "template_match",
    }
    assert selected.both_graphic.sum() == 1
    assert selected.pair_id.is_unique
    assert selected.equals(select_pairs(pairs, graphic_sample=1, seed=4))


def test_label_export_refuses_to_overwrite_human_work(tmp_path):
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"label_a": ["recycled_photo"], "label_b": [""], "final": [""]}).to_csv(sheet, index=False)
    with pytest.raises(FileExistsError, match="human labels"):
        export_labels(sheet_path=sheet)


def test_adjudication_requires_complete_valid_labels_and_reports_metrics(tmp_path):
    sheet = tmp_path / "sheet.csv"
    metadata = tmp_path / "metadata.parquet"
    report = tmp_path / "study.md"
    pd.DataFrame(
        {
            "pair_id": ["1", "2", "3"],
            "label_a": ["recycled_photo", "template_or_placeholder", "false_match"],
            "label_b": ["recycled_photo", "template_or_placeholder", "false_match"],
            "final": ["recycled_photo", "template_or_placeholder", "false_match"],
        }
    ).to_csv(sheet, index=False)
    pd.DataFrame(
        {
            "pair_id": ["1", "2", "3"],
            "detector_status": ["recycled", "template_match", "candidate_unflagged"],
        }
    ).to_parquet(metadata)
    score_labels(sheet, metadata, report)
    content = report.read_text()
    assert "κ between the two independent raters: 1.000" in content
    assert "Precision of recycled/concurrent flags: 1.0" in content
    assert "Template precision: 1.0; template recall: 1.0" in content
    frame = pd.read_csv(sheet, keep_default_na=False)
    frame.loc[0, "final"] = ""
    frame.to_csv(sheet, index=False)
    with pytest.raises(ValueError, match="final needs one valid label"):
        score_labels(sheet, metadata, report)
