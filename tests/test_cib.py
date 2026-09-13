import pandas as pd
import pytest

from sakshi.data import cib

_HARVEY_TWEET = "905960092822003712"  # 2017-09-08
_DMD = "data/multimodal-deep-learning-disaster-response-mouzannar/multimodal"


def _row(task, split, path, image_id, label):
    return {"event_name": "x", "image_id": image_id, "image_path": path, "class_label": label, "task": task, "split": split}


def _rows(folder="harvey"):
    aidr = f"data/aidr_disaster_types/{folder}/8_9_2017/{_HARVEY_TWEET}_0.jpg"
    dmd = f"{_DMD}/fires/images/disasters_2017-11-20_20-04-24.jpg"
    return pd.DataFrame(
        [
            _row("disaster_types", "train", aidr, f"{_HARVEY_TWEET}_0", "flood"),
            _row("disaster_types", "test", dmd, "fires/images/disasters_2017-11-20_20-04-24.jpg", "fire"),
            _row("damage", "train", dmd, "fires/images/disasters_2017-11-20_20-04-24.jpg", "severe"),
        ]
    )


def test_manifest_maps_events_labels_and_times():
    m = cib.build_manifest(_rows()).set_index("source")
    a, d = m.loc["aidr_dt"], m.loc["dmd"]
    assert a.event == "hurricane_harvey"
    assert a.y_disaster_type == "flood"
    assert a.tweet_id == _HARVEY_TWEET
    assert a.posted_at.date().isoformat() == "2017-09-08"
    assert a.collection_date.date().isoformat() == "2017-09-08"
    assert pd.isna(d.event) and d.folder == "fires"
    assert d.y_severity == "severe_damage"
    assert d.posted_at == pd.Timestamp("2017-11-20 20:04:24", tz="UTC")
    assert (d.split_cib_disaster_types, d.split_cib_damage) == ("test", "train")


def test_unaliased_crisismmd_like_folder_fails_loudly():
    with pytest.raises(ValueError, match="not aliased"):
        cib.build_manifest(_rows(folder="hurricane_irma_2017"))


def test_audit_flags_folder_derived_labels():
    rows = [
        {"source": "aidr_dt", "folder": f, "y_disaster_type": t, "sample_id": f"{f}{i}"}
        for f, t in [("nepal_earthquake", "earthquake"), ("chennai_flood", "flood")]
        for i in range(10)
    ]
    derived = cib.disaster_type_audit(pd.DataFrame(rows))
    assert derived["folder_derived"] and derived["aidr_weighted_folder_purity"] == 1.0

    rows[0]["y_disaster_type"] = rows[1]["y_disaster_type"] = rows[2]["y_disaster_type"] = "not_disaster"
    assert not cib.disaster_type_audit(pd.DataFrame(rows))["folder_derived"]  # purity 0.85
