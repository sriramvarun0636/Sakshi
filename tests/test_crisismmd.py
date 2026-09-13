import pandas as pd
import pytest

from sakshi.data import crisismmd

_TWEET_A = str(905960092822003712)  # Harvey, 2017-09-08
_TWEET_B = str(917791291823591425)  # California wildfires, 2017-10-10


def _row(task, variant, split, tweet, event, idx, label, lt=None, li=None):
    return {
        "event_name": event,
        "tweet_id": tweet,
        "image_id": f"{tweet}_{idx}",
        "tweet_text": "Weâ€™re safe",
        "image": f"data_image/{event}/8_9_2017/{tweet}_{idx}.jpg",
        "label": label,
        "label_text": lt,
        "label_image": li,
        "task": task,
        "variant": variant,
        "split": split,
    }


def _rows(agreed_split="train"):
    a, b = (_TWEET_A, "hurricane_harvey"), (_TWEET_B, "california_wildfires")
    return pd.DataFrame(
        [
            _row("informative", "all", "train", *a, 0, "informative", "informative", "informative"),
            _row("informative", "all", "test", *b, 0, "informative", "informative", "not_informative"),
            _row("humanitarian", "all", "test", *a, 0, "vehicle_damage", "vehicle_damage", "vehicle_damage"),
            _row(
                "humanitarian",
                "all",
                "train",
                *b,
                0,
                "other_relevant_information",
                "other_relevant_information",
                "not_humanitarian",
            ),
            _row("informative", "agreed", agreed_split, *a, 0, "informative", "informative", "informative"),
            _row(
                "humanitarian", "agreed", "test", *a, 0, "infrastructure_and_utility_damage", "vehicle_damage", "vehicle_damage"
            ),
            _row("damage", "all", "dev", *a, 0, "severe_damage"),
        ]
    )


def test_manifest_labels_splits_and_time():
    m = crisismmd.build_manifest(_rows()).set_index("image_id")
    a, b = f"{_TWEET_A}_0", f"{_TWEET_B}_0"
    assert m.loc[a, "y_informative"] == "informative"
    assert pd.isna(m.loc[b, "y_informative"])  # text and image disagree -> no agreed label
    assert m.loc[a, "humanitarian_image"] == "infrastructure_and_utility_damage"  # 8 -> 5 merge
    assert m.loc[a, "y_severity"] == "severe_damage"
    assert (m.loc[a, "split_informative"], m.loc[a, "split_humanitarian"], m.loc[a, "split_damage"]) == ("train", "test", "dev")
    assert m.loc[a, "text"] == "We're safe" and m.loc[a, "text_repaired"]
    assert m.loc[a, "posted_at"].date().isoformat() == "2017-09-08"
    assert m.loc[a, "collection_date"].date().isoformat() == "2017-09-08"


def test_leak_counts():
    leaks = crisismmd.split_leaks(crisismmd.build_manifest(_rows()))
    assert leaks["images_test_for_one_task_train_for_another"] == 2


def test_agreed_rows_must_keep_their_official_split():
    with pytest.raises(ValueError, match="different split"):
        crisismmd.build_manifest(_rows(agreed_split="dev"))
