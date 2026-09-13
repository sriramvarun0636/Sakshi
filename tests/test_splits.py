import pandas as pd
import pytest

from sakshi.data import splits


def test_components_link_groups_and_edges():
    ids = pd.Series(["a", "b", "c", "d", "e"])
    tweet = pd.Series(["t1", "t1", "t2", None, None])
    edges = pd.DataFrame({"a": ["c"], "b": ["d"]})
    comp = splits.components(ids, [tweet], edges)
    assert comp[0] == comp[1]
    assert comp[2] == comp[3]
    assert comp[0] != comp[2]
    assert comp[4] not in {comp[0], comp[2]}  # NA group values do not link


def test_components_reject_unknown_edge_ids():
    with pytest.raises(ValueError, match="unknown"):
        splits.components(pd.Series(["a"]), [], pd.DataFrame({"a": ["a"], "b": ["zzz"]}))


def test_resolve_takes_most_restrictive_split_per_component():
    official = pd.DataFrame(
        {
            "task1": ["train", "train", "train", None, "dev"],
            "task2": ["train", "test", None, None, None],
        }
    )
    comp = pd.Series([0, 1, 1, 3, 4])
    out = splits.resolve(official, comp)
    # sample 0: train everywhere; 1 and 2 share a component where one task says test;
    # 3 has no official split at all; 4 is dev.
    assert out.tolist() == ["train", "test", "test", "train", "dev"]


def test_resolve_keeps_every_official_test_sample_in_test():
    official = pd.DataFrame({"t1": ["test", "train", "dev"], "t2": ["train", "test", "test"]})
    out = splits.resolve(official, pd.Series([0, 1, 2]))
    for col in official:
        assert (out[official[col] == "test"] == "test").all()


def test_resolve_rejects_unknown_split_names():
    with pytest.raises(ValueError, match="unexpected"):
        splits.resolve(pd.DataFrame({"t": ["train", "validation"]}), pd.Series([0, 1]))


def test_loeo_excludes_anything_linked_to_the_held_out_event():
    event = pd.Series(["harvey", "harvey", "irma", "irma", None, "harvey"])
    split = pd.Series(["train", "test", "train", "dev", "train", "train"])
    # sample 2 (irma) is a repost linked to harvey sample 0; sample 5 is a MEDIC Harvey image.
    comp = pd.Series([0, 1, 0, 3, 4, 5])
    eligible = pd.Series([True, True, True, True, False, False])
    folds = splits.loeo_folds(event, split, comp, eligible, held_out_events=["harvey"])
    assert folds["loeo_harvey"].tolist() == ["test", "test", "excluded", "dev", "train", "excluded"]
