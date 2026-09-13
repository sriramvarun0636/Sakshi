"""One split assignment that is safe for a multi-task model trained on several datasets.

The official CrisisMMD and Crisis Image Benchmark splits were each drawn independently per task.
Training one model on the union would put one head's test images in another head's
training set. So we link every pair of samples that must not be separated -- images of
the same tweet, the same image appearing in two datasets, and perceptual near-duplicates
-- and give each linked group the most restrictive split any member was officially
assigned (test > dev > train). Every official test set therefore survives intact, which
keeps results comparable to published numbers; the cost is a smaller training set.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from sakshi.data.events import EVENTS

PRIORITY = {"train": 0, "dev": 1, "test": 2}
_BY_PRIORITY = {v: k for k, v in PRIORITY.items()}


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def components(sample_ids: pd.Series, groups: Iterable[pd.Series], edges: pd.DataFrame | None = None) -> pd.Series:
    """Connected-component id per sample.

    groups: Series aligned with sample_ids; samples sharing a non-null value are linked
            (e.g. tweet_id).
    edges:  DataFrame with columns a, b holding sample ids to link (e.g. near-duplicates).
    """
    ids = sample_ids.reset_index(drop=True)
    if ids.duplicated().any():
        raise ValueError("sample_ids must be unique")
    pos = pd.Series(np.arange(len(ids)), index=ids.values)
    uf = UnionFind(len(ids))
    for g in groups:
        g = g.reset_index(drop=True)
        for members in pd.Series(np.arange(len(g))).groupby(g.values, dropna=True):
            idx = members[1].to_numpy()
            for j in idx[1:]:
                uf.union(idx[0], j)
    if edges is not None and len(edges):
        unknown = set(edges.a).union(edges.b) - set(pos.index)
        if unknown:
            raise ValueError(f"edges reference {len(unknown)} unknown sample ids")
        for a, b in zip(pos[edges.a].to_numpy(), pos[edges.b].to_numpy(), strict=True):
            uf.union(a, b)
    return pd.Series(uf.roots(), index=sample_ids.index, name="component")


def resolve(official: pd.DataFrame, component: pd.Series) -> pd.Series:
    """Most restrictive official split per component, broadcast to every member.

    official: one column per official split source (values train/dev/test or NA).
    A sample with no official split anywhere, in a component with none either, is train.
    """
    bad = set(official.melt().value.dropna()) - set(PRIORITY)
    if bad:
        raise ValueError(f"unexpected split values {sorted(bad)}")
    per_sample = official.apply(lambda col: col.map(PRIORITY)).max(axis=1).fillna(0)
    per_component = per_sample.groupby(component).transform("max").astype(int)
    return per_component.map(_BY_PRIORITY).rename("split")


def loeo_folds(
    event: pd.Series,
    split: pd.Series,
    component: pd.Series,
    test_eligible: pd.Series,
    held_out_events: Iterable[str] = EVENTS,
) -> pd.DataFrame:
    """Leave-one-event-out roles: one column per held-out event, values train/dev/test/excluded.

    event:         canonical CrisisMMD event per sample (NA when unknown, e.g. most of the benchmark).
    split:         the resolved split; outside the held-out event, dev stays dev and
                   train/test both become train.
    component:     linked groups. Any group touching the held-out event is kept out of
                   training, so a repost of a held-out image cannot leak in through another
                   event or dataset.
    test_eligible: which held-out-event samples are tested (CrisisMMD rows). The others --
                   e.g. the benchmark's AIDR Harvey images, whose labels are not CrisisMMD labels --
                   are excluded instead.
    """
    base = split.map({"train": "train", "test": "train", "dev": "dev"})
    folds = {}
    for e in held_out_events:
        in_event = event == e
        role = base.mask(component.isin(component[in_event]), "excluded")
        folds[f"loeo_{e}"] = role.mask(in_event & test_eligible, "test")
    return pd.DataFrame(folds)


def summary(split: pd.Series, official: pd.DataFrame, component: pd.Series) -> dict:
    """What the unification cost, per official split source."""
    sizes = component.value_counts()
    out = {
        "resolved": split.value_counts().to_dict(),
        "components": int(component.nunique()),
        "largest_component": int(sizes.max()),
        "components_over_10": int((sizes > 10).sum()),
        "per_source": {},
    }
    for col in official:
        o = official[col]
        out["per_source"][col] = {
            "official_train": int((o == "train").sum()),
            "kept_as_train": int(((o == "train") & (split == "train")).sum()),
            "official_test_kept_as_test": int(((o == "test") & (split == "test")).sum()),
            "official_test": int((o == "test").sum()),
        }
    return out
