"""Data preparation pipeline.

Steps run in this order. Each reads the previous steps' outputs from `data/processed/`,
so any step can be re-run on its own:

    extract    unpack downloaded archives into data/raw/
    manifests  one row per image for each dataset: harmonised labels, official splits, timestamps
    hash       64-bit perceptual hash of every image (cached across runs)
    features   frozen CLIP tokens and retrieval embeddings (optional ML extra)
    dedupe     near-duplicate pairs, threshold calibration, cross-event provenance candidates
    audit      whether disaster-type labels were derived from event folders
    splits     unified leakage-free split and leave-one-event-out folds -> manifest.parquet
    report     regenerate reports/data/tables.md from the step outputs

Steps that need the Crisis Image Benchmark, HumAID, or the optional ML extra are skipped,
with a log message, until those inputs are present.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sakshi import atomic, paths
from sakshi.data import cib, crisismmd, humaid, phash, report, splits

log = logging.getLogger(__name__)

OUT = paths.PROCESSED
REPORT_DIR = paths.REPORTS / "data"

# Near-duplicate pairs are stored up to this distance; the linking threshold is applied later.
STORE_MAX_DISTANCE = 12
OFFICIAL_SPLITS = ["split_informative", "split_humanitarian", "split_damage", "split_cib_disaster_types", "split_cib_damage"]
LABELS = ["y_informative", "y_humanitarian", "y_severity", "y_disaster_type"]


@dataclass(frozen=True)
class Config:
    max_distance: int = 8  # pHash Hamming distance at or below which two images are linked
    workers: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.max_distance <= STORE_MAX_DISTANCE:
            raise ValueError(f"max_distance must be between 0 and {STORE_MAX_DISTANCE}; stored pairs stop there")
        if self.workers < 1:
            raise ValueError("workers must be positive")


def _save_json(name: str, obj: Any) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic.text(REPORT_DIR / name, json.dumps(obj, indent=2, default=str) + "\n")


def _load_json(name: str) -> Any:
    p = REPORT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- extract

_ARCHIVES: list[tuple[str, Path, Path]] = [  # (file in data/raw, destination, directory that must exist afterwards)
    ("crisismmd_datasplit_all.zip", paths.RAW / "splits_all", paths.CRISISMMD_SPLITS_ALL),
    ("crisismmd_datasplit_agreed_label.zip", paths.RAW / "splits_agreed", paths.CRISISMMD_SPLITS_AGREED),
    ("CrisisMMD_v2.0.tar.gz", paths.RAW, paths.CRISISMMD_ROOT / "data_image"),
    ("crisis_vision_benchmarks_task_readme.tar.gz", paths.RAW, paths.CIB_ROOT / "tasks"),
    ("crisis_vision_benchmarks_DMD.tar.gz", paths.RAW, paths.CIB_ROOT / "data"),
    ("crisis_vision_benchmarks_aidr_disaster_types.tar.gz", paths.RAW, paths.CIB_ROOT / "data"),
    ("HumAID_data_events_set1_47K.tar.gz", paths.HUMAID_ROOT, paths.HUMAID_ROOT / "events_set1"),
]


def extract(_cfg: Config) -> None:
    for archive, dest, expected in _ARCHIVES:
        src = paths.RAW / archive
        marker = paths.RAW / f".extracted_{archive}"
        if marker.exists():
            log.info("%s: already extracted", archive)
            continue
        if not src.exists() or any(paths.RAW.glob(f"{archive}.part*")):
            log.warning("%s: not downloaded yet; skipping", archive)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        log.info("%s: extracting to %s", archive, dest)
        if archive.endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                z.extractall(dest, members=[m for m in z.namelist() if "__MACOSX" not in m])
        else:  # tar fails loudly on a truncated archive, so the marker is only written on success
            subprocess.run(["tar", "-xzf", str(src), "-C", str(dest), "--exclude", "._*", "--exclude", "__MACOSX"], check=True)
        if not expected.exists():
            raise RuntimeError(f"{archive}: expected {expected} after extraction")
        marker.touch()


# ----------------------------------------------------------------------------- manifests


def manifests(_cfg: Config) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cm = crisismmd.build_manifest()
    stats: dict[str, Any] = {"crisismmd": crisismmd.check_consistency(cm)}
    if (paths.CRISISMMD_ROOT / "data_image").exists():
        missing = [p for p in cm.image_path.unique() if not (paths.CRISISMMD_ROOT / p).exists()]
        stats["crisismmd"]["missing_image_files"] = len(missing)
        if missing:
            log.warning("crisismmd: %d image files missing, e.g. %s", len(missing), missing[:3])
    atomic.parquet(OUT / "crisismmd.parquet", cm)
    log.info("crisismmd: %s images", f"{len(cm):,}")

    if (paths.CIB_ROOT / "tasks").exists():
        cb = cib.build_manifest()
        present = cb.image_path.map(lambda p: (paths.CIB_ROOT / p).exists())
        conflicting = cb.split_cib_disaster_types.notna() & cb.split_cib_damage.notna()
        conflicting &= cb.split_cib_disaster_types != cb.split_cib_damage
        stats["cib"] = {
            "images_labelled": len(cb),
            "images_with_files": int(present.sum()),
            "by_source": cb.source.value_counts().to_dict(),
            "files_present_by_source": cb[present].source.value_counts().to_dict(),
            "mapped_to_crisismmd_event": cb.event.value_counts().to_dict(),
            "labelled": {c: int(cb[c].notna().sum()) for c in ["y_disaster_type", "y_severity"]},
            "images_with_conflicting_official_splits": int(conflicting.sum()),
        }
        if present.any():
            atomic.parquet(OUT / "cib.parquet", cb[present])
            log.info("cib: %s of %s labelled images have files", f"{int(present.sum()):,}", f"{len(cb):,}")
        else:
            log.warning("cib: label files found but no images extracted; skipping")
    else:
        log.warning("cib: not extracted; skipping")

    if (paths.HUMAID_ROOT / "events_set1").exists():
        exclude, stats["humaid"] = humaid.overlap(humaid.read_tweets(), cm)
        atomic.csv(OUT / "humaid_exclude.csv", exclude)
        log.info("humaid: %d tweets repeat CrisisMMD tweets and are excluded from evaluation", len(exclude))
    _save_json("manifest_stats.json", stats)


# ----------------------------------------------------------------------------- hash


def hash_images(cfg: Config) -> None:
    for name, root in [("crisismmd", paths.CRISISMMD_ROOT), ("cib", paths.CIB_ROOT)]:
        manifest = OUT / f"{name}.parquet"
        if not manifest.exists() or not root.exists():
            log.warning("%s: no manifest or images; skipping", name)
            continue
        rel = pd.read_parquet(manifest, columns=["image_path"]).image_path.drop_duplicates()
        cache_path = OUT / f"phash_{name}.parquet"
        cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["image_path"])
        todo = rel[~rel.isin(cache.image_path)]
        log.info("%s: %s images, %s to hash", name, f"{len(rel):,}", f"{len(todo):,}")
        if len(todo):
            new = phash.hash_images([root / p for p in todo], workers=cfg.workers)
            new.insert(0, "image_path", todo.to_numpy())
            cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
            atomic.parquet(cache_path, cache)
        log.info("%s: %d unreadable, %d degenerate", name, int(cache.error.notna().sum()), int(cache.degenerate.sum()))


def cache_features(_cfg: Config) -> None:
    """Compute frozen CLIP features when the optional ML dependencies are installed."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        log.warning("features: optional ML dependencies missing; install with pip install -e '.[ml]' to build the cache")
        return
    from sakshi.data.features import ClipFeaturizer, build_cache

    build_cache(load_samples(), ClipFeaturizer())


# ----------------------------------------------------------------------------- dedupe


def load_samples() -> pd.DataFrame:
    """Every image from every dataset with its hash, one row each."""
    frames = []
    for name in ["crisismmd", "cib"]:
        manifest, hashes = OUT / f"{name}.parquet", OUT / f"phash_{name}.parquet"
        if manifest.exists() and hashes.exists():
            frames.append(pd.read_parquet(manifest).merge(pd.read_parquet(hashes), on="image_path", how="left"))
    if not frames:
        raise FileNotFoundError("run the manifests and hash steps first")
    return pd.concat(frames, ignore_index=True)


def exact_links(s: pd.DataFrame) -> pd.DataFrame:
    """The same tweet image in two datasets (AIDR-DT and CrisisMMD share the '<tweet_id>_<n>' id form)."""
    cm = s[s.dataset == "crisismmd"][["sample_id", "image_id"]]
    other = s[s.dataset != "crisismmd"][["sample_id", "image_id"]]
    j = other.merge(cm, on="image_id", suffixes=("_other", "_cm"))
    return pd.DataFrame({"a": j.sample_id_cm, "b": j.sample_id_other})


def dedupe(cfg: Config) -> None:
    s = load_samples()
    usable = s.phash.notna() & ~s.degenerate.fillna(True)
    idx = np.flatnonzero(usable.to_numpy())
    h = s.phash[usable].to_numpy(dtype=np.uint64)
    log.info("dedupe: %s usable hashes of %s images", f"{len(idx):,}", f"{len(s):,}")

    # For provenance every AIDR-DT folder counts as its own event (Nepal 2015, Kerala 2018 ...),
    # not just the three that coincide with CrisisMMD events: an older AIDR photo reposted in
    # 2017 is exactly the recycled-image case the provenance evaluation needs.
    s["prov_event"] = s.event.astype("string")
    if "folder" in s:
        s["prov_event"] = s.prov_event.fillna(("aidr:" + s.folder).where(s.source == "aidr_dt"))

    pairs = phash.near_duplicates(h, STORE_MAX_DISTANCE)
    a, b = idx[pairs.i.to_numpy()], idx[pairs.j.to_numpy()]
    cols = ["sample_id", "dataset", "event", "prov_event", "tweet_id", "posted_at", "image_path"]
    nd = pd.concat(
        [s.iloc[a][cols].add_suffix("_a").reset_index(drop=True), s.iloc[b][cols].add_suffix("_b").reset_index(drop=True)],
        axis=1,
    ).rename(columns={"sample_id_a": "a", "sample_id_b": "b"})
    nd["distance"] = pairs.distance.to_numpy()
    nd["same_tweet"] = nd.tweet_id_a.notna() & (nd.tweet_id_a == nd.tweet_id_b)
    atomic.parquet(OUT / "near_duplicates.parquet", nd)
    atomic.parquet(OUT / "exact_links.parquet", exact_links(s))
    _save_json("phash_calibration.json", phash.calibrate_threshold(h, nd.distance.to_numpy(), STORE_MAX_DISTANCE))

    linked = nd[nd.distance <= cfg.max_distance]
    both = linked.prov_event_a.notna() & linked.prov_event_b.notna()
    cross_event = both & (linked.prov_event_a != linked.prov_event_b)
    stats = {
        "threshold": cfg.max_distance,
        "pairs_at_threshold": int(len(linked)),
        "within_crisismmd": int(((linked.dataset_a == "crisismmd") & (linked.dataset_b == "crisismmd")).sum()),
        "within_cib": int(((linked.dataset_a == "cib") & (linked.dataset_b == "cib")).sum()),
        "cross_dataset": int((linked.dataset_a != linked.dataset_b).sum()),
        "same_tweet": int(linked.same_tweet.sum()),
        "cross_event": int(cross_event.sum()),
        "cross_event_involving_non_crisismmd_event": int((cross_event & (linked.event_a.isna() | linked.event_b.isna())).sum()),
        "distance_histogram": linked.distance.value_counts().sort_index().to_dict(),
    }
    _save_json("dedupe_stats.json", stats)

    # Cross-event matches: raw material for a hand-verified provenance test set.
    cand = linked[cross_event].copy()
    a_first = cand.posted_at_a < cand.posted_at_b
    cand["order_known"] = cand.posted_at_a.notna() & cand.posted_at_b.notna()
    cand["earlier"] = np.where(cand.order_known, np.where(a_first, cand.a, cand.b), "")
    cand["later"] = np.where(cand.order_known, np.where(a_first, cand.b, cand.a), "")
    cand["verdict"] = ""  # manual review: repost / template_or_placeholder / stock_or_press / false_match
    atomic.csv(REPORT_DIR / "provenance_candidates.csv", cand.sort_values("distance"))
    log.info("dedupe: %s linked pairs at distance <= %d", f"{len(linked):,}", cfg.max_distance)


# ----------------------------------------------------------------------------- audit


def audit(cfg: Config) -> None:
    if not (OUT / "cib.parquet").exists():
        log.warning("audit: Crisis Image Benchmark not available; skipping")
        return
    cm = pd.read_parquet(OUT / "crisismmd.parquet")
    cb = pd.read_parquet(OUT / "cib.parquet")
    nd = pd.read_parquet(OUT / "near_duplicates.parquet")
    nd = nd[(nd.distance <= cfg.max_distance) & (nd.dataset_a != nd.dataset_b)]
    links = pd.concat([pd.read_parquet(OUT / "exact_links.parquet"), nd[["a", "b"]]], ignore_index=True).drop_duplicates()
    result = cib.disaster_type_audit(cb, links, cm)
    _save_json("cib_disaster_type_audit.json", result)
    log.info(
        "audit: single-event folder purity %.3f -> folder-derived labels: %s",
        result["aidr_weighted_folder_purity"],
        result["folder_derived"],
    )


# ----------------------------------------------------------------------------- splits


def build_splits(cfg: Config) -> None:
    s = load_samples()
    for col in OFFICIAL_SPLITS:
        if col not in s:
            s[col] = pd.NA

    exact = OUT / "exact_links.parquet"
    links = pd.read_parquet(exact) if exact.exists() else pd.DataFrame(columns=["a", "b"])
    nd = pd.read_parquet(OUT / "near_duplicates.parquet")
    edges = pd.concat([links[["a", "b"]], nd.loc[nd.distance <= cfg.max_distance, ["a", "b"]]], ignore_index=True)

    decision = _load_json("cib_disaster_type_audit.json")
    folder_derived = bool(decision and decision.get("folder_derived"))
    if folder_derived:  # labels read off event folders would leak the event name into training
        s.loc[s.source == "aidr_dt", "y_disaster_type"] = pd.NA

    s["component"] = splits.components(s.sample_id, [s.tweet_id], edges)
    s["split"] = splits.resolve(s[OFFICIAL_SPLITS], s.component)
    folds = splits.loeo_folds(s.event, s.split, s.component, test_eligible=s.dataset == "crisismmd")
    s = pd.concat([s, folds], axis=1)
    s["has_label"] = s[LABELS].notna().any(axis=1)
    atomic.parquet(OUT / "manifest.parquet", s)

    summary = splits.summary(s.split, s[OFFICIAL_SPLITS], s.component)
    official = s.set_index("sample_id")[OFFICIAL_SPLITS]
    in_test, in_train = (official == "test").any(axis=1), (official == "train").any(axis=1)
    ea, eb = edges.a.to_numpy(), edges.b.to_numpy()
    straddle = (in_test[ea].to_numpy() & in_train[eb].to_numpy()) | (in_train[ea].to_numpy() & in_test[eb].to_numpy())
    summary |= {
        "threshold": cfg.max_distance,
        "linked_pairs": int(len(edges)),
        "linked_pairs_straddling_official_train_test": int(straddle.sum()),
        "exact_cross_dataset_copies": int(len(links)),
        "components_spanning_datasets": int((s.groupby("component").dataset.nunique() > 1).sum()),
        "aidr_disaster_type_dropped": folder_derived,
        "labels_by_split": {c: pd.crosstab(s[c], s.split).to_dict() for c in LABELS},
        "loeo": {c: s[c].value_counts().to_dict() for c in folds},
    }
    _save_json("split_summary.json", summary)
    log.info("splits: %s", summary["resolved"])


# ----------------------------------------------------------------------------- report


def write_report(_cfg: Config) -> None:
    path = report.write_tables(REPORT_DIR)
    log.info("report: %s", path)


STEPS: dict[str, Callable[[Config], None]] = {
    "extract": extract,
    "manifests": manifests,
    "hash": hash_images,
    "features": cache_features,
    "dedupe": dedupe,
    "audit": audit,
    "splits": build_splits,
    "report": write_report,
}


def run(steps: list[str] | None = None, cfg: Config | None = None) -> None:
    """Run the named steps (default: all) in pipeline order."""
    cfg = cfg or Config()
    unknown = set(steps or []) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {list(STEPS)}")
    for name, fn in STEPS.items():
        if not steps or name in steps:
            log.info("── %s", name)
            fn(cfg)
