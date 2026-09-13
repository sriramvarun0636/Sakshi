"""Crisis Image Benchmark (CIB) subset: disaster-type and extra severity labels.

The CIB (Alam et al., ASONAM 2020) is MEDIC's predecessor. SAKSHI uses only the two
sources that carry disaster-type labels, which is ~3.2 GB instead of MEDIC's 11 GB:

  AIDR-DT  tweets collected by AIDR, stored in per-event folders (harvey/, nepal_earthquake/ ...)
  DMD      the Damage Multimodal Dataset (Mozannar et al., ISCRAM 2018), mostly Instagram

Disaster type is trained only on these per-image labels. DMD also supplies damage
severity. CIB's per-task splits were drawn independently, like CrisisMMD's, so each task's
official split is recorded here and `splits.py` resolves them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from sakshi import paths
from sakshi.data import labels
from sakshi.data.events import EVENTS, snowflake_time

AIDR_DT = "aidr_disaster_types"
DMD = "multimodal-deep-learning-disaster-response-mouzannar"
SOURCES = {AIDR_DT: "aidr_dt", DMD: "dmd"}

# Label files, relative to <CIB_ROOT>/tasks: (task, sub-directory, file prefix).
TASK_FILES = [
    ("disaster_types", "disaster_types/aidr_disaster_types", "aidr_disaster_types"),
    ("disaster_types", "disaster_types/damage_multimodal_data", "damage_multimodal_disaster_types"),
    ("damage", "damage_severity/damage_multimodal_data", "damage_multimodal_damage"),
]

# AIDR-DT event folders that are one of CrisisMMD's seven events. These matter for
# leave-one-event-out: an AIDR Harvey image must not train the Harvey-held-out fold.
# A None value records a folder that looks like a CrisisMMD event but is a different
# disaster. The build fails on any folder that looks like a CrisisMMD event and is not
# listed here, so the table cannot silently fall out of date.
AIDR_EVENT_ALIASES: dict[str, str | None] = {
    "harvey": "hurricane_harvey",
    "irma": "hurricane_irma",
    "maria": "hurricane_maria",
}
_CRISISMMD_EVENT_HINTS = ("harvey", "irma", "maria", "mexico", "iraq", "iran", "srilanka", "sri_lanka", "california")

_AIDR_DATE_RE = re.compile(r"/(\d{1,2})_(\d{1,2})_(\d{4})/")
_DMD_TIME_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.")


def read_task_files(root: Path = paths.CIB_ROOT) -> pd.DataFrame:
    """All label rows, long format: one row per (task, image)."""
    frames = []
    for task, subdir, prefix in TASK_FILES:
        for split in ("train", "dev", "test"):
            f = root / "tasks" / subdir / f"{prefix}_{split}_final.tsv"
            df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
            df["task"], df["split"] = task, split
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _aidr_folder(image_path: str) -> str | None:
    parts = Path(image_path).parts  # data/aidr_disaster_types/<event>/<d_m_yyyy>/<file>
    return parts[2] if len(parts) > 3 and parts[1] == AIDR_DT else None


def _aidr_date(image_path: str) -> pd.Timestamp | None:
    """AIDR-DT date folder, day_month_year like CrisisMMD's."""
    m = _AIDR_DATE_RE.search(image_path)
    if not m:
        return None
    day, month, year = map(int, m.groups())
    return pd.Timestamp(year=year, month=month, day=day)


def _posted_at(image_path: str, image_id: str, source: str) -> pd.Timestamp | None:
    if source == AIDR_DT:
        t = snowflake_time(image_id.split("_", 1)[0])
        return pd.Timestamp(t) if t else None
    m = _DMD_TIME_RE.search(image_path)  # DMD file names embed the post time
    if m:
        day, hh, mm, ss = m.groups()
        return pd.Timestamp(f"{day} {hh}:{mm}:{ss}", tz="UTC")
    return None


def build_manifest(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = read_task_files() if rows is None else rows
    rows = rows.assign(source=rows.image_path.map(lambda p: Path(p).parts[1]))
    unknown = set(rows.source) - set(SOURCES)
    if unknown:
        raise ValueError(f"unexpected CIB sources {sorted(unknown)}")

    def per_task(task: str, label_col: str, split_col: str) -> pd.DataFrame:
        sub = rows[rows.task == task]
        if sub.image_path.duplicated().any():
            raise ValueError(f"CIB {task}: an image appears more than once")
        return sub.set_index("image_path")[["class_label", "split"]].rename(
            columns={"class_label": label_col, "split": split_col}
        )

    base = rows.drop_duplicates("image_path").set_index("image_path")[["image_id", "source"]]
    m = pd.concat(
        [
            base,
            per_task("disaster_types", "raw_disaster_type", "split_cib_disaster_types"),
            per_task("damage", "raw_severity", "split_cib_damage"),
        ],
        axis=1,
    )
    m.index.name = "image_path"
    m = m.reset_index()

    m["y_disaster_type"] = labels.harmonise(m.raw_disaster_type, labels.CIB_DISASTER_TYPE, "CIB disaster_types")
    m["y_severity"] = labels.harmonise(m.raw_severity, labels.CIB_SEVERITY, "CIB damage_severity")
    m["folder"] = [(_aidr_folder(p) if s == AIDR_DT else Path(p).parts[3]) for p, s in zip(m.image_path, m.source, strict=True)]

    folders = set(m.folder[m.source == AIDR_DT].dropna())
    unaliased = sorted(f for f in folders if f not in AIDR_EVENT_ALIASES and any(k in f.lower() for k in _CRISISMMD_EVENT_HINTS))
    if unaliased:
        raise ValueError(f"AIDR folders that look like CrisisMMD events but are not aliased: {unaliased}")
    m["event"] = m.folder.where(m.source == AIDR_DT).map(AIDR_EVENT_ALIASES).astype("string")
    assert set(m.event.dropna()) <= set(EVENTS)

    heads = m.image_id.str.split("_", n=1).str[0]  # AIDR-DT image ids are '<tweet_id>_<n>'
    is_tweet = (m.source == AIDR_DT) & heads.map(lambda h: snowflake_time(h) is not None)
    m["tweet_id"] = heads.where(is_tweet)
    posted = [_posted_at(p, i, s) for p, i, s in zip(m.image_path, m.image_id, m.source, strict=True)]
    m["posted_at"] = pd.to_datetime(posted, utc=True)
    m["collection_date"] = pd.to_datetime(m.image_path.map(_aidr_date))
    m.insert(0, "dataset", "cib")
    m.insert(1, "sample_id", "cib:" + m.image_path)
    m["source"] = m.source.map(SOURCES)
    return m.drop(columns=["raw_disaster_type", "raw_severity"])


def disaster_type_audit(cib: pd.DataFrame, links: pd.DataFrame | None = None, crisismmd: pd.DataFrame | None = None) -> dict:
    """Were the disaster-type labels read off the folder / event rather than the image?

    AIDR-DT images sit in per-event folders. If the label were derived from the folder,
    every image in e.g. harvey/ would be 'hurricane', and each single-event folder would
    be ~100% one class. If a person labelled each image, memes, selfies and fundraising
    posters in harvey/ would be 'not_disaster', and purity would be well below 1.

    Decision rule, fixed before looking at the numbers: labels are treated as
    folder-derived if the image-weighted purity of the single-event AIDR folders is >= 0.9.

    If `links` (pairs a, b of sample ids) and `crisismmd` are given, CIB images that are
    copies of CrisisMMD images are cross-checked against CrisisMMD's own informativeness
    annotation, an independent labelling of the same pixels.
    """
    aidr = cib[(cib.source == "aidr_dt") & cib.y_disaster_type.notna()]
    # Folders named after a topic rather than one event are excluded from the purity test.
    mixed = {"natural_disaster", "human-induced_disaster", "landslide_crawled_images_v2", "security-incidents-activities"}
    single = aidr[~aidr.folder.isin(mixed)]
    ct = pd.crosstab(single.folder, single.y_disaster_type)
    purity = ct.max(axis=1) / ct.sum(axis=1)
    weighted = float((ct.max(axis=1).sum()) / ct.values.sum()) if len(ct) else float("nan")
    out = {
        "aidr_single_event_folders": int(len(ct)),
        "aidr_images_in_single_event_folders": int(ct.values.sum()),
        "aidr_weighted_folder_purity": round(weighted, 3),
        "folder_purity": {k: round(float(v), 3) for k, v in purity.sort_values().items()},
        "crisismmd_event_folders": {
            f: aidr[aidr.folder == f].y_disaster_type.value_counts().to_dict()
            for f in AIDR_EVENT_ALIASES
            if (aidr.folder == f).any()
        },
        "dmd_folder_vs_type": {
            f: g.y_disaster_type.value_counts().to_dict() for f, g in cib[cib.source == "dmd"].groupby("folder")
        },
        "folder_derived": bool(weighted >= 0.9),
    }
    if links is not None and crisismmd is not None and len(links):
        info = crisismmd.set_index("sample_id").informative_image
        dtype = cib.set_index("sample_id").y_disaster_type
        pairs = [(a, b) if a in info.index else (b, a) for a, b in zip(links.a, links.b, strict=True)]
        pairs = [(c, d) for c, d in pairs if c in info.index and d in dtype.index and pd.notna(dtype[d]) and pd.notna(info[c])]
        x = pd.DataFrame([(info[c], dtype[d]) for c, d in pairs], columns=["crisismmd_informative", "cib_disaster_type"])
        out["crisismmd_cross_check"] = {
            "linked_pairs": int(len(x)),
            "crosstab": pd.crosstab(x.crisismmd_informative, x.cib_disaster_type).to_dict() if len(x) else {},
        }
    return out
