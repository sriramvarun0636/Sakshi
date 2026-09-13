"""Resumable, parallel download of the source datasets.

Both hosts throttle single connections (CrisisNLP to roughly 500 KB/s, Harvard
Dataverse's S3 to less), so each file is split into byte ranges fetched concurrently.
An existing partial file is kept as a completed prefix, and each range resumes from
its own part file, so a download can be interrupted and restarted at any point.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sakshi import paths

log = logging.getLogger(__name__)

_CRISISNLP = "https://crisisnlp.qcri.org/data"
# Crisis Image Benchmark on Harvard Dataverse (doi:10.7910/DVN/QXT5QL); numbers are file ids.
_DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    description: str


SOURCES: dict[str, Source] = {
    "crisismmd_splits_all": Source(
        f"{_CRISISNLP}/crisismmd/crisismmd_datasplit_all.zip",
        "crisismmd_datasplit_all.zip",
        "CrisisMMD v2.0 official splits, all labels",
    ),
    "crisismmd_splits_agreed": Source(
        f"{_CRISISNLP}/crisismmd/crisismmd_datasplit_agreed_label.zip",
        "crisismmd_datasplit_agreed_label.zip",
        "CrisisMMD v2.0 official splits, text-image agreed labels",
    ),
    "crisismmd": Source(
        f"{_CRISISNLP}/crisismmd/CrisisMMD_v2.0.tar.gz",
        "CrisisMMD_v2.0.tar.gz",
        "CrisisMMD v2.0 tweets and images (1.9 GB)",
    ),
    "cib_labels": Source(
        f"{_DATAVERSE}/4494253",
        "crisis_vision_benchmarks_task_readme.tar.gz",
        "Crisis Image Benchmark label files",
    ),
    "cib_aidr_disaster_types": Source(
        f"{_DATAVERSE}/4494248",
        "crisis_vision_benchmarks_aidr_disaster_types.tar.gz",
        "Crisis Image Benchmark, AIDR-DT images (2.1 GB)",
    ),
    "cib_dmd": Source(
        f"{_DATAVERSE}/4494249",
        "crisis_vision_benchmarks_DMD.tar.gz",
        "Crisis Image Benchmark, DMD images (1.1 GB)",
    ),
    "humaid": Source(
        f"{_CRISISNLP}/humaid/HumAID_data_events_set1_47K.tar.gz",
        "HumAID_data_events_set1_47K.tar.gz",
        "HumAID text-only tweets, per-event splits",
    ),
}

_CHUNK = 1 << 20
# Dataverse answers 403 to urllib's default "Python-urllib" user agent.
_USER_AGENT = "Mozilla/5.0 (sakshi-dataset-downloader)"


def remote_size(url: str) -> int:
    """Total size from a one-byte ranged GET (Dataverse's signed S3 redirects refuse HEAD)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        content_range = r.headers.get("Content-Range")  # "bytes 0-0/<total>"
        if r.status != 206 or not content_range:
            raise RuntimeError(f"{url}: server does not support ranged downloads")
        return int(content_range.rsplit("/", 1)[1])


def _fetch_range(url: str, part: Path, start: int, end: int, retries: int = 20) -> None:
    """Fetch bytes [start, end) into `part`, resuming from whatever it already holds."""
    want = end - start
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            raise RuntimeError(f"{part} is larger than its range; delete it and re-run")
        headers = {"Range": f"bytes={start + have}-{end - 1}", "User-Agent": _USER_AGENT}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored Range request (HTTP {r.status})")
                with open(part, "ab") as f:
                    while chunk := r.read(_CHUNK):
                        f.write(chunk)
        except (OSError, RuntimeError) as e:
            wait = min(60, 2**attempt)
            log.warning("%s: %s; retrying in %ds", part.name, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"{part.name}: gave up after {retries} attempts")


def download(source: Source, out_dir: Path = paths.RAW, parts: int = 8) -> Path:
    """Download one source into `out_dir`, resuming any earlier partial download."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / source.filename
    total = remote_size(source.url)
    prefix = out.stat().st_size if out.exists() else 0
    if prefix == total:
        log.info("%s: already complete (%s bytes)", out.name, f"{total:,}")
        return out
    if prefix > total:
        raise RuntimeError(f"{out} is larger than the remote file; delete it and re-run")

    # The range layout is saved on the first run and reused, so restarting with a
    # different `parts` cannot misalign the existing part files.
    layout = out.with_name(f"{out.name}.ranges.json")
    if layout.exists():
        saved = json.loads(layout.read_text())
        if saved["prefix"] != prefix or saved["total"] != total:
            raise RuntimeError(f"{layout} does not match {out}; delete the part files and the layout to restart")
        ranges = [tuple(r) for r in saved["ranges"]]
    else:
        step = -(-(total - prefix) // parts)
        ranges = [(prefix + i * step, min(prefix + (i + 1) * step, total)) for i in range(parts)]
        ranges = [(s, e) for s, e in ranges if s < e]
        layout.write_text(json.dumps({"prefix": prefix, "total": total, "ranges": ranges}))
    part_files = [out.with_name(f"{out.name}.part{i:02d}") for i in range(len(ranges))]
    log.info("%s: fetching %s bytes in %d ranges", out.name, f"{total - prefix:,}", len(ranges))

    with ThreadPoolExecutor(len(ranges)) as pool:
        futures = [pool.submit(_fetch_range, source.url, p, s, e) for p, (s, e) in zip(part_files, ranges, strict=True)]
        while not all(f.done() for f in futures):
            got = prefix + sum(p.stat().st_size for p in part_files if p.exists())
            log.info("%s: %5.1f%% of %.2f GiB", out.name, 100 * got / total, total / 2**30)
            time.sleep(30)
        for f in futures:
            f.result()

    with open(out, "ab") as f:
        for p in part_files:
            with open(p, "rb") as src:
                while chunk := src.read(64 * _CHUNK):
                    f.write(chunk)
    if out.stat().st_size != total:
        raise RuntimeError(f"{out.name}: assembled size {out.stat().st_size:,} != {total:,}")
    for p in part_files:
        p.unlink()
    layout.unlink()
    log.info("%s: complete (%s bytes)", out.name, f"{total:,}")
    return out
