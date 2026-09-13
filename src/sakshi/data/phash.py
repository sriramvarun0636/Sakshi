"""Perceptual hashing, near-duplicate search and threshold calibration.

pHash (64-bit, DCT-based) survives re-encoding, resizing and mild edits. It is used for
cross-split deduplication and as the fast first tier of recycled-image (provenance)
detection. Near-uniform images (blank frames, flat colour) hash to nearly
identical values without being duplicates, so they are flagged as degenerate and never
linked.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# Some crawled images are truncated; the dataset authors' own loaders accept them too.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Grayscale standard deviation (0-255) below which an image is treated as near-uniform.
DEGENERATE_STD = 8.0


def hash_one(path: str) -> dict:
    """pHash and basic statistics of one image; unreadable files are recorded, not raised."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            gray = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
            bits = imagehash.phash(im).hash.ravel()
        value = int.from_bytes(np.packbits(bits).tobytes(), "big")
        return {"phash": value, "gray_std": float(gray.std()), "width": w, "height": h, "error": None}
    except Exception as e:  # noqa: BLE001 - any unreadable file is recorded, not fatal
        return {"phash": None, "gray_std": None, "width": None, "height": None, "error": f"{type(e).__name__}: {e}"}


def hash_images(paths: list[Path], workers: int = 8) -> pd.DataFrame:
    """One row per path: phash (uint64), gray_std, width, height, error, degenerate."""
    str_paths = [str(p) for p in paths]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(tqdm(pool.map(hash_one, str_paths, chunksize=64), total=len(str_paths), desc="phash"))
    df = pd.DataFrame(rows)
    df["phash"] = df.phash.astype("UInt64")
    df["degenerate"] = df.gray_std.lt(DEGENERATE_STD).fillna(False).astype(bool)
    return df


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise Hamming distance between two aligned uint64 arrays."""
    return np.bitwise_count(np.asarray(a, dtype=np.uint64) ^ np.asarray(b, dtype=np.uint64))


def near_duplicates(hashes: np.ndarray, max_distance: int, block: int = 256) -> pd.DataFrame:
    """All pairs (i, j), i < j, whose Hamming distance is <= max_distance.

    Exact brute force in blocks with vectorised popcount. For ~36k hashes this takes
    seconds, and exactness keeps the deduplication numbers easy to defend.
    """
    h = np.ascontiguousarray(hashes, dtype=np.uint64)
    out_i, out_j, out_d = [], [], []
    for s in range(0, len(h), block):
        e = min(s + block, len(h))
        d = np.bitwise_count(h[s:e, None] ^ h[None, s:])  # shape (block, n - s)
        ii, jj = np.nonzero(d <= max_distance)
        keep = jj > ii  # rows and columns are both offset by s, so j > i holds in local indices too
        out_i.append(ii[keep] + s)
        out_j.append(jj[keep] + s)
        out_d.append(d[ii[keep], jj[keep]])
    empty = np.array([], dtype=np.int64)
    return pd.DataFrame(
        {
            "i": np.concatenate(out_i) if out_i else empty,
            "j": np.concatenate(out_j) if out_j else empty,
            "distance": (np.concatenate(out_d) if out_d else empty).astype(np.int8),
        }
    )


def calibrate_threshold(
    hashes: np.ndarray,
    linked_distances: np.ndarray,
    max_distance: int,
    samples: int = 1_000_000,
    fit_range: tuple[int, ...] = (14, 16, 18, 20),
    seed: int = 0,
) -> dict:
    """Estimate how many linked pairs at each distance are chance collisions.

    Distances between random image pairs form a smooth tail below the mode (~32). At
    small distances that sample is polluted by genuine duplicates (a random pair
    occasionally *is* a duplicate pair), so a log-linear model is fitted on `fit_range`,
    where duplicates are negligible, and extrapolated downwards. Multiplying by the
    number of all pairs gives the expected number of chance collisions per distance,
    which bounds the precision of each candidate threshold.

    pHash sets exactly 32 of 64 bits (it thresholds the DCT at its median), so distances
    are always even. The estimate covers random collisions only; systematic look-alikes
    such as images sharing a template are reviewed separately.
    """
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(hashes), samples), rng.integers(0, len(hashes), samples)
    d = hamming(hashes[i[i != j]], hashes[j[i != j]])
    hist = pd.Series(d).value_counts().sort_index()
    fit_d = np.array([x for x in fit_range if hist.get(x, 0) > 0])
    slope, intercept = np.polyfit(fit_d, np.log(hist[fit_d].to_numpy() / len(d)), 1)
    total_pairs = len(hashes) * (len(hashes) - 1) / 2
    observed = pd.Series(linked_distances).value_counts()

    def precision(linked: float, chance: float) -> float | None:
        return round(1 - min(chance, linked) / linked, 3) if linked else None

    per_distance, cumulative = {}, {}
    run_linked, run_chance = 0, 0.0
    for t in range(0, max_distance + 1, 2):
        chance = float(np.exp(intercept + slope * t) * total_pairs)
        linked = int(observed.get(t, 0))
        run_linked, run_chance = run_linked + linked, run_chance + chance
        per_distance[t] = {"linked": linked, "expected_chance": round(chance, 1), "est_precision": precision(linked, chance)}
        cumulative[t] = {
            "linked": run_linked,
            "expected_chance": round(run_chance, 1),
            "est_precision": precision(run_linked, run_chance),
        }
    return {
        "random_pairs_sampled": int(len(d)),
        "random_pair_histogram": {int(k): int(v) for k, v in hist.items()},
        "tail_fit": {"distances": fit_d.tolist(), "ratio_per_2_bits": round(float(np.exp(-2 * slope)), 2)},
        "per_distance": per_distance,
        "cumulative": cumulative,
    }
