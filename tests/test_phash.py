import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from sakshi.data import phash


def _pattern(seed: int, size=(320, 240)) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BILINEAR)


def _save(im: Image.Image, path, **kw):
    im.save(path, **kw)
    return str(path)


def test_near_duplicates_matches_brute_force():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**63, size=600, dtype=np.uint64)
    h[10] = h[3] ^ np.uint64(0b1011)  # distance 3
    h[500] = h[7]  # distance 0
    got = phash.near_duplicates(h, max_distance=4, block=64)
    want = {(i, j) for i in range(len(h)) for j in range(i + 1, len(h)) if bin(int(h[i]) ^ int(h[j])).count("1") <= 4}
    assert set(zip(got.i, got.j, strict=True)) == want
    assert {(3, 10), (7, 500)} <= want


def test_reencoded_image_is_near_and_different_image_is_far(tmp_path):
    a = _pattern(1)
    orig = phash.hash_one(_save(a, tmp_path / "a.png"))
    buf = io.BytesIO()
    a.resize((200, 150)).filter(ImageFilter.GaussianBlur(1)).save(buf, "JPEG", quality=40)
    reenc = phash.hash_one(_save(Image.open(buf), tmp_path / "a2.jpg"))
    other = phash.hash_one(_save(_pattern(2), tmp_path / "b.png"))
    d_same = phash.hamming(np.array([orig["phash"]]), np.array([reenc["phash"]]))[0]
    d_other = phash.hamming(np.array([orig["phash"]]), np.array([other["phash"]]))[0]
    assert d_same <= 6
    assert d_other > 16


def test_uniform_image_is_degenerate_and_bad_file_is_recorded(tmp_path):
    flat = _save(Image.new("RGB", (100, 100), (200, 30, 30)), tmp_path / "flat.png")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    df = phash.hash_images([flat, bad], workers=1)
    assert df.degenerate.tolist() == [True, False]
    assert df.error[0] is None or df.error.isna()[0]
    assert "UnidentifiedImageError" in df.error[1]


def test_calibration_separates_planted_duplicates_from_chance():
    rng = np.random.default_rng(1)
    # Balanced 64-bit hashes (32 bits set), like real pHashes.
    bits = np.zeros((4000, 64), dtype=np.uint8)
    for row in bits:
        row[rng.choice(64, 32, replace=False)] = 1
    h = np.array([int.from_bytes(np.packbits(r).tobytes(), "big") for r in bits], dtype=np.uint64)
    h[1:200:2] = h[0:199:2]  # 100 exact duplicates
    pairs = phash.near_duplicates(h, 12)
    cal = phash.calibrate_threshold(h, pairs.distance.to_numpy(), 12, samples=200_000)
    assert cal["cumulative"][0]["linked"] >= 100
    assert cal["cumulative"][0]["est_precision"] == 1.0  # random 64-bit hashes essentially never collide
    precisions = [v["est_precision"] for v in cal["cumulative"].values() if v["est_precision"] is not None]
    assert precisions == sorted(precisions, reverse=True)  # looser thresholds are never more precise


@pytest.mark.parametrize("mode", ["L", "P", "RGBA"])
def test_non_rgb_modes_hash(tmp_path, mode):
    im = _pattern(3).convert(mode)
    assert phash.hash_one(_save(im, tmp_path / f"x_{mode}.png"))["error"] is None
