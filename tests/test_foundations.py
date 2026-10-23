import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from sakshi import paths
from sakshi.images import load_rgb
from sakshi.text import clean_for_display, normalise_for_model
from sakshi.types import ProvStatus


def test_image_file_uses_dataset_root_and_rejects_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "CRISISMMD_ROOT", tmp_path / "cm")
    monkeypatch.setattr(paths, "CIB_ROOT", tmp_path / "cib")
    assert paths.image_file({"dataset": "crisismmd", "image_path": "data_image/a.png"}) == tmp_path / "cm/data_image/a.png"
    assert paths.image_file({"dataset": "cib", "image_path": "data/b.png"}) == tmp_path / "cib/data/b.png"
    with pytest.raises(ValueError, match="stay within"):
        paths.image_file({"dataset": "cib", "image_path": "../private.png"})
    with pytest.raises(ValueError, match="unknown"):
        paths.image_file({"dataset": "other", "image_path": "x.png"})


def test_text_normalisation_and_display_privacy(monkeypatch):
    monkeypatch.setitem(sys.modules, "emoji", SimpleNamespace(demojize=lambda value: value.replace("🔥", ":fire:")))
    raw = "RT @Alice: Fire &amp; smoke 🔥 https://example.org/x"
    assert normalise_for_model(raw) == "RT @USER: Fire & smoke :fire: HTTPURL"
    shown = clean_for_display(raw)
    assert shown == "RT [USER]: Fire & smoke 🔥 [LINK]"
    assert "@" not in shown and "http" not in shown.lower()
    assert normalise_for_model("RT @Alice https://example.org/x") is None
    assert normalise_for_model(" ") is None


def test_load_rgb_composites_transparency_and_handles_bad_files(tmp_path):
    rgba = Image.new("RGBA", (2, 1), (255, 0, 0, 128))
    path = tmp_path / "alpha.png"
    rgba.save(path)
    loaded = load_rgb(path)
    assert loaded is not None and loaded.mode == "RGB"
    red, green, blue = loaded.getpixel((0, 0))
    assert red == 255 and green in {127, 128} and blue in {127, 128}
    assert load_rgb(tmp_path / "absent.png") is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert load_rgb(bad) is None


def test_provenance_status_values_match_contract():
    assert len(ProvStatus) == 9
    assert ProvStatus.RECYCLED.value == "recycled"
