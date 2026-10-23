import numpy as np
import pytest
from PIL import Image

from sakshi.provenance.transforms import NAMES, transform


@pytest.mark.parametrize("name", NAMES)
def test_transform_is_rgb_and_deterministic(name):
    pixels = np.random.default_rng(5).integers(0, 256, (100, 140, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    first = transform(image, name, seed=17)
    second = transform(image, name, seed=17)
    assert first.mode == "RGB"
    assert first.size == second.size
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert image.size == (140, 100)
    if name == "identity":
        np.testing.assert_array_equal(np.asarray(first), pixels)
    elif name == "crop90":
        assert first.size == (126, 90)
    elif name == "resize50":
        assert first.size == (70, 50)
    elif name == "caption_bar":
        assert first.size == (140, 115)
    elif name == "screenshot":
        assert first.width > 1080 and first.height > 1920


def test_unknown_transform_rejected():
    with pytest.raises(ValueError, match="unknown provenance transformation"):
        transform(Image.new("RGB", (10, 10)), "blur", seed=0)
