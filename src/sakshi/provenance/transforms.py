"""Seeded image transformations for the closed-world injection benchmark (§8.9)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from sakshi.images import normalise_rgb

NAMES = ("identity", "jpeg30", "crop90", "resize50", "caption_bar", "filter", "screenshot", "flip", "combo")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _crop90(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * 0.9))
    crop_height = max(1, round(height * 0.9))
    left = int(rng.integers(0, width - crop_width + 1))
    top = int(rng.integers(0, height - crop_height + 1))
    return image.crop((left, top, left + crop_width, top + crop_height))


def _caption_bar(image: Image.Image) -> Image.Image:
    width, height = image.size
    bar_height = max(1, round(height * 0.15))
    result = Image.new("RGB", (width, height + bar_height), "white")
    result.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=max(1, round(bar_height * 0.6)))
    draw.text((max(1, width // 50), 0), "BREAKING NEWS", fill="black", font=font)
    return result


def _filter(image: Image.Image) -> Image.Image:
    enhanced = ImageEnhance.Color(image).enhance(1.3)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
    pixels = np.asarray(enhanced, dtype=np.float32).copy()
    pixels[..., 0] *= 1.08
    pixels[..., 2] *= 0.92
    height, width = pixels.shape[:2]
    y, x = np.ogrid[:height, :width]
    # Normalise to one at each corner, including non-square images.
    x = (x - (width - 1) / 2) / max((width - 1) / 2, 1)
    y = (y - (height - 1) / 2) / max((height - 1) / 2, 1)
    radius_sq = (x * x + y * y) / 2
    pixels *= (1 - 0.4 * radius_sq)[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _screenshot(image: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (1080, 1920), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1079, 59), fill=(40, 40, 40))
    draw.rectangle((0, 1800, 1079, 1919), fill=(230, 230, 230))
    width = 1000
    height = max(1, round(image.height * width / image.width))
    scaled = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas.paste(scaled, (40, 100))
    return _jpeg(canvas.rotate(1.5, resample=Image.Resampling.BICUBIC, expand=True), 70)


def transform(image: Image.Image, name: str, *, seed: int) -> Image.Image:
    """Return an RGB copy after the exact named benchmark transformation."""
    if name not in NAMES:
        raise ValueError(f"unknown provenance transformation {name!r}")
    rgb = normalise_rgb(image)
    rng = np.random.default_rng(seed)
    if name == "identity":
        return rgb
    if name == "jpeg30":
        return _jpeg(rgb, 30)
    if name == "crop90":
        return _crop90(rgb, rng)
    if name == "resize50":
        return rgb.resize((max(1, round(rgb.width * 0.5)), max(1, round(rgb.height * 0.5))), Image.Resampling.BICUBIC)
    if name == "caption_bar":
        return _caption_bar(rgb)
    if name == "filter":
        return _filter(rgb)
    if name == "screenshot":
        return _screenshot(rgb)
    if name == "flip":
        return ImageOps.mirror(rgb)
    return _jpeg(_caption_bar(_crop90(rgb, rng)), 50)
