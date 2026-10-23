"""Robust image loading shared by the feature cache and live inference."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalise_rgb(opened: Image.Image) -> Image.Image:
    """Copy the first frame, honour EXIF orientation and composite transparency on white."""
    opened.seek(0)
    image = ImageOps.exif_transpose(opened)
    if image.mode in {"P", "LA", "RGBA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def load_rgb(path: str | Path | None) -> Image.Image | None:
    """Open the first frame, apply EXIF orientation, and composite alpha onto white."""
    if path is None:
        return None
    try:
        with Image.open(path) as opened:
            return normalise_rgb(opened)
    except Exception:  # noqa: BLE001 - every unsupported or corrupt image is a missing image
        return None
