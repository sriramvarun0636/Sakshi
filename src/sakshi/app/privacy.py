"""One privacy boundary for every image and free-text value shown by the dashboard."""

from __future__ import annotations

import html
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from sakshi.text import clean_for_display

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_PRIVATE = re.compile(r"@\w+|https?://\S+|www\.\S+", re.IGNORECASE)


def display_text(value: object) -> str:
    """Sanitize arbitrary data, including values that were previously cleaned."""
    if value is None:
        return ""
    cleaned = clean_for_display(str(value)) or ""
    return _PRIVATE.sub("[REDACTED]", cleaned)


def escaped(value: object) -> str:
    return html.escape(display_text(value), quote=True)


def non_english_warning(text: str) -> bool:
    """Conservative script check; emoji and accents on Latin letters do not trigger it."""
    return any(character.isalpha() and "LATIN" not in unicodedata.name(character, "") for character in text)


def decode_upload(content: bytes, *, filename: str = "") -> Image.Image:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit.")
    if not content:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise ValueError("Choose a JPEG, PNG, or WebP image.")
            if opened.width * opened.height > 40_000_000:
                raise ValueError("Image dimensions are too large to process safely.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable JPEG, PNG, or WebP image.") from exc
    if filename and Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        image.close()
        raise ValueError("Choose a JPEG, PNG, or WebP image.")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return image


def blur_faces(image: Image.Image, *, detector=None) -> Image.Image:
    """Blur detected faces before display using the bundled OpenCV Haar cascade."""
    import cv2

    rgb = np.asarray(image.convert("RGB")).copy()
    if detector is None:
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        if detector.empty():
            raise RuntimeError("Face detection is unavailable; presentation image hidden.")
    boxes = detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for x, y, width, height in boxes:
        left, top = max(0, x - width // 8), max(0, y - height // 8)
        right, bottom = min(rgb.shape[1], x + width + width // 8), min(rgb.shape[0], y + height + height // 8)
        region = rgb[top:bottom, left:right]
        rgb[top:bottom, left:right] = cv2.GaussianBlur(region, (0, 0), sigmaX=max(width, height) / 3)
    return Image.fromarray(rgb)


def presentation_image(image: Image.Image, *, presentation: bool = True, max_side: int = 256) -> Image.Image:
    result = image.convert("RGB")
    if presentation:
        result = blur_faces(result)
    result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return result


def load_thumbnail(path: Path | None, *, presentation: bool = True, max_side: int = 256) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as source:
            return presentation_image(ImageOps.exif_transpose(source), presentation=presentation, max_side=max_side)
    except (OSError, ValueError, RuntimeError):
        # Fail closed when decoding or face detection fails.
        return None
