"""Face-blurred heatmap overlays; no unblurred figure export path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sakshi import atomic
from sakshi.explain.gradcam import crop_heatmap


def blur_faces(image: Image.Image, *, detector=None) -> Image.Image:
    import cv2

    rgb = np.array(image.convert("RGB"))
    if detector is None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("face detector needs opencv-python-headless>=4.9,<5; refusing figure export")
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        if detector.empty():
            raise RuntimeError("face detector is unavailable; refusing figure export")
    boxes = detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5)
    for x, y, width, height in boxes:
        # Include a margin around every detected face before applying strong blur.
        left, top = max(0, x - width // 8), max(0, y - height // 8)
        right, bottom = min(rgb.shape[1], x + width + width // 8), min(rgb.shape[0], y + height + height // 8)
        roi = rgb[top:bottom, left:right]
        rgb[top:bottom, left:right] = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(width, height) / 3)
    return Image.fromarray(rgb)


def export_overlay(image: Image.Image, heatmap: np.ndarray, output: Path, *, detector=None) -> Path:
    import cv2

    if not np.isfinite(heatmap).all():
        raise ValueError("cannot display a nonfinite heatmap")
    blurred = blur_faces(image, detector=detector)
    try:
        rgb = np.array(blurred)
        width, height = blurred.size
        scale = 224 / min(width, height)
        crop_width, crop_height = 224 / scale, 224 / scale
        left, top = round((width - crop_width) / 2), round((height - crop_height) / 2)
        right, bottom = min(width, left + round(crop_width)), min(height, top + round(crop_height))
        crop = cv2.resize(crop_heatmap(heatmap), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
        colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(crop, 0, 1) * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        region = rgb[top:bottom, left:right]
        mixed = np.uint8(np.clip(0.6 * region + 0.4 * colour, 0, 255))
        region[crop > 0] = mixed[crop > 0]
        atomic.write(output, lambda temporary: Image.fromarray(rgb).save(temporary))
    finally:
        blurred.close()
    return output
