"""Optional on-demand Qwen image captions stored in the WP3 caption-cache format."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import pandas as pd
from PIL import Image

from sakshi import atomic, paths
from sakshi.images import load_rgb

PROMPT = (
    "Describe this image in one factual sentence of at most 20 words. "
    "Mention visible damage, flooding, fire, debris, people, vehicles or rescue activity if present. "
    "Do not guess the location, date, event or cause. "
    "If the image is a map, chart, screenshot, logo, poster or mostly text, say which."
)
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class Captioner(Protocol):
    model_name: str

    def caption(self, image: Image.Image, prompt: str) -> str: ...


class QwenCaptioner:
    model_name = MODEL

    def __init__(self) -> None:
        self._processor = self._model = None

    def caption(self, image: Image.Image, prompt: str) -> str:
        if self._model is None:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

            if not torch.cuda.is_available():
                raise RuntimeError("Qwen2.5-VL 7B captions require a CUDA GPU")
            memory = torch.cuda.get_device_properties(0).total_memory
            options = {"device_map": "auto"}
            if memory < 24 * 1024**3:
                options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            else:
                options["torch_dtype"] = torch.bfloat16
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **options).eval()
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        rendered = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[rendered], images=[image], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, do_sample=False, max_new_tokens=48)
        return self._processor.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0].strip()


def generate_captions(
    sample_ids: list[str],
    cache: Path,
    *,
    captioner: Captioner | None = None,
    manifest: Path = paths.PROCESSED / "manifest.parquet",
) -> Path:
    """Append only missing sample IDs; never regenerate an existing caption."""
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("caption request has duplicate sample IDs")
    columns = ["sample_id", "caption", "model", "prompt_sha256", "created_at"]
    existing = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=columns)
    if set(existing) != set(columns) or existing.sample_id.duplicated().any():
        raise ValueError("caption cache schema or uniqueness differs")
    pending = [sample_id for sample_id in sample_ids if sample_id not in set(existing.sample_id)]
    if not pending:
        return cache
    captioner = captioner or QwenCaptioner()
    rows = pd.read_parquet(manifest).set_index("sample_id")
    created = []
    prompt_sha = sha256(PROMPT.encode()).hexdigest()
    for sample_id in pending:
        if sample_id not in rows.index:
            raise ValueError("caption sample is absent from the manifest")
        image = load_rgb(paths.image_file(rows.loc[sample_id]))
        if image is None:
            raise ValueError("caption image could not be loaded")
        try:
            caption = captioner.caption(image, PROMPT)
        finally:
            image.close()
        if not caption:
            raise ValueError("captioner returned an empty caption")
        created.append(
            {
                "sample_id": sample_id,
                "caption": caption,
                "model": captioner.model_name,
                "prompt_sha256": prompt_sha,
                "created_at": datetime.now(timezone.utc),
            }
        )
        atomic.parquet(cache, pd.concat([existing, pd.DataFrame(created)], ignore_index=True))
    return cache
