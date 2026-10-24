"""Structured LLM backends with immutable request and response caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from sakshi import atomic, paths


class LLMClient(Protocol):
    def generate_json(
        self, system: str, user: str, schema: dict, *, max_tokens: int = 1024, temperature: float = 0.0, seed: int = 42
    ) -> dict: ...


class RawClient(Protocol):
    backend: str
    model: str

    def generate(self, system: str, user: str, schema: dict, *, max_tokens: int, temperature: float, seed: int) -> str: ...


@dataclass(frozen=True)
class GenerationFailure(Exception):
    reason: str
    keys: tuple[str, ...]

    def __str__(self) -> str:
        return self.reason


class OllamaClient:
    backend = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct", *, url: str = "http://127.0.0.1:11434/api/chat", num_ctx: int = 8192):
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("post content must be sent only to a local Ollama service")
        self.model, self.url, self.num_ctx = model, url, num_ctx

    def generate(self, system: str, user: str, schema: dict, *, max_tokens: int, temperature: float, seed: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        request = Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        return str(result["message"]["content"])


class HFClient:
    backend = "hf"

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model = model
        self._tokenizer = self._model = None

    def generate(self, system: str, user: str, schema: dict, *, max_tokens: int, temperature: float, seed: int) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self._model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("HF 7B generation requires a CUDA GPU; use Ollama on a laptop")
            memory = torch.cuda.get_device_properties(0).total_memory
            quantized = memory < 24 * 1024**3
            options = {"device_map": "auto"}
            if quantized:
                options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            else:
                options["torch_dtype"] = torch.bfloat16
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model = AutoModelForCausalLM.from_pretrained(self.model, **options).eval()

        torch.manual_seed(seed)
        messages = [
            {
                "role": "system",
                "content": system + "\nReturn only JSON matching this schema: " + json.dumps(schema, sort_keys=True),
            },
            {"role": "user", "content": user},
        ]
        input_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(self._model.device)
        options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "pad_token_id": self._tokenizer.eos_token_id}
        if temperature > 0:
            options["temperature"] = temperature
        with torch.no_grad():
            generated = self._model.generate(input_ids, **options)
        return self._tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def cache_key(
    backend: str, model: str, system: str, user: str, schema: dict, *, max_tokens: int, temperature: float, seed: int
) -> str:
    data = {
        "backend": backend,
        "model": model,
        "system": system,
        "user": user,
        "schema": schema,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "seed": int(seed),
    }
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class CachedLLM:
    """Validate every response, preserving invalid raw output and deterministic retries."""

    def __init__(self, raw: RawClient, cache_root: Path = paths.REPO / "runs" / "llm_cache", *, retries: int = 2):
        self.raw, self.cache_root, self.retries = raw, cache_root, retries
        self.keys: list[str] = []

    def generate_json(
        self, system: str, user: str, schema: dict, *, max_tokens: int = 1024, temperature: float = 0.0, seed: int = 42
    ) -> dict:
        validator = Draft202012Validator(schema)
        keys = []
        for attempt in range(self.retries + 1):
            prompt = (
                user
                if attempt == 0
                else user + "\nYour previous output was not valid JSON for the required schema. Return only the JSON object."
            )
            if attempt > 1:
                prompt += f" Retry {attempt}."
            key = cache_key(
                self.raw.backend,
                self.raw.model,
                system,
                prompt,
                schema,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            path = self.cache_root / f"{key}.json"
            keys.append(key)
            self.keys.append(key)
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("key") != key:
                    raise ValueError("LLM cache identity mismatch")
                raw = record["raw"]
            else:
                try:
                    raw = self.raw.generate(system, prompt, schema, max_tokens=max_tokens, temperature=temperature, seed=seed)
                except Exception as exc:
                    raise GenerationFailure(f"backend error: {exc}", tuple(keys)) from exc
                record = {"key": key, "backend": self.raw.backend, "model": self.raw.model, "raw": raw}
            try:
                parsed = json.loads(raw)
                validator.validate(parsed)
                record["parsed"] = parsed
                record["error"] = None
                if not path.exists():
                    atomic.text(path, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                return parsed
            except (ValueError, TypeError, KeyError) as exc:
                record["parsed"] = None
                record["error"] = str(exc)
            except Exception as exc:
                record["parsed"] = None
                record["error"] = str(exc)
            if not path.exists():
                atomic.text(path, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        raise GenerationFailure("invalid JSON after two retries", tuple(keys))
