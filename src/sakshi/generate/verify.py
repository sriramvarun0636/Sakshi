"""Five independent gates for every LLM-generated situation-report sentence."""

from __future__ import annotations

import re
from typing import Protocol

from sakshi.generate.lexicons import CASUALTY, NO_DAMAGE, SEVERE, terms
from sakshi.generate.numbers import unsupported


class EntailmentScorer(Protocol):
    model_name: str

    def score(self, premise: str, hypothesis: str) -> tuple[float, float]: ...

    def token_count(self, premise: str) -> int: ...


class DebertaNLI:
    """Lazy NLI model; label positions are read from the checkpoint config."""

    def __init__(
        self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", *, device: str | None = None
    ):
        self.model_name, self.device = model_name, device
        self._tokenizer = self._model = None
        self._indices = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        device = self.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self._model.to(device)
        names = {str(label).lower(): int(index) for index, label in self._model.config.id2label.items()}
        self._indices = {}
        for target in ("entailment", "contradiction"):
            matches = [index for label, index in names.items() if target in label]
            if len(matches) != 1:
                raise ValueError(f"NLI checkpoint lacks an unambiguous {target} label")
            self._indices[target] = matches[0]

    def token_count(self, premise: str) -> int:
        self._load()
        return len(self._tokenizer(premise, add_special_tokens=False)["input_ids"])

    def score(self, premise: str, hypothesis: str) -> tuple[float, float]:
        import torch

        self._load()
        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt", max_length=512, truncation="only_first")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with torch.no_grad():
            probabilities = self._model(**inputs).logits.softmax(dim=-1)[0]
        return float(probabilities[self._indices["entailment"]]), float(probabilities[self._indices["contradiction"]])


def _gate(passed: bool, detail: str, score: float | None = None) -> dict:
    result = {"passed": bool(passed), "detail": detail}
    if score is not None:
        result["score"] = round(score, 6)
    return result


def premise(pid: str, packet: dict) -> str:
    caption = packet.get("image_caption")
    description = caption["text"] if caption else "no description"
    return f"Post {pid}: {packet.get('text_clean') or '(no text)'} Image: {description}."


class Verifier:
    def __init__(
        self, scorer: EntailmentScorer, *, tau_entail: float = 0.5, tau_contra: float = 0.3, max_premise_tokens: int = 480
    ):
        if not 0 <= tau_entail <= 1 or not 0 <= tau_contra <= 1 or max_premise_tokens < 1:
            raise ValueError("invalid verifier thresholds")
        self.scorer, self.tau_entail, self.tau_contra = scorer, tau_entail, tau_contra
        self.max_premise_tokens = max_premise_tokens

    def verify(
        self,
        sentence: str,
        citations: list[str],
        packets: dict[str, dict],
        *,
        severity_by_pid: dict[str, str | None] | None = None,
        origin_status_by_pid: dict[str, str | None] | None = None,
    ) -> dict:
        """Run every gate, retaining all failures and non-failing provenance warnings."""
        citations = list(dict.fromkeys(citations))
        known = [pid for pid in citations if re.fullmatch(r"P[0-9]{1,3}", pid) and pid in packets]
        errors = []
        if not citations:
            errors.append("no_citation")
        if len(citations) > 5:
            errors.append("too_many_citations")
        errors.extend(f"unknown_id:{pid}" for pid in citations if pid not in known)
        gates = {"citation": _gate(not errors, ", ".join(errors) if errors else "ok")}

        origin_status_by_pid = origin_status_by_pid or {}

        def status(pid: str) -> str:
            return origin_status_by_pid.get(pid) or packets[pid]["provenance"]["status"]

        excluded = [pid for pid in known if status(pid) in {"recycled", "template_match"}]
        flagged = [pid for pid in known if status(pid) in {"cross_event_concurrent", "match_undated"}]
        detail = "excluded_provenance:" + ",".join(excluded) if excluded else "unverified_image_origin" if flagged else "ok"
        gates["provenance"] = _gate(not excluded, detail)

        if known:
            premises = [premise(pid, packets[pid]) for pid in known]
            combined = "\n".join(premises)
            scores = [(self.scorer.score(combined, sentence), "combined")]
            if self.scorer.token_count(combined) > self.max_premise_tokens:
                scores.extend((self.scorer.score(item, sentence), pid) for pid, item in zip(known, premises, strict=True))
            (p_entail, p_contra), source = max(scores, key=lambda result: result[0][0])
            good = p_entail >= self.tau_entail and p_contra < self.tau_contra
            gates["entailment"] = _gate(good, f"p_entail={p_entail:.3f}, p_contra={p_contra:.3f}, source={source}", p_entail)
            gates["entailment"]["p_contra"] = round(p_contra, 6)
        else:
            gates["entailment"] = _gate(False, "no valid cited premise")

        severe = terms(sentence, SEVERE)
        no_damage = terms(sentence, NO_DAMAGE)
        casualties = terms(sentence, CASUALTY)
        severity_by_pid = severity_by_pid or {}

        def severity(pid: str) -> str | None:
            return severity_by_pid.get(pid, packets[pid]["predictions"]["severity"]["label"])

        severe_support = any(severity(pid) == "severe_damage" for pid in known)
        minor_support = any(
            severity(pid) in {"little_or_no_damage", "mild_damage"} or terms(packets[pid].get("text_clean") or "", NO_DAMAGE)
            for pid in known
        )
        casualty_support = any(
            packets[pid]["predictions"]["humanitarian"]["label"] == "affected_individuals"
            or terms(packets[pid].get("text_clean") or "", CASUALTY)
            for pid in known
        )
        label_errors = []
        if any(not negated for _, negated in severe) and not severe_support:
            label_errors.append("severe")
        if (no_damage or any(negated for _, negated in severe)) and not minor_support:
            label_errors.append("no_damage")
        if any(not negated for _, negated in casualties) and not casualty_support:
            label_errors.append("casualty")
        gates["label"] = _gate(
            not label_errors,
            ", ".join(label_errors) if label_errors else "ok" if severe or no_damage or casualties else "not_applicable",
        )

        evidence = "\n".join(
            (packets[pid].get("text_clean") or "") + " " + ((packets[pid].get("image_caption") or {}).get("text") or "")
            for pid in known
        )
        # P-IDs can appear in citations or text; both are removed by the number extractor.
        number_errors = unsupported(sentence, evidence)
        gates["number"] = _gate(not number_errors, "ok" if not number_errors else "unsupported:" + ",".join(number_errors))
        return gates


def accepted(gates: dict) -> bool:
    return all(result["passed"] for result in gates.values())


def split_sentences(text: str) -> list[str]:
    """Split only on terminal punctuation followed by whitespace and a capital letter."""
    abbreviations = {"U.S.", "U.K.", "St.", "Dr.", "Mr.", "Mrs.", "No.", "Gov.", "Sept.", "Oct.", "Nov."}
    pieces = []
    start = 0
    for match in re.finditer(r"[.!?](?=\s+[A-Z])", text):
        prefix = text[start : match.end()]
        last = prefix.split()[-1] if prefix.split() else ""
        if last in abbreviations:
            continue
        pieces.append(prefix.strip())
        start = match.end()
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]
