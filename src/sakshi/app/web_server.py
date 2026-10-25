"""Loopback-only, framework-free HTTP interface for the SAKSHI presentation app."""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import threading
from contextlib import suppress
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from sakshi import paths
from sakshi.app import data
from sakshi.app.privacy import decode_upload, display_text, load_thumbnail, non_english_warning, presentation_image
from sakshi.data.events import EVENTS

ROOT = Path(os.environ.get("SAKSHI_APP_ROOT", paths.REPO)).expanduser().resolve()
WEB = Path(__file__).with_name("web")
_KEY = secrets.token_bytes(32)
_INFERENCE_LOCK = threading.RLock()
_LIVE: dict[str, object] = {}
_LIVE_ORDER: list[str] = []


def _known(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    answer = str(value).strip()
    return answer if answer and answer.lower() not in {"nan", "none", "null"} else None


def _human(value: object) -> str:
    return (_known(value) or "not available").replace("_", " ").capitalize()


@lru_cache(maxsize=1)
def _manifest() -> dict[str, dict]:
    frame = data.manifest(ROOT)
    return {str(row["sample_id"]): row for row in frame.to_dict("records")} if not frame.empty else {}


@lru_cache(maxsize=1)
def _aggregates() -> dict[str, data.AggregateSource]:
    return data.preferred_aggregates(ROOT)


@lru_cache(maxsize=1)
def _sitreps() -> dict[str, data.SitrepSource]:
    return data.sitreps(ROOT)


@lru_cache(maxsize=12)
def _tweets(directory: str) -> pd.DataFrame:
    return pd.read_parquet(Path(directory) / "tweets.parquet")


@lru_cache(maxsize=12)
def _windows(directory: str) -> tuple[dict, ...]:
    source = next((item for item in _aggregates().values() if str(item.directory) == directory), None)
    return tuple(data.read_json(path) for path in source.windows) if source else ()


@lru_cache(maxsize=12)
def _replay(event: str) -> pd.DataFrame:
    return data.replay(ROOT, event)


@lru_cache(maxsize=1)
def _predictions() -> tuple[pd.DataFrame, str]:
    return data.classifier_predictions(ROOT)


def _image_url(image, *, max_side: int = 512) -> str | None:
    if image is None:
        return None
    output = io.BytesIO()
    image.thumbnail((max_side, max_side))
    image.save(output, format="WEBP", quality=80)
    return "data:image/webp;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@lru_cache(maxsize=1024)
def _thumbnail(sample_id: str, presentation: bool, max_side: int = 360) -> str | None:
    row = _manifest().get(sample_id)
    image = load_thumbnail(data.image_path(ROOT, row), presentation=presentation, max_side=max_side)
    return _image_url(image, max_side=max_side)


def _summary() -> dict:
    sources = _aggregates()
    reports = data.report_files(ROOT)
    checkpoint = data.classifier_run(ROOT)
    return {
        "classifier": bool(checkpoint),
        "events": [
            {"id": event, "name": _human(event), "windows": len(source.windows), "evidence_ready": source.evidence_ready}
            for event, source in sources.items()
        ],
        "sitrep_events": [_human(event) for event in _sitreps()],
        "report_count": len(reports),
        "provenance_events": len(list((ROOT / "data" / "processed" / "provenance").glob("replay_*.parquet"))),
        "model_name": checkpoint.name if checkpoint else None,
    }


def _replay_view(event: str, index: int, lens: str, presentation: bool) -> dict:
    source = _aggregates().get(event)
    if source is None:
        return {"available": False, "message": "Event aggregates are missing.", "command": "sakshi-aggregate --mode demo ..."}
    index = min(max(index, 0), len(source.windows) - 1)
    all_windows = _windows(str(source.directory))
    current = all_windows[index]
    counts = current["counts"]
    begin = pd.Timestamp(current["window"]["start"])
    end = pd.Timestamp(current["window"]["end"])
    frame = _tweets(str(source.directory))
    posted = pd.to_datetime(frame["posted_at"], utc=True)
    feed = frame.loc[(posted >= begin) & (posted < end)].sort_values("posted_at", kind="stable")
    if lens == "flagged":
        feed = feed.loc[feed.provenance_status.isin(["recycled", "cross_event_concurrent", "match_undated", "template_match"])]
    elif lens == "severe":
        feed = feed.loc[feed.severity_label == "severe_damage"]
    origins = {str(row["post_id"]): row for row in _replay(event).to_dict("records")}
    cards = []
    for row in feed.head(8).to_dict("records"):
        ids = row.get("sample_ids")
        ids = list(ids) if ids is not None else []
        origin = next((origins.get(str(pid)) for pid in ids if origins.get(str(pid), {}).get("status") == "recycled"), None)
        original_id = str(origin.get("origin_post_id")) if origin else ""
        representative_id = str(row.get("representative_id"))
        representative = _manifest().get(representative_id, {})
        status = _known(row.get("provenance_status")) or (
            "origin unassessed" if _known(representative.get("image_path")) else "no image"
        )
        cards.append(
            {
                "text": display_text(_known(row.get("text_clean")) or "Image-only post"),
                "time": str(row.get("posted_at"))[:16].replace("T", " "),
                "category": _human(row.get("humanitarian_label")),
                "severity": _human(row.get("severity_label")) if _known(row.get("severity_label")) else None,
                "status": status.replace("_", " "),
                "recycled": status == "recycled",
                "image": _thumbnail(representative_id, presentation),
                "origin_image": _thumbnail(original_id, presentation) if origin else None,
                "origin_event": _human(origin.get("origin_event")) if origin else None,
                "origin_date": str(origin.get("origin_first_seen"))[:16] if origin else None,
            }
        )
    cumulative = [item["counts"] for item in all_windows[: index + 1]]
    return {
        "available": True,
        "event": _human(event),
        "model": source.model_id,
        "evidence_ready": source.evidence_ready,
        "window": index,
        "window_count": len(all_windows),
        "start": begin.strftime("%d %b %Y · %H:%M UTC"),
        "end": end.strftime("%d %b %Y · %H:%M UTC"),
        "timeline": [
            {
                "date": item["window"]["start"][:10],
                "count": item["counts"]["tweets"],
                "recycled": item["counts"]["excluded_provenance"]["recycled"],
            }
            for item in all_windows
        ],
        "counts": {
            "posts": sum(item["tweets"] for item in cumulative),
            "informative": sum(item["informative"] for item in cumulative),
            "recycled": sum(item["excluded_provenance"]["recycled"] for item in cumulative),
            "review": sum(item["abstained"] for item in cumulative),
        },
        "window_posts": counts["tweets"],
        "feed_count": len(feed),
        "feed": cards,
    }


def _sitrep_view(event: str, index: int, presentation: bool) -> dict:
    source = _sitreps().get(event)
    if source is None:
        preview = _aggregates().get(event)
        lines = []
        if preview:
            chosen = preview.windows[min(max(index, 0), len(preview.windows) - 1)]
            lines = [display_text(line) for line in data.read_json(chosen).get("system_lines", [])]
        return {
            "available": False,
            "system_lines": lines,
            "message": "Verified situation report pending",
            "command": "sakshi-generate sitreps --config configs/generation/sitrep.yaml ...",
        }
    index = min(max(index, 0), len(source.windows) - 1)
    report = data.read_json(source.windows[index])
    citations = {}
    for pid, ids in report.get("input", {}).get("sample_ids", {}).items():
        row = next((_manifest().get(str(sample_id)) for sample_id in ids if _manifest().get(str(sample_id))), None)
        citations[pid] = {
            "text": display_text(row.get("text")) if row else "Cited post unavailable on this laptop",
            "image": _thumbnail(str(row["sample_id"]), presentation) if row else None,
        }
    sentences = []
    for sentence in report.get("sentences", []):
        sentences.append(
            {
                "id": sentence.get("id"),
                "text": display_text(sentence.get("text")),
                "accepted": bool(sentence.get("accepted")),
                "citations": sentence.get("citations", []),
                "failed_gates": [
                    {"name": name.replace("_", " "), "detail": display_text(value.get("detail"))}
                    for name, value in sentence.get("gates", {}).items()
                    if not value.get("passed")
                ],
            }
        )
    return {
        "available": True,
        "event": _human(event),
        "window": index,
        "window_count": len(source.windows),
        "date": report["window"]["start"][:16].replace("T", " ") + " UTC",
        "system_lines": [display_text(line) for line in report.get("system_lines", [])],
        "sentences": sentences,
        "citations": citations,
        "rates": report.get("verification_rate", {}),
        "models": report.get("models", {}),
        "mode": source.mode,
    }


def _review_items() -> tuple[list[dict], str]:
    predictions, source = _predictions()
    items: dict[str, dict] = {}
    if not predictions.empty and "abstain" in predictions:
        for row in predictions.loc[predictions.abstain].to_dict("records"):
            sample_id = str(row["sample_id"])
            items[sample_id] = {
                "id": sample_id,
                "kind": "abstained",
                "reason": _known(row.get("abstain_reason")) or "Low confidence",
            }
    for event in EVENTS:
        replay = _replay(event)
        if replay.empty:
            continue
        for row in replay.loc[replay.status.isin(["match_undated", "cross_event_concurrent"])].to_dict("records"):
            sample_id = str(row["post_id"])
            items.setdefault(
                sample_id, {"id": sample_id, "kind": str(row["status"]), "reason": "Image origin needs human review"}
            )
    return sorted(items.values(), key=lambda item: (item["kind"] != "abstained", item["kind"], item["id"])), source


def _decision_path() -> Path:
    return ROOT / "runs" / "review" / "decisions.jsonl"


def _decisions() -> dict[str, dict]:
    path = _decision_path()
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            result[str(entry["sample_id"])] = entry
        except (ValueError, KeyError, TypeError):
            continue
    return result


def _token(sample_id: str) -> str:
    return sha256(_KEY + sample_id.encode()).hexdigest()[:24]


def _review_view(lens: str, presentation: bool) -> dict:
    items, source = _review_items()
    decisions = _decisions()
    pending = [item for item in items if item["id"] not in decisions]
    if lens == "abstained":
        shown = [item for item in items if item["kind"] == "abstained"]
    elif lens == "origin":
        shown = [item for item in items if item["kind"] != "abstained"]
    elif lens == "all":
        shown = items
    else:
        shown = pending
    cards = []
    for item in shown[:12]:
        row = _manifest().get(item["id"], {})
        cards.append(
            {
                "token": _token(item["id"]),
                "kind": item["kind"].replace("_", " "),
                "reason": display_text(item["reason"]),
                "text": display_text(row.get("text")) or "Image-only post",
                "event": _human(row.get("event")),
                "image": _thumbnail(item["id"], presentation),
                "decision": decisions.get(item["id"], {}).get("decision"),
            }
        )
    return {
        "total": len(items),
        "pending": len(pending),
        "reviewed": len(items) - len(pending),
        "source": source,
        "shown": len(shown),
        "items": cards,
    }


def _save_review(token: str, decision: str) -> dict:
    if decision not in {"confirm", "dismiss"}:
        raise ValueError("Invalid review decision")
    item = next((item for item in _review_items()[0] if _token(item["id"]) == token), None)
    if item is None:
        raise ValueError("Review record unavailable")
    path = _decision_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sample_id": item["id"],
        "decision": decision,
        "reason": item["reason"],
        "reviewed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True}


def _results_view(name: str | None, presentation: bool) -> dict:
    reports = data.report_files(ROOT)
    preferred = ("classifier_three_seed", "comparison_seed0_trained", "classifier")
    default = next((path for stem in preferred for path in reports if path.stem == stem), reports[0] if reports else None)
    selected = next((path for path in reports if path.stem == name), default)
    figure_names = {
        "classifier_three_seed": (
            ("classifier_seed2_reliability.png", "SEED 2 / OFFICIAL TEST CALIBRATION"),
            ("classifier_seed2_risk_coverage.png", "SEED 2 / REVIEW RISK AND COVERAGE"),
        ),
        "classifier_seed2": (
            ("classifier_seed2_reliability.png", "OFFICIAL TEST CALIBRATION"),
            ("classifier_seed2_risk_coverage.png", "REVIEW RISK AND COVERAGE"),
        ),
        "classifier_seed1": (
            ("classifier_seed1_reliability.png", "OFFICIAL TEST CALIBRATION"),
            ("classifier_seed1_risk_coverage.png", "REVIEW RISK AND COVERAGE"),
        ),
        "classifier": (
            ("classifier_seed0_reliability.png", "OFFICIAL TEST CALIBRATION"),
            ("classifier_seed0_risk_coverage.png", "REVIEW RISK AND COVERAGE"),
        ),
        "faithfulness_dev": (("faithfulness_curves_dev.png", "DEVELOPMENT CHECK / FAITHFULNESS"),),
    }
    figures = []
    for filename, label in figure_names.get(selected.stem if selected else "", ()):
        path = ROOT / "reports" / "figures" / filename
        if path.is_file():
            thumbnail = load_thumbnail(path, presentation=presentation, max_side=1600)
            figures.append({"label": label, "src": _image_url(thumbnail, max_side=1600), "alt": path.stem.replace("_", " ")})

    def kind(path: Path) -> str:
        if "dev" in path.stem:
            return "DEVELOPMENT CHECK"
        if "demo" in path.stem:
            return "DEMO RESULT"
        if "baseline" in path.stem:
            return "BASELINE RESULT"
        return "REPORTED RESULT"

    return {
        "reports": [{"id": path.stem, "title": path.stem.replace("_", " ").title(), "kind": kind(path)} for path in reports],
        "selected": selected.stem if selected else None,
        "content": "\n".join(display_text(line) for line in selected.read_text(encoding="utf-8").splitlines())
        if selected
        else "",
        "figures": figures,
    }


def _model():
    from sakshi.models.predict import Predictor

    run = data.classifier_run(ROOT)
    if run is None:
        raise ValueError("Classifier checkpoint missing")
    cache = Path("/private/tmp/sakshi_hf_cache")
    if "HF_HOME" not in os.environ and cache.is_dir():
        os.environ["HF_HOME"] = str(cache)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return Predictor.from_run(run, device="cpu")


_predictor = lru_cache(maxsize=1)(_model)


def _classify(payload: dict) -> dict:
    from sakshi.app.live import classify

    text = str(payload.get("text") or "")[:5000]
    encoded = payload.get("image")
    presentation = payload.get("presentation", True) is not False
    image = None
    if encoded:
        content = base64.b64decode(str(encoded), validate=True)
        image = decode_upload(content, filename=str(payload.get("filename") or "upload.jpg"))
    with _INFERENCE_LOCK:
        predictor = _predictor()
        result = classify(predictor, text, image)
        origin = None
        if image is not None:
            try:
                from sakshi.provenance.index import ClipGraphicScorer, ProvenanceIndex

                index = _index(ClipGraphicScorer, ProvenanceIndex)
                index.live_featurizer = predictor._live_clip
                event = payload.get("event") if payload.get("event") in EVENTS else None
                time = datetime.fromisoformat(payload["time"].replace("Z", "+00:00")) if event and payload.get("time") else None
                if time is not None and time.tzinfo is None:
                    time = time.replace(tzinfo=timezone.utc)
                checked = index.check(image=image, event=event if time else None, time=time)
                origin = {
                    "status": checked.status.value,
                    "event": _human(checked.origin_event) if checked.origin_event else None,
                    "date": checked.origin_first_seen.isoformat() if checked.origin_first_seen else None,
                    "similarity": checked.similarity,
                    "image": _thumbnail(checked.origin_post_id, presentation) if checked.origin_post_id else None,
                }
            except Exception:
                origin = {"status": "unavailable", "event": None, "date": None, "similarity": None, "image": None}
        token = secrets.token_urlsafe(20)
        _LIVE[token] = result
        _LIVE_ORDER.append(token)
        while len(_LIVE_ORDER) > 8:
            _LIVE.pop(_LIVE_ORDER.pop(0), None)
    prediction = result.prediction
    heads = []
    for key, name in [
        ("informative", "Informative"),
        ("humanitarian", "Humanitarian signal"),
        ("severity", "Damage severity"),
        ("disaster_type", "Disaster type"),
    ]:
        output = getattr(prediction, key)
        heads.append(
            {
                "key": key,
                "name": name,
                "label": _human(output.label) if output.label else None,
                "status": output.status,
                "probability": output.probs.get(output.label) if output.label else None,
            }
        )
    return {
        "token": token,
        "text": result.text_clean,
        "image": _image_url(presentation_image(image, presentation=presentation, max_side=640)) if image else None,
        "heads": heads,
        "abstain": prediction.abstain,
        "abstain_reason": prediction.abstain_reason,
        "modality": prediction.modality_weights,
        "model": prediction.model_id,
        "origin": origin,
        "non_english": non_english_warning(text),
    }


@lru_cache(maxsize=1)
def _index(graphic_scorer, provenance_index):
    return provenance_index.load(
        index_path=ROOT / "data" / "processed" / "provenance" / "index.parquet",
        feature_dir=ROOT / "data" / "processed" / "features",
        manifest_path=ROOT / "data" / "processed" / "manifest.parquet",
        graphic_scorer=graphic_scorer(device="cpu"),
    )


def _explain(payload: dict) -> dict:
    from sakshi.app.live import explain, overlay

    result = _LIVE.get(str(payload.get("token")))
    head = str(payload.get("head"))
    presentation = payload.get("presentation", True) is not False
    if result is None or head not in {"informative", "humanitarian", "severity", "disaster_type"}:
        raise ValueError("Explanation input expired or invalid")
    with _INFERENCE_LOCK:
        words, image = explain(_predictor(), result, head)
    return {
        "words": [{"text": display_text(word), "score": score} for word, score in words.words] if words else [],
        "steps": words.steps if words else None,
        "heatmap": _image_url(overlay(result.image, image.heatmap, presentation=presentation), max_side=800)
        if image and result.image
        else None,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' data:; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            if url.path in {"/", "/app.css", "/app.js", "/favicon.svg"}:
                filename = {"/": "index.html", "/app.css": "app.css", "/app.js": "app.js", "/favicon.svg": "favicon.svg"}[
                    url.path
                ]
                content_type = {
                    "index.html": "text/html",
                    "app.css": "text/css",
                    "app.js": "text/javascript",
                    "favicon.svg": "image/svg+xml",
                }[filename]
                self._send((WEB / filename).read_bytes(), content_type + "; charset=utf-8")
            elif url.path == "/health":
                self._json({"status": "ok"})
            elif url.path == "/api/bootstrap":
                self._json(_summary())
            elif url.path == "/api/replay":
                self._json(
                    _replay_view(
                        query.get("event", "hurricane_maria"),
                        int(query.get("window", "0")),
                        query.get("lens", "all"),
                        query.get("presentation", "true") != "false",
                    )
                )
            elif url.path == "/api/sitrep":
                self._json(
                    _sitrep_view(
                        query.get("event", "hurricane_maria"),
                        int(query.get("window", "0")),
                        query.get("presentation", "true") != "false",
                    )
                )
            elif url.path == "/api/review":
                self._json(_review_view(query.get("lens", "pending"), query.get("presentation", "true") != "false"))
            elif url.path == "/api/results":
                self._json(_results_view(query.get("report"), query.get("presentation", "true") != "false"))
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            self._json({"error": display_text(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        if self.headers.get("X-Sakshi-Client") != "fieldroom":
            self._json({"error": "Request rejected"}, HTTPStatus.FORBIDDEN)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 16 * 1024 * 1024:
            self._json({"error": "Request exceeds the 10 MB upload limit"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/classify":
                answer = _classify(payload)
            elif self.path == "/api/explain":
                answer = _explain(payload)
            elif self.path == "/api/review":
                answer = _save_review(str(payload.get("token")), str(payload.get("decision")))
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(answer)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self._json({"error": display_text(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json(
                {"error": "The local model could not complete this request. Check the checkpoint and encoder cache."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main(port: int = 8501) -> None:
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"SAKSHI fieldroom → http://127.0.0.1:{port}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
