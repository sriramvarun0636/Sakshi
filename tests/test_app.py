"""WP6 acceptance checks for privacy and absent local artefacts."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sakshi.app.privacy import blur_faces, decode_upload, display_text, escaped, presentation_image

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "sakshi" / "app" / "streamlit_app.py"


def test_post_text_and_rendered_markup_never_expose_handles_or_links():
    raw = 'Need help from @responder near the bridge; https://example.org/trace?user=42 <script>alert("x")</script>'
    cleaned = display_text(raw)
    assert not re.search(r"@\w", cleaned)
    assert "https://" not in cleaned
    assert "example.org" not in cleaned
    assert "[USER]" in cleaned and "[LINK]" in cleaned
    assert "<script>" not in escaped(raw)


def test_upload_validation_and_downscale():
    source = Image.new("RGB", (2200, 1200), "navy")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    decoded = decode_upload(buffer.getvalue(), filename="flood.png")
    assert max(decoded.size) == 1024
    assert decoded.mode == "RGB"
    with pytest.raises(ValueError, match="10 MB"):
        decode_upload(b"x" * (10 * 1024 * 1024 + 1), filename="large.png")
    with pytest.raises(ValueError, match="JPEG, PNG, or WebP"):
        decode_upload(b"not an image", filename="fake.png")
    with pytest.raises(ValueError, match="JPEG, PNG, or WebP"):
        decode_upload(buffer.getvalue(), filename="disguised.exe")


def test_presentation_mode_blurs_synthetic_face_and_uses_required_detector_settings():
    checkerboard = ((np.indices((128, 128)).sum(axis=0) % 2) * 255).astype(np.uint8)
    rgb = np.repeat(checkerboard[:, :, None], 3, axis=2)
    calls = []

    class Detector:
        def detectMultiScale(self, gray, **kwargs):
            calls.append(kwargs)
            assert gray.shape == (128, 128)
            return [(32, 32, 64, 64)]

    source = Image.fromarray(rgb)
    output = blur_faces(source, detector=Detector())
    assert calls == [{"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (24, 24)}]
    assert np.var(np.asarray(output)[40:88, 40:88]) < np.var(rgb[40:88, 40:88]) * 0.1
    # The actual presentation path also routes through the face-blurring function.
    assert presentation_image(source, presentation=True).size == source.size


def test_review_decisions_are_append_only_and_local(tmp_path):
    from sakshi.app.streamlit_app import _decisions, _save_decision

    path = tmp_path / "runs" / "review" / "decisions.jsonl"
    _save_decision(path, "private:sample", "confirm", "uncertain origin")
    _save_decision(path, "private:sample", "dismiss", "reviewed again")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert [line["decision"] for line in lines] == ["confirm", "dismiss"]
    assert _decisions(path)["private:sample"]["decision"] == "dismiss"


def test_all_pages_start_when_artifacts_are_absent(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SAKSHI_APP_ROOT", str(tmp_path))
    pages = ["01  Classify a post", "02  Event replay", "03  Situation report", "04  Review queue", "05  Results"]
    for page in pages:
        app = AppTest.from_file(SCRIPT, default_timeout=30).run()
        if page != pages[0]:
            app.radio[0].set_value(page).run()
        assert not app.exception, page
        visible = " ".join(item.value for item in app.markdown if not item.value.startswith("<style>"))
        assert not re.search(r"@\w", visible), page


def test_existing_replay_and_review_pages_render_without_private_handles(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("SAKSHI_APP_ROOT", raising=False)
    for page in ("02  Event replay", "04  Review queue"):
        app = AppTest.from_file(SCRIPT, default_timeout=30).run()
        app.radio[0].set_value(page).run()
        assert not app.exception, page
        visible = " ".join(item.value for item in app.markdown if not item.value.startswith("<style>"))
        assert not re.search(r"@\w", visible), page
        assert ">NAN<" not in visible.upper(), page


def test_completed_sitrep_renders_claims_citations_and_rejected_gates(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    directory = tmp_path / "runs" / "sitreps" / "model-one" / "hurricane_maria"
    directory.mkdir(parents=True)
    (directory / "complete.json").write_text(json.dumps({"windows": 1, "mode": "loeo"}))
    report = {
        "event": "hurricane_maria",
        "window": {"start": "2017-09-20T00:00:00Z", "end": "2017-09-21T00:00:00Z"},
        "models": {"classifier": "model-one", "llm": "local-llm", "nli": "local-nli"},
        "input": {"sample_ids": {"P01": ["private:sample"]}},
        "system_lines": ["One observation from computed counts."],
        "verification_rate": {"initial": 0.5, "final": 0.5},
        "sentences": [
            {"id": "S1", "accepted": True, "text": "Accepted claim.", "citations": ["P01"], "gates": {}},
            {
                "id": "S2",
                "accepted": False,
                "text": "Rejected claim.",
                "citations": [],
                "gates": {"citation": {"passed": False, "detail": "No citation supplied."}},
            },
        ],
    }
    (directory / "20170920T000000Z.json").write_text(json.dumps(report))
    monkeypatch.setenv("SAKSHI_APP_ROOT", str(tmp_path))
    app = AppTest.from_file(SCRIPT, default_timeout=30).run()
    app.radio[0].set_value("03  Situation report").run()
    assert not app.exception
    visible = " ".join(item.value for item in app.markdown if not item.value.startswith("<style>"))
    assert "Accepted claim." in visible and "Rejected claim." in visible
    assert "P01" in visible and "citation gate" in visible
    assert "private:sample" not in visible


def test_fieldroom_starts_with_no_artifacts(tmp_path, monkeypatch):
    from sakshi.app import web_server

    monkeypatch.setattr(web_server, "ROOT", tmp_path)
    monkeypatch.setattr(web_server, "_aggregates", lambda: {})
    monkeypatch.setattr(web_server, "_sitreps", lambda: {})
    monkeypatch.setattr(web_server, "_review_items", lambda: ([], "unavailable"))
    assert not web_server._summary()["classifier"]
    assert not web_server._replay_view("hurricane_maria", 0, "all", True)["available"]
    assert not web_server._sitrep_view("hurricane_maria", 0, True)["available"]
    assert web_server._review_view("pending", True)["total"] == 0
    assert web_server._results_view(None, True)["reports"] == []


def test_fieldroom_results_show_figures_for_selected_report(tmp_path, monkeypatch):
    from PIL import Image

    from sakshi.app import web_server

    reports = tmp_path / "reports" / "results"
    figures = tmp_path / "reports" / "figures"
    reports.mkdir(parents=True)
    figures.mkdir()
    (reports / "baselines_seed0.md").write_text("# Baselines\n")
    (reports / "classifier_three_seed.md").write_text("# Three seeds\n")
    (reports / "faithfulness_dev.md").write_text("# Development\n")
    for name in ("classifier_seed2_reliability.png", "classifier_seed2_risk_coverage.png", "faithfulness_curves_dev.png"):
        Image.new("RGB", (20, 20), "white").save(figures / name)
    monkeypatch.setattr(web_server, "ROOT", tmp_path)
    selected = web_server._results_view(None, True)
    assert selected["selected"] == "classifier_three_seed"
    assert len(selected["figures"]) == 2
    assert all("SEED 2" in figure["label"] for figure in selected["figures"])
    assert web_server._results_view("baselines_seed0", True)["figures"] == []
    development = web_server._results_view("faithfulness_dev", True)
    assert len(development["figures"]) == 1 and "DEVELOPMENT" in development["figures"][0]["label"]


def test_fieldroom_review_uses_opaque_token_and_sanitized_text(tmp_path, monkeypatch):
    from sakshi.app import web_server

    monkeypatch.setattr(web_server, "ROOT", tmp_path)
    monkeypatch.setattr(
        web_server,
        "_review_items",
        lambda: ([{"id": "private:sample", "kind": "abstained", "reason": "low confidence"}], "dev preview"),
    )
    monkeypatch.setattr(
        web_server,
        "_manifest",
        lambda: {"private:sample": {"text": "Help @rescuer https://example.com", "event": "hurricane_maria"}},
    )
    monkeypatch.setattr(web_server, "_thumbnail", lambda *_: None)
    view = web_server._review_view("pending", True)
    assert "private:sample" not in json.dumps(view)
    assert "@rescuer" not in json.dumps(view)
    assert "https://example.com" not in json.dumps(view)
    assert web_server._save_review(view["items"][0]["token"], "confirm") == {"saved": True}
    entry = json.loads((tmp_path / "runs" / "review" / "decisions.jsonl").read_text())
    assert entry["sample_id"] == "private:sample" and entry["decision"] == "confirm"


def test_review_preview_prefers_context_and_never_renders_nan(tmp_path, monkeypatch):
    from sakshi.app import web_server

    monkeypatch.setattr(web_server, "ROOT", tmp_path)
    monkeypatch.setattr(
        web_server,
        "_review_items",
        lambda: (
            [
                {"id": "cib:image", "kind": "abstained", "reason": "low confidence"},
                {"id": "crisismmd:post", "kind": "abstained", "reason": "low confidence"},
            ],
            "trained classifier",
        ),
    )
    monkeypatch.setattr(
        web_server,
        "_manifest",
        lambda: {
            "cib:image": {"text": float("nan"), "event": float("nan")},
            "crisismmd:post": {"text": "Residents need shelter", "event": "hurricane_maria"},
        },
    )
    monkeypatch.setattr(web_server, "_thumbnail", lambda *_: None)

    view = web_server._review_view("pending", True)
    assert [card["text"] for card in view["items"]] == ["Residents need shelter", "Image-only post"]
    assert view["items"][1]["event"] == "Not available"
