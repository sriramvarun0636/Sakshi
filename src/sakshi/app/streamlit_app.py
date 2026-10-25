"""SAKSHI's local, privacy-aware Streamlit field dashboard."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from sakshi import paths
from sakshi.app import data, ui
from sakshi.app.privacy import decode_upload, display_text, escaped, load_thumbnail, non_english_warning, presentation_image
from sakshi.data.events import EVENTS

ROOT = Path(os.environ.get("SAKSHI_APP_ROOT", paths.REPO)).expanduser().resolve()


def _cache_hint() -> None:
    # The team's pinned encoder files live here; the checkpoint alone is insufficient.
    local = Path("/private/tmp/sakshi_hf_cache")
    if "HF_HOME" not in os.environ and local.is_dir():
        os.environ["HF_HOME"] = str(local)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


@st.cache_resource(show_spinner="Loading calibrated classifier on CPU…")
def _predictor(run: str):
    from sakshi.models.predict import Predictor

    _cache_hint()
    return Predictor.from_run(Path(run), device="cpu")


@st.cache_resource(show_spinner="Opening image provenance index…")
def _provenance_index(root: str):
    from sakshi.provenance.index import ClipGraphicScorer, ProvenanceIndex

    _cache_hint()
    base = Path(root)
    return ProvenanceIndex.load(
        index_path=base / "data" / "processed" / "provenance" / "index.parquet",
        feature_dir=base / "data" / "processed" / "features",
        manifest_path=base / "data" / "processed" / "manifest.parquet",
        graphic_scorer=ClipGraphicScorer(device="cpu"),
    )


@st.cache_data(show_spinner=False)
def _manifest(root: str) -> pd.DataFrame:
    return data.manifest(Path(root))


@st.cache_data(show_spinner=False, ttl=60)
def _sources(root: str) -> dict[str, data.AggregateSource]:
    return data.preferred_aggregates(Path(root))


@st.cache_data(show_spinner=False, ttl=60)
def _sitreps(root: str) -> dict[str, data.SitrepSource]:
    return data.sitreps(Path(root))


@st.cache_data(show_spinner=False)
def _tweets(directory: str) -> pd.DataFrame:
    return pd.read_parquet(Path(directory) / "tweets.parquet")


@st.cache_data(show_spinner=False, max_entries=24)
def _window_data(files: tuple[str, ...]) -> list[dict]:
    return [data.read_json(Path(name)) for name in files]


@st.cache_data(show_spinner=False)
def _replay(root: str, event: str) -> pd.DataFrame:
    return data.replay(Path(root), event)


@st.cache_data(show_spinner=False, ttl=60)
def _predictions(root: str) -> tuple[pd.DataFrame, str]:
    return data.classifier_predictions(Path(root))


@st.cache_data(show_spinner=False, max_entries=512)
def _thumbnail(path: str | None, presentation: bool, max_side: int = 256):
    return load_thumbnail(Path(path) if path is not None else None, presentation=presentation, max_side=max_side)


def _row_thumbnail(row: dict | None, presentation: bool, max_side: int = 256):
    path = data.image_path(ROOT, row)
    return _thumbnail(str(path) if path is not None else None, presentation, max_side)


def _present(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    result = str(value).strip()
    return result if result and result.lower() not in {"nan", "none", "null"} else None


def _human(value: object) -> str:
    return (_present(value) or "not available").replace("_", " ").capitalize()


def _pct(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "—"


def _lookup(frame: pd.DataFrame) -> dict[str, dict]:
    return {str(row["sample_id"]): row for row in frame.to_dict("records")} if not frame.empty else {}


def _origin(result, presentation: bool) -> None:
    if result is None:
        return
    tone = (
        "alert"
        if result.status.value == "recycled"
        else "warn"
        if result.status.value in {"seen_before", "match_undated", "cross_event_concurrent"}
        else ""
    )
    st.markdown(
        '<div class="panel"><div class="kicker">IMAGE PROVENANCE</div><h3 class="panel-title" style="margin-top:.5rem">'
        f"{ui.pill(result.status.value, tone)}</h3>"
        '<p class="panel-copy">The image is compared with the indexed archive. '
        "An event and UTC time enable the full temporal decision.</p></div>",
        unsafe_allow_html=True,
    )
    if result.origin_post_id is None:
        return
    lookup = _lookup(_manifest(str(ROOT)))
    original = lookup.get(result.origin_post_id)
    col_image, col_details = st.columns([1, 2], vertical_alignment="center")
    with col_image:
        thumb = _row_thumbnail(original, presentation)
        if thumb is not None:
            st.image(thumb, width="stretch")
        else:
            st.caption("Original image unavailable on this laptop")
    with col_details:
        st.markdown("**Earliest indexed match**")
        st.write(_human(result.origin_event))
        st.caption(result.origin_first_seen.strftime("%d %b %Y · %H:%M UTC") if result.origin_first_seen else "Date unavailable")
        if result.similarity is not None:
            st.caption(f"{result.tier or 'visual'} match · similarity {result.similarity:.3f}")


def _prediction_panel(prediction) -> None:
    heads = [
        ("Informative", prediction.informative),
        ("Humanitarian signal", prediction.humanitarian),
        ("Damage severity", prediction.severity),
        ("Disaster type", prediction.disaster_type),
    ]
    cards = []
    for name, output in heads:
        if output.status == "unavailable":
            value, probability = "Not applicable (no image)", "n/a"
        elif output.status == "gated":
            value, probability = "Gated by informativeness", "n/a"
        else:
            value = _human(output.label)
            probability = _pct(output.probs.get(output.label)) if output.label else "—"
        detail = probability + " calibrated probability" if probability != "n/a" else probability
        cards.append(
            f'<div class="decision"><div class="name">{escaped(name)}</div><div class="value">{escaped(value)}</div>'
            f'<div class="prob">{escaped(detail)}</div></div>'
        )
    badge = (
        ui.pill("Abstained · " + (prediction.abstain_reason or "review"), "warn")
        if prediction.abstain
        else ui.pill("Confident classification")
    )
    st.markdown(
        f'<div style="margin-bottom:.8rem">{badge}</div><div class="decision-grid">{"".join(cards)}</div>', unsafe_allow_html=True
    )
    text_weight = max(0.0, min(1.0, prediction.modality_weights["text"]))
    image_weight = max(0.0, min(1.0, prediction.modality_weights["image"]))
    st.markdown(
        '<div class="section-heading"><h2>Modality reliance</h2><small>Learned fusion gate</small></div>'
        f'<div class="reliance"><span class="text" style="width:{text_weight * 100:.2f}%"></span>'
        f'<span class="image" style="width:{image_weight * 100:.2f}%"></span></div>'
        f'<div class="micro" style="color:#9cb3b4">TEXT {_pct(text_weight)} &nbsp; · &nbsp; IMAGE {_pct(image_weight)}</div>',
        unsafe_allow_html=True,
    )


def _explanation(result, predictor, presentation: bool) -> None:
    from sakshi.app.live import explain, overlay

    available = [
        name
        for name in ("informative", "humanitarian", "severity", "disaster_type")
        if getattr(result.prediction, name).status == "predicted"
    ]
    if not available:
        return
    ui.section("Evidence map", "Computed on demand · CPU")
    head = st.selectbox("Explain a decision", available, format_func=_human, key="explain_head")
    if st.button("Reveal word and image evidence", key="explain_button", width="stretch"):
        try:
            with st.spinner("Tracing model evidence…"):
                st.session_state.live_explanation = (result.prediction.post_id, head, *explain(predictor, result, head))
        except Exception:
            st.session_state.live_explanation = None
            st.warning("This explanation could not be computed on this laptop. Classification remains available.")
    saved = st.session_state.get("live_explanation")
    if not saved or saved[:2] != (result.prediction.post_id, head):
        st.caption(
            "Choose a head and reveal its evidence map. Word attribution applies to text decisions; "
            "the heatmap applies to image decisions."
        )
        return
    words, image = saved[2:]
    columns = st.columns(2)
    with columns[0]:
        st.markdown('<div class="kicker">01 / WORD ATTRIBUTION</div>', unsafe_allow_html=True)
        if words is None:
            st.caption("No text attribution applies to this decision.")
        else:
            magnitude = max((max(0.0, score) for _, score in words.words), default=0.0) or 1.0
            chips = []
            for word, score in words.words:
                opacity = min(0.52, max(0, score) / magnitude * 0.48)
                chips.append(
                    f'<span class="word-chip" style="background:rgba(184,233,134,{opacity:.3f});'
                    f'border-color:rgba(184,233,134,{opacity + 0.2:.3f})">{escaped(word)}</span>'
                )
            st.markdown('<div class="panel">' + " ".join(chips) + "</div>", unsafe_allow_html=True)
            st.caption(f"Integrated Gradients · {words.steps} steps")
    with columns[1]:
        st.markdown('<div class="kicker">02 / IMAGE FOCUS</div>', unsafe_allow_html=True)
        if image is None or result.image is None:
            st.caption("No image heatmap applies to this decision.")
        else:
            try:
                st.image(overlay(result.image, image.heatmap, presentation=presentation), width="stretch")
                st.caption("Grad-CAM · centre crop projected onto the source image")
            except Exception:
                st.caption("The heatmap could not be displayed safely.")


def classify_page(presentation: bool) -> None:
    from sakshi.app.live import classify

    ui.hero(
        "01",
        "Read the signal.",
        "Classify a post, see what the model relied on, and check whether its image appeared earlier in the archive.",
    )
    run = data.classifier_run(ROOT)
    if run is None:
        ui.empty(
            "01",
            "Classifier checkpoint missing",
            "A trained and calibrated checkpoint is needed for live classification.",
            "sakshi-model train --config configs/classifier/full.yaml",
        )
        return
    st.markdown(f"{ui.pill('CPU inference')}{ui.pill('calibrated model')}", unsafe_allow_html=True)
    ui.section("Input desk", "JPEG · PNG · WebP  /  10 MB maximum")
    with st.form("classify_form", clear_on_submit=False):
        left, right = st.columns([1.2, 0.8])
        with left:
            text = st.text_area(
                "Post text", height=155, placeholder="Paste a crisis post here. Handles and links are removed before display."
            )
        with right:
            upload = st.file_uploader(
                "Add an image", type=["jpg", "jpeg", "png", "webp"], help="The longest side is reduced to 1024 pixels."
            )
            st.caption("Text-only and image-only posts are supported.")
        with st.expander("Optional event context", expanded=False):
            event = st.selectbox("Treat as a post from", ["No event context", *EVENTS], format_func=_human)
            date = st.date_input("Post date (UTC)", value=datetime(2017, 9, 25).date())
            clock = st.time_input("Post time (UTC)", value=datetime(2017, 9, 25, 12).time())
        submitted = st.form_submit_button("Analyze post  →", type="primary", width="stretch")
    if submitted:
        try:
            image = decode_upload(upload.getvalue(), filename=upload.name) if upload is not None else None
            if non_english_warning(text):
                st.warning("Model trained on English. Interpret this prediction with care.")
            with st.spinner("Reading evidence…"):
                predictor = _predictor(str(run))
                result = classify(predictor, text, image)
            st.session_state.live_result = result
            st.session_state.live_explanation = None
            st.session_state.live_provenance = None
            if image is not None:
                try:
                    index = _provenance_index(str(ROOT))
                    index.live_featurizer = predictor._live_clip
                    context_time = datetime.combine(date, clock, tzinfo=timezone.utc) if event != "No event context" else None
                    st.session_state.live_provenance = index.check(
                        image=image, event=event if context_time else None, time=context_time
                    )
                except Exception:
                    st.session_state.provenance_unavailable = True
                else:
                    st.session_state.provenance_unavailable = False
        except ValueError as exc:
            st.error(display_text(exc))
            st.session_state.live_result = None
        except Exception:
            st.error("The classifier could not load. Check the local encoder cache and checkpoint, then retry.")
            st.session_state.live_result = None
    result = st.session_state.get("live_result")
    if result is None:
        ui.empty(
            "READY",
            "Evidence begins with a post",
            "Add text, an image, or both. The calibrated model will return four decisions and an abstention signal.",
        )
        return
    ui.section("Model reading", f"Model {result.prediction.model_id} · calibrated")
    if result.image is not None:
        left, right = st.columns([0.7, 1.3], vertical_alignment="top")
        with left:
            try:
                st.image(presentation_image(result.image, presentation=presentation, max_side=512), width="stretch")
            except Exception:
                st.caption("Image hidden because presentation-safe rendering is unavailable.")
        with right:
            _prediction_panel(result.prediction)
    else:
        _prediction_panel(result.prediction)
    if result.text_clean:
        st.markdown(
            '<div class="panel" style="margin-top:.85rem"><div class="kicker">CLEANED POST TEXT</div>'
            f'<p class="panel-copy">{escaped(result.text_clean)}</p></div>',
            unsafe_allow_html=True,
        )
    if result.image is not None:
        ui.section("Image origin", "Earliest indexed visual match")
        if st.session_state.get("provenance_unavailable"):
            ui.empty(
                "INDEX",
                "Provenance index unavailable",
                "Build the image index before inspecting prior appearances.",
                "sakshi-provenance build-index",
            )
        else:
            _origin(st.session_state.get("live_provenance"), presentation)
    _explanation(result, _predictor(str(run)), presentation)
    st.button(
        "Generate evidence rationale",
        disabled=True,
        help="Available after a matching evidence packet and Qwen/NLI models are prepared.",
    )
    st.caption("Requires same-model evidence packets plus local Qwen and NLI weights. See docs/GENERATION.md.")


def _story(row: dict, lookup: dict[str, dict], origins: dict[str, dict], presentation: bool) -> None:
    sample_ids = row.get("sample_ids")
    if sample_ids is None:
        sample_ids = []
    representative = lookup.get(str(row.get("representative_id")))
    status = _present(row.get("provenance_status")) or (
        "origin unassessed" if _present((representative or {}).get("image_path")) else "no image"
    )
    recycled = status == "recycled"
    original = next(
        (origins.get(str(sample_id)) for sample_id in sample_ids if origins.get(str(sample_id), {}).get("status") == "recycled"),
        None,
    )
    origin_row = lookup.get(str(original.get("origin_post_id"))) if original else None
    badges = ui.pill(_present(row.get("humanitarian_label")) or "uncategorized", "dim")
    severity = _present(row.get("severity_label"))
    if severity:
        badges += ui.pill(severity, "warn" if severity == "severe_damage" else "dim")
    badges += ui.pill(status, "alert" if recycled else "warn" if status in {"cross_event_concurrent", "match_undated"} else "dim")
    story = escaped(_present(row.get("text_clean")) or "Image-only post")
    card = (
        f'<div class="story-card {"recycled" if recycled else ""}"><div class="meta">'
        f"{escaped(str(row.get('posted_at'))[:16].replace('T', ' '))} UTC / OBSERVED POST</div>"
        f'<div class="story {"struck" if recycled else ""}">{story}</div><div style="margin-top:.8rem">{badges}</div></div>'
    )
    if recycled:
        posted, earlier = st.columns(2, vertical_alignment="top")
        with posted:
            thumb = _row_thumbnail(representative, presentation)
            if thumb is not None:
                st.image(thumb, width=256)
            else:
                st.caption("Image unavailable")
            st.markdown(card, unsafe_allow_html=True)
        with earlier:
            thumb = _row_thumbnail(origin_row, presentation)
            if thumb is not None:
                st.image(thumb, width=256)
            else:
                st.caption("Original image unavailable")
            event = _human(original.get("origin_event")) if original else "Unknown origin"
            date = str(original.get("origin_first_seen") or "Date unavailable")[:16]
            st.markdown(
                '<div class="story-card"><div class="meta">EARLIEST INDEXED APPEARANCE</div>'
                f'<div class="story">{escaped(event)}</div>'
                f'<div class="origin-note">{escaped(date)} UTC · excluded from evidence</div></div>',
                unsafe_allow_html=True,
            )
    else:
        thumb_col, text_col = st.columns([0.2, 0.8], vertical_alignment="center")
        with thumb_col:
            thumb = _row_thumbnail(representative, presentation)
            if thumb is not None:
                st.image(thumb, width="stretch")
            else:
                st.markdown('<div class="panel-copy">IMAGE<br>UNAVAILABLE</div>', unsafe_allow_html=True)
        with text_col:
            st.markdown(card, unsafe_allow_html=True)


def _timeline(aggregates: list[dict], current: int) -> None:
    values = [int(item["counts"]["tweets"]) for item in aggregates]
    maximum = max(values, default=1) or 1
    count = len(values)
    step = 900 / count
    bars = []
    for index, value in enumerate(values):
        height = max(3, 51 * value / maximum)
        x = index * step + 1
        y = 59 - height
        fill = (
            "#B8E986"
            if index == current
            else "#FD8A70"
            if int(aggregates[index]["counts"]["excluded_provenance"]["recycled"])
            else "#537E7D"
        )
        opacity = "1" if index == current else ".78"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(2, step - 2):.2f}" '
            f'height="{height:.2f}" fill="{fill}" opacity="{opacity}"/>'
        )
    first = aggregates[0]["window"]["start"][:10]
    last = aggregates[-1]["window"]["start"][:10]
    st.markdown(
        '<div class="timeline-panel"><div class="timeline-top"><span>OBSERVED POSTS / WINDOW</span>'
        "<span>REPLAY INTENSITY</span></div>"
        '<svg viewBox="0 0 900 64" preserveAspectRatio="none" role="img" aria-label="Post volume by event window">'
        '<path d="M0 60H900" stroke="#43605F"/>' + "".join(bars) + "</svg>"
        f'<div class="timeline-bottom"><span>{escaped(first)}</span><span>{escaped(last)}</span></div></div>',
        unsafe_allow_html=True,
    )


def replay_page(presentation: bool) -> None:
    ui.hero(
        "02",
        "Trace the event.",
        "Move through an event window by window. Watch the signal accumulate while recycled imagery "
        "is separated from new evidence.",
        art_label="CHRONOLOGY / SOURCE / ORIGIN",
    )
    sources = _sources(str(ROOT))
    if not sources:
        ui.empty(
            "02",
            "Event aggregates missing",
            "Prepare provenance replay and aggregation before opening the event timeline.",
            "sakshi-provenance replay --event hurricane_maria && sakshi-aggregate --config configs/aggregation/default.yaml",
        )
        return
    events = sorted(sources)
    event = st.selectbox(
        "Event",
        events,
        index=events.index("hurricane_maria") if "hurricane_maria" in events else 0,
        format_func=_human,
        key="replay_event",
    )
    source = sources[event]
    preview = not source.evidence_ready
    st.markdown(
        ui.pill(
            "Evidence ready" if source.evidence_ready else "Baseline preview · unverified evidence",
            "" if source.evidence_ready else "warn",
        )
        + ui.pill(f"{len(source.windows)} windows", "dim")
        + ui.pill(source.model_id, "dim"),
        unsafe_allow_html=True,
    )
    if preview:
        st.caption(
            "These windows use the CLIP zero-shot baseline. They show the replay interface and provenance decisions; "
            "they are not final classifier or sitrep evidence."
        )
    try:
        tweets = _tweets(str(source.directory))
        aggregates = _window_data(tuple(str(path) for path in source.windows))
        provenance = _replay(str(ROOT), event)
        lookup = _lookup(_manifest(str(ROOT)))
        origins = {str(row["post_id"]): row for row in provenance.to_dict("records")} if not provenance.empty else {}
    except (OSError, ValueError):
        ui.empty(
            "02",
            "Replay data unavailable",
            "The aggregate's tweet table or provenance replay could not be read.",
            "sakshi-provenance replay --event hurricane_maria",
        )
        return
    if st.session_state.get("replay_event_loaded") != event:
        st.session_state.replay_event_loaded = event
        st.session_state.replay_index = 0
        st.session_state.replay_playing = False
        st.session_state.replay_tick = time.monotonic()

    @st.fragment(run_every=0.33 if st.session_state.get("replay_playing") else None)
    def _player() -> None:
        speed = st.session_state.get("replay_speed", 1)
        if st.session_state.get("replay_playing") and time.monotonic() - st.session_state.get("replay_tick", 0) >= 1 / speed:
            st.session_state.replay_index = (st.session_state.replay_index + 1) % len(source.windows)
            st.session_state.replay_tick = time.monotonic()
        controls = st.columns([0.17, 0.63, 0.2], vertical_alignment="bottom")
        with controls[0]:
            if st.button("Ⅱ  Pause" if st.session_state.replay_playing else "▶  Play", key="replay_toggle", width="stretch"):
                st.session_state.replay_playing = not st.session_state.replay_playing
                st.session_state.replay_tick = time.monotonic()
                st.rerun()
        with controls[1]:
            index = st.slider("Window", 0, len(source.windows) - 1, key="replay_index", format="%d")
        with controls[2]:
            st.select_slider("Speed", options=[1, 2, 3], value=speed, format_func=lambda n: f"{n}×", key="replay_speed")
        aggregate = aggregates[index]
        begin = pd.Timestamp(aggregate["window"]["start"])
        end = pd.Timestamp(aggregate["window"]["end"])
        _timeline(aggregates, index)
        cumulative = [item["counts"] for item in aggregates[: index + 1]]
        ui.metrics(
            [
                ("Posts observed", f"{sum(item['tweets'] for item in cumulative):,}", "through this window"),
                ("Informative", f"{sum(item['informative'] for item in cumulative):,}", "running count"),
                (
                    "Recycled images",
                    f"{sum(item['excluded_provenance']['recycled'] for item in cumulative):,}",
                    "excluded from evidence",
                ),
                ("Awaiting review", f"{sum(item['abstained'] for item in cumulative):,}", "running count"),
            ]
        )
        ui.section(
            f"{begin.strftime('%d %b %Y')} / {event.replace('_', ' ').upper()}",
            f"Window {index + 1:02d} of {len(source.windows):02d} · "
            f"{begin.strftime('%d %b %H:%M')} → {end.strftime('%d %b %H:%M')} UTC",
        )
        posted = pd.to_datetime(tweets["posted_at"], utc=True)
        feed = tweets.loc[(posted >= begin) & (posted < end)].sort_values("posted_at", kind="stable")
        lens = st.radio("Feed lens", ["All posts", "Flagged imagery", "Severe damage"], horizontal=True, key="replay_lens")
        if lens == "Flagged imagery":
            feed = feed.loc[
                feed.provenance_status.isin(["recycled", "cross_event_concurrent", "match_undated", "template_match"])
            ]
        elif lens == "Severe damage":
            feed = feed.loc[feed.severity_label == "severe_damage"]
        if feed.empty:
            ui.empty("FEED", "No posts in this lens", "Move to another window or switch the feed lens.")
        else:
            st.caption(f"Showing {min(len(feed), 8)} of {len(feed)} posts in this window")
            for row in feed.head(8).to_dict("records"):
                _story(row, lookup, origins, presentation)

    _player()


def _citation(pid: str, report: dict, lookup: dict[str, dict], presentation: bool) -> None:
    sample_ids = report.get("input", {}).get("sample_ids", {}).get(pid, [])
    row = next((lookup.get(str(sample_id)) for sample_id in sample_ids if lookup.get(str(sample_id)) is not None), None)
    with st.expander(f"{pid}  ·  cited post"):
        if row is None:
            st.caption("Cited post is unavailable on this laptop.")
            return
        st.write(display_text(row.get("text")) or "Image-only post")
        image = _row_thumbnail(row, presentation)
        if image is not None:
            st.image(image, width=256)
        else:
            st.caption("Image unavailable")


def sitrep_page(presentation: bool) -> None:
    ui.hero(
        "03",
        "Claims, with receipts.",
        "Every sentence carries its evidence trail. Accepted claims stay visible; rejected claims keep "
        "the gates that stopped them.",
        art_label="CLAIM / CITATION / VERIFICATION",
    )
    sources = _sitreps(str(ROOT))
    if not sources:
        ui.empty(
            "03",
            "Verified situation reports are pending",
            "No completed generation artifacts are present. The page will populate when evidence-ready aggregates, "
            "matching packets, and the verifier are available.",
            "sakshi-generate sitreps --config configs/generation/sitrep.yaml "
            "--aggregate-dir <evidence-ready-aggregate> --packets <matching-packets.jsonl>",
        )
        previews = _sources(str(ROOT))
        if previews:
            event = "hurricane_maria" if "hurricane_maria" in previews else next(iter(previews))
            source = previews[event]
            selected = data.read_json(source.windows[min(5, len(source.windows) - 1)])
            ui.section("Computed system lines", "Baseline preview · no generated claims")
            for line in selected.get("system_lines", []):
                st.markdown(
                    f'<div class="story-card"><div class="kicker">SYSTEM / {escaped(event.replace("_", " "))}</div>'
                    f'<div class="story" style="margin-top:.45rem">{escaped(line)}</div></div>',
                    unsafe_allow_html=True,
                )
        return
    event = st.selectbox("Event", sorted(sources), format_func=_human, key="sitrep_event")
    source = sources[event]
    window = st.select_slider(
        "Window",
        options=list(range(len(source.windows))),
        format_func=lambda n: source.windows[n].stem[:8],
        key="sitrep_window",
    )
    report = data.read_json(source.windows[window])
    sentences = report.get("sentences", [])
    accepted = [item for item in sentences if item.get("accepted")]
    rejected = [item for item in sentences if not item.get("accepted")]
    rates = report.get("verification_rate", {})
    ui.metrics(
        [
            ("Accepted claims", str(len(accepted)), "citation-backed"),
            ("Rejected claims", str(len(rejected)), "gates preserved"),
            ("Verification / initial", _pct(rates.get("initial")), "round zero"),
            ("Verification / final", _pct(rates.get("final")), "after one rewrite"),
        ]
    )
    st.markdown(
        ui.pill(source.mode, "dim")
        + ui.pill(report["models"]["classifier"], "dim")
        + ui.pill(report["models"]["llm"], "dim")
        + ui.pill(report["models"]["nli"], "dim"),
        unsafe_allow_html=True,
    )
    ui.section("System observations", report["window"]["start"][:16].replace("T", " ") + " UTC")
    for line in report.get("system_lines", []):
        st.markdown(
            '<div class="story-card"><div class="kicker">COMPUTED / SOURCE WINDOW</div>'
            f'<div class="story" style="margin-top:.45rem">{escaped(line)}</div></div>',
            unsafe_allow_html=True,
        )
    lookup = _lookup(_manifest(str(ROOT)))
    ui.section("Accepted claims", "Open any citation to inspect the supporting post")
    if not accepted:
        ui.empty(
            "CLAIMS",
            "No claim cleared verification",
            "System observations remain available above. No unsupported sentence is promoted as fact.",
        )
    for sentence in accepted:
        citations = "".join(ui.pill(pid, "dim") for pid in sentence.get("citations", []))
        st.markdown(
            f'<div class="story-card"><div class="kicker">VERIFIED / {escaped(sentence.get("id"))}</div>'
            f'<div class="story" style="margin:.5rem 0 .8rem">{escaped(sentence.get("text"))}</div>{citations}</div>',
            unsafe_allow_html=True,
        )
        for pid in sentence.get("citations", []):
            _citation(pid, report, lookup, presentation)
    ui.section("Stopped by verification", "Rejected sentences remain visible for audit")
    if not rejected:
        st.caption("No rejected sentences in this window.")
    for sentence in rejected:
        failed = [name for name, value in sentence.get("gates", {}).items() if not value.get("passed", False)]
        gates = "".join(ui.pill(f"{gate} gate", "alert") for gate in failed)
        st.markdown(
            '<div class="story-card recycled"><div class="kicker" style="color:#fd8a70">'
            f"REJECTED / {escaped(sentence.get('id'))}</div>"
            f'<div class="story struck" style="margin:.5rem 0 .8rem">{escaped(sentence.get("text"))}</div>{gates}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Gate details"):
            for name, value in sentence.get("gates", {}).items():
                st.write(
                    f"{name.capitalize()}: {'passed' if value.get('passed') else 'failed'} — {display_text(value.get('detail'))}"
                )


def _review_items(root: Path) -> tuple[list[dict], str]:
    manifest = _lookup(_manifest(str(root)))
    predictions, source = _predictions(str(root))
    items: dict[str, dict] = {}
    if not predictions.empty and "abstain" in predictions:
        for row in predictions.loc[predictions.abstain].to_dict("records"):
            sample_id = str(row["sample_id"])
            items[sample_id] = {
                "sample_id": sample_id,
                "kind": "abstained",
                "reason": str(row.get("abstain_reason") or "low confidence"),
                "row": manifest.get(sample_id),
                "source": source,
            }
    for event in EVENTS:
        replay = _replay(str(root), event)
        if replay.empty:
            continue
        flagged = replay.loc[replay.status.isin(["match_undated", "cross_event_concurrent"])]
        for row in flagged.to_dict("records"):
            sample_id = str(row["post_id"])
            current = items.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "kind": str(row["status"]),
                    "reason": "Image origin needs human review",
                    "row": manifest.get(sample_id),
                    "source": "provenance replay",
                },
            )
            if current["kind"] == "abstained":
                current["reason"] += " · " + str(row["status"])
    return sorted(items.values(), key=lambda item: (item["kind"] != "abstained", item["kind"], item["sample_id"])), source


def _decisions(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            result[str(entry["sample_id"])] = entry
        except (ValueError, KeyError, TypeError):
            continue
    return result


def _save_decision(path: Path, sample_id: str, decision: str, reason: str) -> None:
    if decision not in {"confirm", "dismiss"}:
        raise ValueError("invalid review decision")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sample_id": sample_id,
        "decision": decision,
        "reason": reason,
        "reviewed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def review_page(presentation: bool) -> None:
    ui.hero(
        "04",
        "Keep a human in the loop.",
        "Uncertain classifications and image origins go to a local review lane. Every decision is recorded "
        "without exposing post identifiers on screen.",
        art_label="TRIAGE / INSPECT / DECIDE",
    )
    items, source = _review_items(ROOT)
    decision_path = ROOT / "runs" / "review" / "decisions.jsonl"
    decisions = _decisions(decision_path)
    pending = [item for item in items if item["sample_id"] not in decisions]
    ui.metrics(
        [
            ("In review lane", str(len(items)), "flagged records"),
            ("Awaiting decision", str(len(pending)), "local queue"),
            ("Reviewed", str(len(items) - len(pending)), "confirm or dismiss"),
            ("Classifier source", "DEV" if "dev" in source else "FULL" if source != "unavailable" else "—", source),
        ]
    )
    if not items:
        ui.empty(
            "04",
            "No review records yet",
            "Produce a trained-classifier prediction table and provenance replay to populate the queue.",
            "sakshi-model predict --run <classifier-run> --rows all && sakshi-provenance replay --event hurricane_maria",
        )
        return
    if "dev" in source:
        st.caption(
            "The abstention lane currently uses a small trained-model development sample. Generate full classifier "
            "predictions to review the complete event population."
        )
    lens = st.radio("Review lens", ["Pending", "All", "Abstained", "Origin uncertain"], horizontal=True)
    if lens == "Pending":
        shown = pending
    elif lens == "Abstained":
        shown = [item for item in items if item["kind"] == "abstained"]
    elif lens == "Origin uncertain":
        shown = [item for item in items if item["kind"] != "abstained"]
    else:
        shown = items
    ui.section("Review records", f"Showing {min(12, len(shown))} of {len(shown)}")
    if not shown:
        ui.empty("QUEUE", "This lane is clear", "Choose another lens or return after new predictions are prepared.")
    for index, item in enumerate(shown[:12]):
        record = item["row"] or {}
        col_image, col_body = st.columns([0.19, 0.81], vertical_alignment="center")
        with col_image:
            thumb = _row_thumbnail(record, presentation)
            if thumb is not None:
                st.image(thumb, width="stretch")
            else:
                st.caption("Image unavailable")
        with col_body:
            status = decisions.get(item["sample_id"], {}).get("decision")
            badges = ui.pill(item["kind"], "warn") + (ui.pill(status, "dim") if status else "")
            st.markdown(
                f'<div class="story-card"><div class="meta">{escaped(_human(record.get("event")))} / '
                f"REVIEW RECORD {index + 1:02d}</div>"
                f'<div class="story">{escaped(_present(record.get("text")) or "Image-only post")}</div>'
                f'<div style="margin-top:.8rem">{badges}</div><div class="origin-note">{escaped(item["reason"])}</div></div>',
                unsafe_allow_html=True,
            )
            if status is None:
                a, b = st.columns(2)
                with a:
                    if st.button("Confirm", key=f"confirm_{item['sample_id']}", width="stretch"):
                        _save_decision(decision_path, item["sample_id"], "confirm", item["reason"])
                        st.rerun()
                with b:
                    if st.button("Dismiss", key=f"dismiss_{item['sample_id']}", width="stretch"):
                        _save_decision(decision_path, item["sample_id"], "dismiss", item["reason"])
                        st.rerun()


def results_page(presentation: bool) -> None:
    ui.hero(
        "05",
        "The record, in view.",
        "Read the evaluation record directly from committed reports. Development checks and final research outputs "
        "stay clearly distinguished.",
        art_label="METHOD / RESULT / LIMIT",
    )
    reports = data.report_files(ROOT)
    classifier = data.classifier_run(ROOT)
    sources = _sources(str(ROOT))
    replays = list((ROOT / "data" / "processed" / "provenance").glob("replay_*.parquet"))
    ui.metrics(
        [
            ("Classifier checkpoint", "01" if classifier else "00", "calibrated run" if classifier else "pending"),
            ("Event replays", f"{len(replays):02d}", "precomputed provenance"),
            (
                "Aggregate events",
                f"{len(sources):02d}",
                "evidence-ready" if any(s.evidence_ready for s in sources.values()) else "demo baseline",
            ),
            ("Result notes", f"{len(reports):02d}", "read-only files"),
        ]
    )
    if not reports:
        ui.empty(
            "05",
            "Result tables missing",
            "The results workspace reads the reports directory without running evaluation jobs.",
            "sakshi-model evaluate --run <frozen-classifier-run>",
        )
        return
    ui.section("Research record", "Report files are rendered read-only")
    titles = {path.stem: path.stem.replace("_", " ").title() for path in reports}
    selected = st.selectbox("Open report", reports, format_func=lambda path: titles[path.stem])
    if "dev" in selected.stem or "demo" in selected.stem:
        st.markdown(ui.pill("Development / demo result", "warn"), unsafe_allow_html=True)
    else:
        st.markdown(ui.pill("Reported result", "dim"), unsafe_allow_html=True)
    # Preserve Markdown line structure while applying the same display sanitizer as posts.
    safe_markdown = "\n".join(display_text(line) for line in selected.read_text(encoding="utf-8").splitlines())
    with st.container(border=True):
        st.markdown(safe_markdown)
    figures = sorted((ROOT / "reports" / "figures").glob("*.png"))
    if figures:
        ui.section("Figures", "Development figures are marked as such")
        for figure in figures:
            st.markdown(f'<div class="kicker">{escaped(figure.stem.replace("_", " "))}</div>', unsafe_allow_html=True)
            shown = _thumbnail(str(figure), presentation, max_side=1200)
            if shown is not None:
                st.image(shown, width="stretch")
            else:
                st.caption("Figure hidden because presentation-safe rendering is unavailable.")
            st.caption("Inspect exported figures by hand before presentation or submission.")


def main() -> None:
    ui.setup()
    page, presentation = ui.sidebar()
    ui.topbar()
    if page.startswith("01"):
        classify_page(presentation)
    elif page.startswith("02"):
        replay_page(presentation)
    elif page.startswith("03"):
        sitrep_page(presentation)
    elif page.startswith("04"):
        review_page(presentation)
    else:
        results_page(presentation)
    ui.footer()


if __name__ == "__main__":
    main()
