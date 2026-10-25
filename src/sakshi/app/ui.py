"""Quiet, data-first presentation components for the five dashboard views."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sakshi.app.privacy import escaped

_CSS = Path(__file__).with_name("brand.css")


def setup() -> None:
    st.set_page_config(page_title="SAKSHI — Evidence in motion", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar() -> tuple[str, bool]:
    mark = (
        '<svg viewBox="0 0 40 40" fill="none" aria-hidden="true">'
        '<path d="M20 3L37 20 20 37 3 20 20 3Z" stroke="currentColor" stroke-width="1.5"/>'
        '<path d="M20 9v22M9 20h22" stroke="currentColor" stroke-width="1"/>'
        '<circle cx="20" cy="20" r="5" fill="currentColor"/></svg>'
    )
    st.sidebar.markdown(
        f'<div class="brand-lockup"><div class="brand-mark">{mark}</div><div><div class="brand-name">SAKSHI</div>'
        '<div class="brand-sub">EVIDENCE INTELLIGENCE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="nav-caption">WORKSPACES / 01—05</div>', unsafe_allow_html=True)
    names = ["01  Classify a post", "02  Event replay", "03  Situation report", "04  Review queue", "05  Results"]
    page = st.sidebar.radio("Workspace", names, label_visibility="collapsed")
    st.sidebar.divider()
    presentation = st.sidebar.toggle("Presentation mode", value=True, help="Blur detected faces in every displayed image.")
    st.sidebar.markdown(
        '<div class="sidebar-note"><strong>FIELD STANDARD  /  INV–11</strong>'
        "<p>Visible posts are cleaned of handles and links. Presentation mode blurs detected faces. "
        "Review images by hand before exporting figures.</p></div>",
        unsafe_allow_html=True,
    )
    return page, presentation


def topbar(state: str = "LOCAL RESEARCH WORKSPACE") -> None:
    st.markdown(
        '<div class="topline"><div class="left"><span class="live-dot"></span><span class="micro">SAKSHI / '
        f'{escaped(state)}</span></div><span class="micro right">SITUATIONAL AWARENESS  —  HUMAN IN THE LOOP</span></div>',
        unsafe_allow_html=True,
    )


_ARTWORK = {
    "01": """<svg viewBox="0 0 420 235" fill="none" aria-hidden="true">
      <circle cx="219" cy="120" r="94" stroke="#6C9E93" stroke-opacity=".39"/>
      <circle cx="219" cy="120" r="69" stroke="#8DC1AB" stroke-opacity=".4"/>
      <circle cx="219" cy="120" r="43" stroke="#B8E986" stroke-opacity=".59"/>
      <path d="M24 164C84 164 83 79 149 79S218 166 278 166 344 69 410 69" stroke="#B8E986" stroke-width="2"/>
      <path d="M16 186C79 186 103 122 155 122S223 193 281 193 351 105 414 105" stroke="#82D1CC" stroke-opacity=".55"/>
      <path d="M220 18V217M116 120H322" stroke="#8DB8A6" stroke-opacity=".32" stroke-dasharray="3 6"/>
      <circle cx="149" cy="79" r="5" fill="#B8E986"/><circle cx="278" cy="166" r="5" fill="#B8E986"/>
      <circle cx="219" cy="120" r="7" fill="#D8F3BA"/><circle cx="219" cy="120" r="15" stroke="#B8E986" stroke-opacity=".6"/>
      <path d="M67 40h38M67 45h21M343 176h32M343 181h17" stroke="#80B2A6" stroke-opacity=".52"/>
    </svg>""",
    "02": """<svg viewBox="0 0 420 235" fill="none" aria-hidden="true">
      <path d="M23 119H404M23 152H404M23 86H404" stroke="#78A89B" stroke-opacity=".25"/>
      <path d="M56 42v152M133 42v152M210 42v152M287 42v152M364 42v152"
        stroke="#78A89B" stroke-opacity=".22" stroke-dasharray="3 6"/>
      <path d="M22 152H79L110 94H154L188 119H230L260 64H309L344 137H405" stroke="#B8E986" stroke-width="2.2"/>
      <path d="M22 172H79L110 147H154L188 156H230L260 128H309L344 172H405" stroke="#82D1CC" stroke-opacity=".58"/>
      <circle cx="110" cy="94" r="6" fill="#B8E986"/><circle cx="188" cy="119" r="6" fill="#B8E986"/>
      <circle cx="260" cy="64" r="7" fill="#B8E986"/><circle cx="344" cy="137" r="8" fill="#FD8A70"/>
      <circle cx="344" cy="137" r="17" stroke="#FD8A70" stroke-opacity=".5"/>
      <path d="M329 47h30M329 52h17M52 206h310" stroke="#8EBAA9" stroke-opacity=".55"/>
      <path d="M52 202v8M133 202v8M210 202v8M287 202v8M362 202v8" stroke="#8EBAA9"/>
    </svg>""",
    "03": """<svg viewBox="0 0 420 235" fill="none" aria-hidden="true">
      <rect x="55" y="27" width="186" height="174" stroke="#719F96" stroke-opacity=".56"/>
      <path d="M75 51h37M75 72h141M75 84h119M75 108h141M75 120h105M75 146h141M75 158h127" stroke="#82B5A8" stroke-opacity=".61"/>
      <path d="M75 72h79M75 108h119" stroke="#B8E986" stroke-width="2"/>
      <circle cx="281" cy="77" r="20" stroke="#B8E986"/><path d="m271 77 7 7 13-15" stroke="#B8E986" stroke-width="2"/>
      <circle cx="337" cy="136" r="20" stroke="#FD8A70"/><path d="m330 129 14 14m0-14-14 14" stroke="#FD8A70" stroke-width="2"/>
      <path d="M241 77h20M241 136h76M281 97v48" stroke="#729E97" stroke-dasharray="3 5"/>
      <path d="M262 189h96M262 196h62" stroke="#80B2A6" stroke-opacity=".55"/>
    </svg>""",
    "04": """<svg viewBox="0 0 420 235" fill="none" aria-hidden="true">
      <path d="M65 26v183M210 26v183M354 26v183" stroke="#79A89B" stroke-opacity=".35" stroke-dasharray="3 6"/>
      <rect x="42" y="45" width="113" height="43" stroke="#85B6A3"/>
      <rect x="42" y="100" width="113" height="43" stroke="#85B6A3"/>
      <rect x="266" y="45" width="113" height="43" stroke="#B8E986"/>
      <rect x="266" y="100" width="113" height="43" stroke="#FD8A70"/>
      <path d="M60 59h53M60 69h75M60 114h66M60 124h42M284 59h53M284 69h75M284 114h66M284 124h42"
        stroke="#8FBCB0" stroke-opacity=".7"/>
      <path d="M155 67h111M155 122h111" stroke="#82D1CC" stroke-opacity=".65"/>
      <path d="m256 61 10 6-10 6m0 43 10 6-10 6" stroke="#82D1CC"/>
      <circle cx="210" cy="67" r="17" stroke="#B8E986" stroke-opacity=".48"/>
      <circle cx="210" cy="122" r="17" stroke="#FD8A70" stroke-opacity=".5"/>
      <path d="M42 186h337M42 194h235" stroke="#80B2A6" stroke-opacity=".55"/>
    </svg>""",
    "05": """<svg viewBox="0 0 420 235" fill="none" aria-hidden="true">
      <rect x="49" y="35" width="322" height="166" stroke="#78A99C" stroke-opacity=".5"/>
      <path d="M72 64h69M72 80h120M72 112h69M72 128h161M72 160h69M72 176h203" stroke="#789F98" stroke-opacity=".5"/>
      <rect x="213" y="70" width="120" height="11" fill="#B8E986" fill-opacity=".8"/>
      <rect x="190" y="118" width="96" height="11" fill="#82D1CC" fill-opacity=".7"/>
      <rect x="236" y="166" width="99" height="11" fill="#B8E986" fill-opacity=".6"/>
      <path d="M49 201h322M49 35h322" stroke="#B8E986" stroke-opacity=".4"/>
      <circle cx="342" cy="75" r="4" fill="#B8E986"/><circle cx="297" cy="123" r="4" fill="#82D1CC"/>
      <path d="M309 18h62M309 23h31" stroke="#80B2A6" stroke-opacity=".55"/>
    </svg>""",
}


def hero(number: str, title: str, description: str, *, art_label: str = "SIGNAL / EVIDENCE / DECISION") -> None:
    artwork = _ARTWORK.get(number, _ARTWORK["01"])
    st.markdown(
        f'<section class="hero"><div class="hero-copy"><div class="eyebrow">{escaped(number)} / FIELD WORKSPACE</div>'
        f"<h1>{escaped(title)}</h1><p>{escaped(description)}</p></div>"
        f'<div class="hero-art">{artwork}<span class="art-caption">{escaped(art_label)}</span></div></section>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-heading"><h2>{escaped(title)}</h2><small>{escaped(note)}</small></div><div class="rule"></div>',
        unsafe_allow_html=True,
    )


def metrics(items: list[tuple[str, str, str]]) -> None:
    cells = []
    for label, value, detail in items:
        cells.append(
            f'<div class="metric-cell"><div class="label">{escaped(label)}</div><div class="value">{escaped(value)}</div>'
            f'<div class="detail">{escaped(detail)}</div><div class="accent">↗</div></div>'
        )
    st.markdown('<div class="metric-strip">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def pill(label: str, tone: str = "") -> str:
    return f'<span class="pill {tone}">{escaped(label.replace("_", " "))}</span>'


def empty(index: str, title: str, detail: str, command: str | None = None) -> None:
    command_html = f"<code>{escaped(command)}</code>" if command else ""
    st.markdown(
        f'<div class="empty-state"><div class="index">{escaped(index)} / ARTEFACT STATUS</div>'
        f"<h3>{escaped(title)}</h3><p>{escaped(detail)}</p>{command_html}</div>",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="footer-line"><span>SAKSHI / EVIDENCE WITH CONTEXT</span>'
        "<span>LOCAL ONLY · PRIVACY AWARE · HUMAN REVIEW</span></div>",
        unsafe_allow_html=True,
    )
