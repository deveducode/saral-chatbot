"""
SARAL Chatbot — Streamlit UI

A polished chat-style interface wiring together the full RAG pipeline:
  ingest.py -> retrieve.py -> generate.py

Features:
  - PDF upload with automatic indexing
  - Audience / length / style parameter controls
  - Structured output display (slide bullets, speaker script, speaker notes,
    tweet abstract) in styled cards
  - Iterative change-tracking with inline diff highlighting
  - Provenance citation badges linking to source page numbers
  - Conversation logging to markdown

Visual identity aligned to SARAL AI's actual brand (saral.democratiseresearch.in):
  background #111315, minimal sans-serif, restrained single-accent palette.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from ingest import build_index
from generate import (
    generate,
    generate_structured,
    apply_change,
    citation_coverage,
    StructuredOutput,
)
from retrieve import Retriever

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Saral AI — Audience-Adaptive Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — SARAL AI brand palette
#   Base background #111315 (matches saral.democratiseresearch.in's theme-color)
#   Single restrained accent instead of multiple competing colors
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Imports ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Global Dark Theme & Contrast Reset ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #111315 !important;
    color: #F5F5F4 !important;
    font-family: 'Inter', -apple-system, sans-serif;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span, div {
    color: #F5F5F4;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #16181B !important;
    border-right: 1px solid #26282B !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #E4E4E2 !important;
}

/* Form Controls (Inputs, Selectboxes, Textareas) */
textarea, input[type="text"] {
    background-color: #1A1C1F !important;
    color: #F5F5F4 !important;
    border: 1px solid #2A2C2F !important;
    border-radius: 8px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6B6E73 !important;
    box-shadow: 0 0 0 1px #6B6E73 !important;
}

div[data-baseweb="select"] > div {
    background-color: #1A1C1F !important;
    border-color: #2A2C2F !important;
    color: #F5F5F4 !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"] {
    background-color: #1A1C1F !important;
    border: 1px solid #2A2C2F !important;
}
div[data-baseweb="option"] {
    background-color: #1A1C1F !important;
    color: #F5F5F4 !important;
}
div[data-baseweb="option"]:hover {
    background-color: #232528 !important;
}

section[data-testid="stFileUploadDropzone"] {
    background-color: #1A1C1F !important;
    border: 1px dashed #2A2C2F !important;
}

/* ---------- Primary buttons — clean off-white pill, minimal SaaS style ---------- */
div.stButton > button[kind="primary"],
div.stButton > button {
    background-color: #F5F5F4 !important;
    color: #111315 !important;
    border: 1px solid #F5F5F4 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button:hover {
    background-color: #E4E4E2 !important;
    border-color: #E4E4E2 !important;
    color: #111315 !important;
}
div.stButton > button[kind="primary"]:focus,
div.stButton > button:focus {
    box-shadow: 0 0 0 2px rgba(245, 245, 244, 0.35) !important;
}

/* Expanders */
.stExpander {
    background-color: #16181B !important;
    border: 1px solid #26282B !important;
    border-radius: 10px !important;
}

/* ---------- Header — simple "Saral AI" wordmark, matching real branding ---------- */
.saral-header {
    background: #16181B;
    border: 1px solid #26282B;
    padding: 1.75rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.saral-wordmark {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F5F5F4;
    line-height: 1;
    margin-bottom: 0.5rem;
    display: block;
}
.saral-header-subtitle {
    font-size: 1.05rem;
    font-weight: 600;
    color: #C7C8CA;
    margin-bottom: 0.35rem;
}
.saral-header-desc {
    font-size: 0.9rem;
    color: #8E9094;
    margin: 0;
}

/* ---------- Cards — minimal, format-preview style ---------- */
.output-card {
    background-color: #16181B !important;
    border: 1px solid #26282B !important;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    color: #F5F5F4 !important;
    transition: border-color 0.15s ease;
    position: relative;
}
.output-card:hover {
    border-color: #3A3C3F !important;
}
.output-card h3 {
    margin: 0 0 0.85rem 0;
    font-size: 1.0rem;
    font-weight: 600;
    color: #F5F5F4 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.output-card div,
.output-card p,
.output-card span,
.output-card li {
    color: #D4D5D7 !important;
    line-height: 1.6;
    font-size: 0.93rem;
}
.card-badge {
    position: absolute;
    top: 1.1rem;
    right: 1.25rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #8E9094;
    background: #1F2124;
    border: 1px solid #2A2C2F;
    padding: 2px 8px;
    border-radius: 20px;
}

/* ---------- Citation Badges — single restrained accent ---------- */
.citation-badge {
    display: inline-block;
    background: #232528;
    border: 1px solid #35373A;
    color: #D4D5D7 !important;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 6px;
    margin: 0 2px;
    vertical-align: middle;
    letter-spacing: 0.02em;
}

/* ---------- Tweet Card ---------- */
.tweet-card {
    background: #16181B !important;
    border: 1px solid #26282B !important;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    color: #F5F5F4 !important;
    position: relative;
}
.tweet-card div, .tweet-card p {
    color: #D4D5D7 !important;
    line-height: 1.55;
}
.tweet-char-count {
    text-align: right;
    font-size: 0.75rem;
    color: #8E9094 !important;
    margin-top: 0.5rem;
}

/* ---------- Diff Highlighting — keep semantic red/green, muted saturation ---------- */
.diff-container {
    background: #16181B !important;
    border: 1px solid #26282B !important;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 0.83rem;
    line-height: 1.6;
    overflow-x: auto;
}
.diff-line-add {
    color: #8FBF9F !important;
    background: rgba(143, 191, 159, 0.10);
    display: block;
    padding: 2px 8px;
    border-radius: 4px;
}
.diff-line-del {
    color: #D68C8C !important;
    background: rgba(214, 140, 140, 0.10);
    display: block;
    padding: 2px 8px;
    border-radius: 4px;
}
.diff-line-hdr {
    color: #8E9094 !important;
    display: block;
    padding: 2px 8px;
    opacity: 0.85;
}
.diff-line-ctx {
    color: #C7C8CA !important;
    display: block;
    padding: 2px 8px;
}

/* ---------- Why-changed Banner — neutral, not a competing accent color ---------- */
.why-changed {
    background: #1A1C1F !important;
    border: 1px solid #2A2C2F !important;
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #D4D5D7 !important;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}
.why-changed strong {
    white-space: nowrap;
    color: #F5F5F4 !important;
}

/* ---------- Status Pill ---------- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1A1C1F !important;
    border: 1px solid #2A2C2F !important;
    color: #C7C8CA !important;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ---------- Chat Bubbles ---------- */
.chat-user {
    background: #1A1C1F !important;
    border: 1px solid #2A2C2F !important;
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.75rem;
    max-width: 85%;
    margin-left: auto;
    font-size: 0.9rem;
    color: #F5F5F4 !important;
}
.chat-assistant {
    background: #16181B !important;
    border: 1px solid #26282B !important;
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.75rem;
    max-width: 85%;
    font-size: 0.9rem;
    color: #F5F5F4 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG_PATH = Path("logs/conversation_example.md")
LOG_PATH.parent.mkdir(exist_ok=True)


def _render_citations(text: str) -> str:
    """Replace [Cx] tags with styled HTML badges."""
    def _badge(m):
        return f'<span class="citation-badge">{m.group(0)}</span>'
    return re.sub(r"\[C\d+\]", _badge, text)


def _render_diff_html(delta_lines: list[str]) -> str:
    """Render unified-diff lines as styled HTML with colour highlighting."""
    if not delta_lines:
        return '<div class="diff-container"><span class="diff-line-ctx">No changes detected.</span></div>'

    html_lines: list[str] = []
    for line in delta_lines:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("+++") or line.startswith("---"):
            html_lines.append(f'<span class="diff-line-hdr">{escaped}</span>')
        elif line.startswith("@@"):
            html_lines.append(f'<span class="diff-line-hdr">{escaped}</span>')
        elif line.startswith("+"):
            html_lines.append(f'<span class="diff-line-add">{escaped}</span>')
        elif line.startswith("-"):
            html_lines.append(f'<span class="diff-line-del">{escaped}</span>')
        else:
            html_lines.append(f'<span class="diff-line-ctx">{escaped}</span>')

    return f'<div class="diff-container">{"".join(html_lines)}</div>'


def _render_section_card(icon: str, title: str, content: str, badge: str = "Generated") -> None:
    """Render a section in a styled card with citation badges."""
    if not content:
        return
    rendered = _render_citations(content.replace("\n", "<br>"))
    st.markdown(
        f'<div class="output-card">'
        f'<span class="card-badge">{badge}</span>'
        f'<h3>{icon} {title}</h3>'
        f'<div>{rendered}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def log_turn(
    kind: str, prompt: str, output: str,
    diff_text: str | None = None,
    why_changed: str | None = None,
) -> None:
    """Append a turn to the conversation log and write to disk."""
    st.session_state.history.append({
        "kind": kind,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "output": output,
        "diff": diff_text,
        "why_changed": why_changed,
    })
    _write_log()


def _write_log() -> None:
    """Persist the full conversation history to a markdown file."""
    lines = ["# Conversation log\n"]
    for turn in st.session_state.history:
        lines.append(f"## {turn['kind']} — {turn['timestamp']}\n")
        lines.append(f"**Prompt:** {turn['prompt']}\n")
        lines.append(f"**Output:**\n\n```\n{turn['output']}\n```\n")
        if turn.get("diff"):
            lines.append(f"**Diff:**\n\n```diff\n{turn['diff']}\n```\n")
        if turn.get("why_changed"):
            lines.append(f"**Why changed:** {turn['why_changed']}\n")
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None
if "current_output" not in st.session_state:
    st.session_state.current_output = ""
if "structured" not in st.session_state:
    st.session_state.structured = None
if "retrieved" not in st.session_state:
    st.session_state.retrieved = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------------------------------------------------
# Header — simple "Saral AI" wordmark, matching real branding
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="saral-header">'
    '<span class="saral-wordmark">Saral AI</span>'
    '<div class="saral-header-subtitle">Audience-Adaptive Script & Bullet Generator</div>'
    '<p class="saral-header-desc">Upload a research paper → get audience-adapted scripts, slide bullets, '
    'speaker notes, and tweet-sized abstracts — with full provenance.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: PDF upload + generation params
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Paper Upload")
    uploaded_pdf = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        help="The paper will be chunked, embedded, and indexed for retrieval.",
    )

    if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.pdf_name:
        with st.spinner(f"Indexing **{uploaded_pdf.name}**..."):
            # Save uploaded file to a temp location and ingest
            stem = Path(uploaded_pdf.name).stem
            index_dir = f"data/index_{stem}"
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            try:
                build_index(tmp_path, index_dir)
                st.session_state.index_dir = index_dir
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.current_output = ""
                st.session_state.structured = None
                st.success(f"Indexed **{uploaded_pdf.name}**")
            except (FileNotFoundError, ValueError) as e:
                st.error(f"Indexing failed: {e}")

    # Check if a default index exists (for demo without upload)
    if st.session_state.index_dir is None and os.path.isfile("data/index/index.faiss"):
        st.session_state.index_dir = "data/index"
        st.info("Using pre-built index at `data/index/`")

    st.divider()
    st.markdown("### Generation Settings")
    audience = st.selectbox(
        "Audience",
        ["grad students", "general public", "policymakers", "technical experts"],
    )
    length = st.selectbox("Length", ["30s", "90s", "5min"])
    style = st.selectbox("Style", ["technical", "plain language", "press release"])

# ---------------------------------------------------------------------------
# Main content: Generate
# ---------------------------------------------------------------------------
if st.session_state.index_dir is None:
    st.info("Upload a PDF in the sidebar to get started, or place a pre-built index at `data/index/`.")
    st.stop()

st.markdown("### Generate")
task = st.text_area(
    "What would you like to generate?",
    placeholder="e.g. Make a 7-slide talk for grad students focusing on methods; include 3 speaker notes per slide and preserve key equations",
    height=100,
)

if st.button("Generate", type="primary", use_container_width=True):
    if not task.strip():
        st.warning("Please enter a task / prompt first.")
    else:
        with st.spinner("Retrieving relevant passages and generating..."):
            try:
                structured, retrieved = generate_structured(
                    st.session_state.index_dir, audience, length, style, task,
                )
                st.session_state.current_output = structured.raw
                st.session_state.structured = structured
                st.session_state.retrieved = retrieved
                log_turn("initial generation", task, structured.raw)
            except RuntimeError as e:
                st.error(f"Generation failed: {e}")

# ---------------------------------------------------------------------------
# Display output
# ---------------------------------------------------------------------------
if st.session_state.structured is not None:
    structured: StructuredOutput = st.session_state.structured
    retrieved = st.session_state.retrieved

    # Status pill
    cov = citation_coverage(structured.raw)
    st.markdown(
        f'<div class="status-pill">Generated · {len(retrieved)} chunks retrieved '
        f'· {cov}% citation coverage</div>',
        unsafe_allow_html=True,
    )

    # Tweet abstract (special card)
    if structured.tweet_abstract:
        char_count = len(structured.tweet_abstract)
        st.markdown(
            f'<div class="tweet-card">'
            f'{structured.tweet_abstract}'
            f'<div class="tweet-char-count">{char_count}/280 chars</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Section cards
    col1, col2 = st.columns(2)
    with col1:
        _render_section_card("", "Slide Bullets", structured.slide_bullets)
    with col2:
        _render_section_card("", "Speaker Notes", structured.speaker_notes)

    _render_section_card("", "Speaker Script", structured.speaker_script)

    # If the model didn't produce structured sections, show raw fallback
    if not structured.speaker_script and not structured.slide_bullets:
        _render_section_card("", "Generated Output", structured.raw)

    # Provenance details (collapsed)
    with st.expander("Provenance — Retrieved Source Chunks"):
        for chunk in retrieved:
            st.markdown(
                f'<div class="output-card" style="padding:0.75rem 1rem;">'
                f'<strong><span class="citation-badge">[{chunk["chunk_id"]}]</span> '
                f'page {chunk["page"]}</strong> '
                f'<small style="color:#8E9094;">(score: {chunk["score"]:.3f})</small>'
                f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#C7C8CA;">'
                f'{chunk["text"][:300]}{"..." if len(chunk["text"]) > 300 else ""}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ----- Iterative editing -----
    st.markdown("---")
    st.markdown("### Iterative Edit")

    change_instruction = st.text_input(
        "Change instruction",
        placeholder='e.g. "make #2 less technical", "dumb down the speaker notes", "add more equations"',
    )

    if st.button("Apply Change", use_container_width=True):
        if not change_instruction.strip():
            st.warning("Please enter a change instruction.")
        else:
            with st.spinner("Applying edit and computing diff..."):
                try:
                    old_output = st.session_state.current_output
                    new_output, delta_lines, why_changed = apply_change(
                        st.session_state.index_dir, old_output, change_instruction,
                    )
                    # Update state
                    new_structured = StructuredOutput.from_raw(new_output)
                    st.session_state.current_output = new_output
                    st.session_state.structured = new_structured

                    diff_text = "\n".join(delta_lines)
                    log_turn("edit", change_instruction, new_output, diff_text, why_changed)

                    # Show why-changed banner
                    st.markdown(
                        f'<div class="why-changed">'
                        f'<strong>Why changed:</strong> {why_changed}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Show styled diff
                    st.markdown("**Changes:**")
                    st.markdown(_render_diff_html(delta_lines), unsafe_allow_html=True)

                    st.rerun()
                except RuntimeError as e:
                    st.error(f"Edit failed: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(f"Conversation log auto-saved to `{LOG_PATH}` · Built for the SARAL recruitment assignment")