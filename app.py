"""
SARAL prototype — Streamlit UI
Wires together retrieve.py + generate.py into a chat-style interface with
iterative "change instruction" editing (delta engine) and conversation logging.

Expected interface from your existing scripts (adjust the two import calls
below if your actual function names differ):

    retrieve.py:
        def retrieve(query: str, k: int = 5) -> list[dict]
            # each dict: {"chunk_id": "C13", "text": ..., "page": ...}

    generate.py:
        def generate(chunks: list[dict], audience: str, length: str,
                     style: str, instruction: str) -> str
            # instruction = the user's task ("make a 90s script...") on first
            # call, or the change instruction ("make #2 less technical") on edits

Run with:  streamlit run app.py
"""

import json
import difflib
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from retrieve import retrieve
    from generate import apply_change, generate
except ImportError:
    # Fallback stubs so the UI is demoable even before wiring is finished.
    def retrieve(query, k=5):
        return [{"chunk_id": f"C{i}", "text": f"(stub chunk {i} for: {query})", "page": i}
                 for i in range(1, k + 1)]

    def generate(chunks, audience, length, style, instruction):
        cite = " ".join(f"[{c['chunk_id']}]" for c in chunks[:3])
        return f"(stub output for '{instruction}', audience={audience}, " \
               f"length={length}, style={style}) {cite}"

LOG_PATH = Path("logs/conversation_example.md")
LOG_PATH.parent.mkdir(exist_ok=True)

st.set_page_config(page_title="SARAL — RAG Script Generator", layout="wide")
st.title("SARAL — RAG Script Generator (prototype)")

# ---- session state ----
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "current_output" not in st.session_state:
    st.session_state.current_output = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts for the log file


def log_turn(kind, prompt, output, diff_text=None, why_changed=None):
    st.session_state.history.append({
        "kind": kind,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "output": output,
        "diff": diff_text,
        "why_changed": why_changed,
    })
    write_log()


def write_log():
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


# ---- sidebar: generation params ----
with st.sidebar:
    st.header("Generation settings")
    audience = st.selectbox("Audience", ["general public", "grad students", "policymakers", "technical experts"])
    length = st.selectbox("Length", ["30s", "90s", "5min"])
    style = st.selectbox("Style", ["plain language", "technical", "press release"])
    source_query = st.text_input("Source query (what to retrieve)", "landslide susceptibility assessment")

st.subheader("1. Initial generation")
task = st.text_area("Task / prompt", "Write a 90-second script summarizing this section.")

if st.button("Generate"):
    st.session_state.chunks = retrieve(source_query)
    st.session_state.current_output, st.session_state.chunks = generate(
        "data/index", audience, length, style, task
    )
    log_turn("initial generation", task, st.session_state.current_output)

if st.session_state.current_output:
    st.markdown("**Current output:**")
    st.markdown(st.session_state.current_output)

    st.subheader("2. Iterative edit")
    change_instruction = st.text_input("Change instruction", "make it less technical")
    if st.button("Apply change"):
        old_output = st.session_state.current_output
        new_output, delta_lines, why_changed = apply_change(
            "data/index", old_output, change_instruction
        )
        diff_text = "\n".join(delta_lines)
        st.session_state.current_output = new_output
        log_turn("edit", change_instruction, new_output, diff_text, why_changed)

        st.markdown("**Updated output:**")
        st.markdown(new_output)
        st.markdown("**Diff:**")
        st.code(diff_text, language="diff")
        st.markdown(f"**Why changed:** {why_changed}")

st.divider()
st.caption(f"Conversation log auto-saved to `{LOG_PATH}`")