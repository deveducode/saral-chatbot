"""
generate.py
Pipeline stage 3: Given a user instruction (audience/length/style + task),
retrieve relevant chunks, build a parameterized prompt, and call Gemini to
produce **structured output** (slide bullets, speaker script, speaker notes,
and a tweet-sized abstract) — all with inline provenance citations.

Also implements:
  - Iterative change-tracking: re-generate a targeted section on user edit
    requests ("make #2 less technical") and show an old-vs-new delta
  - Tweet-sized abstract generation (≤280 chars)

Usage (standalone CLI demo):
    python generate.py --index data/index --audience "grad students" \\
        --length 5min --style technical \\
        --instruction "Make a 7-slide talk focusing on methods; include 3 speaker notes per slide"

Requires GEMINI_API_KEY in a .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from retrieve import Retriever, format_context, instruction_wants_math

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY: str | None = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.6-flash"  # fast + cheap, good enough for this task

# Maximum retries on transient Gemini API errors (rate-limit, network, etc.)
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECS = 2.0

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are SARAL's script & bullet generator. You turn research paper
content into audience-adapted scripts, slide bullets, and short summaries.

PARAMETERS:
- Audience: {audience}
- Length: {length}
- Style: {style}

RULES:
1. Use ONLY the information present in the SOURCE CHUNKS below. Do not invent facts,
   numbers, or citations not present in the source.
2. Preserve any LaTeX/math notation exactly as written (e.g. keep $E=mc^2$ inline).
3. After every factual sentence or claim, add a citation tag referencing the chunk(s)
   it came from, like this: [C12]. If a sentence draws on multiple chunks, cite all,
   e.g. [C12][C15].
4. Match the requested audience and style:
   - policymakers: plain language, focus on implications/impact, avoid jargon
   - grad students: technical depth is fine, explain methods clearly
   - press: short, punchy, avoid jargon, lead with the "so what"
   - technical style: keep precise terminology and equations
   - plain-English style: simplify terminology, explain acronyms
   - press release style: headline-first, quotable, accessible
5. Respect the requested length as an approximate word budget:
   - 30s script: ~75 words
   - 90s script: ~225 words
   - 5min script: ~750 words
6. Do not include offensive, biased, or inaccessible language.
7. Structure your output with these EXACT section headers (Markdown ##):

   ## Slide Bullets
   (Concise, slide-ready bullet points — one line per bullet, grouped by slide/topic)

   ## Speaker Script
   (The full spoken script at the requested length)

   ## Speaker Notes
   (3 brief notes per slide/section for the presenter — tips, emphasis cues, timing)

   ## Tweet Abstract
   (A single tweet-length summary of the paper, ≤280 characters, no citations needed)

SOURCE CHUNKS:
{context}

TASK:
{instruction}
"""

CHANGE_TEMPLATE = """\
You are SARAL's iterative editor. Below is a PREVIOUS OUTPUT you
generated, and a CHANGE INSTRUCTION from the user asking you to revise part of it.

RULES:
1. Apply the change instruction precisely - only touch what the user asked you to change,
   keep everything else as close to the original as reasonably possible.
2. Keep citation tags ([Cx]) on any claim that still comes from a source chunk. If you
   add or change a claim, cite it from the SOURCE CHUNKS below (do not invent new claims
   without a chunk backing them - if you can't find support, simplify/remove the claim
   instead).
3. Preserve LaTeX/math notation exactly.
4. Keep the same section structure (## Slide Bullets, ## Speaker Script,
   ## Speaker Notes, ## Tweet Abstract).
5. Output ONLY the revised content - no meta-commentary.

SOURCE CHUNKS:
{context}

PREVIOUS OUTPUT:
{previous_output}

CHANGE INSTRUCTION:
{change_instruction}
"""


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------

@dataclass
class StructuredOutput:
    """Parsed sections from the LLM's structured response."""
    raw: str
    slide_bullets: str = ""
    speaker_script: str = ""
    speaker_notes: str = ""
    tweet_abstract: str = ""

    @classmethod
    def from_raw(cls, text: str) -> "StructuredOutput":
        """Parse the LLM output into labeled sections.

        Falls back gracefully: if the model doesn't produce the expected
        section headers, the entire text is placed in ``speaker_script``
        (the most important section) so nothing is lost.
        """
        sections = _split_sections(text)
        return cls(
            raw=text,
            slide_bullets=sections.get("slide bullets", ""),
            speaker_script=sections.get("speaker script", ""),
            speaker_notes=sections.get("speaker notes", ""),
            tweet_abstract=sections.get("tweet abstract", ""),
        )


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text on ``## <Header>`` lines into {header: body}."""
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    headers = list(pattern.finditer(text))

    if not headers:
        # Fallback: model didn't use section headers — treat everything as script
        return {"speaker script": text.strip()}

    sections: dict[str, str] = {}
    for i, match in enumerate(headers):
        key = match.group(1).strip().lower()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections[key] = text[start:end].strip()
    return sections


# ---------------------------------------------------------------------------
# Citation helpers (also used by eval.py)
# ---------------------------------------------------------------------------

def citation_coverage(text: str) -> float:
    """Percentage of non-empty lines that contain at least one ``[Cx]`` citation tag."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    cited = sum(1 for line in lines if re.search(r"\[C\d+\]", line))
    return round(100 * cited / len(lines), 1)


# ---------------------------------------------------------------------------
# Gemini API wrapper with retry
# ---------------------------------------------------------------------------

def _get_model() -> genai.GenerativeModel:
    """Configure and return the Gemini model.

    Raises:
        RuntimeError: If ``GEMINI_API_KEY`` is not set.
    """
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=<your key>. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def _call_gemini(prompt: str) -> str:
    """Call Gemini with automatic retry on transient failures.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff for
    rate-limit (429), server errors (5xx), and network timeouts.

    Raises:
        RuntimeError: If all retries are exhausted or a non-retryable error occurs.
    """
    model = _get_model()
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 2):  # +2 because range is exclusive + initial attempt
        try:
            response = model.generate_content(prompt)

            # Handle content-filtered / empty responses
            if not response.parts:
                raise RuntimeError(
                    "Gemini returned an empty response (content may have been "
                    "filtered by safety settings). Try rephrasing the instruction."
                )
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Determine if this is a retryable error
            is_retryable = any(
                keyword in error_str
                for keyword in ("429", "rate limit", "quota", "500", "503",
                                "timeout", "deadline", "unavailable")
            )

            if not is_retryable or attempt > _MAX_RETRIES:
                raise RuntimeError(
                    f"Gemini API error after {attempt} attempt(s): {e}"
                ) from e

            wait = _RETRY_BACKOFF_SECS * (2 ** (attempt - 1))
            logger.warning(
                "Gemini API error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, _MAX_RETRIES + 1, wait, e,
            )
            time.sleep(wait)

    # Should never reach here, but just in case
    raise RuntimeError(f"Gemini API failed after all retries: {last_error}")


# ---------------------------------------------------------------------------
# Public API: generate + apply_change
# ---------------------------------------------------------------------------

def build_prompt(
    audience: str, length: str, style: str,
    instruction: str, context: str,
) -> str:
    """Assemble the full generation prompt from parameters and context."""
    return SYSTEM_TEMPLATE.format(
        audience=audience, length=length, style=style,
        context=context, instruction=instruction,
    )


def generate(
    index_dir: str,
    audience: str,
    length: str,
    style: str,
    instruction: str,
    k: int = 6,
) -> tuple[str, list[dict]]:
    """Full pipeline: retrieve -> prompt -> generate.

    Returns:
        A tuple of (output_text, retrieved_chunks).  The output_text contains
        structured sections (bullets, script, notes, tweet abstract) separated
        by ``## <Header>`` lines.

    Automatically enables math-aware retrieval boosting when the instruction
    implies equations/math should be preserved (see ``retrieve.py``).
    """
    retriever = Retriever(index_dir)
    boost_math = instruction_wants_math(instruction)
    retrieved = retriever.search(instruction, k=k, boost_math=boost_math)
    # Convert tuple (from LRU cache) to list for external consumers
    retrieved = list(retrieved)
    context = format_context(retrieved)

    prompt = build_prompt(audience, length, style, instruction, context)
    output_text = _call_gemini(prompt)

    logger.info("Retrieval cache stats: %s", retriever.cache_stats())
    return output_text, retrieved


def generate_structured(
    index_dir: str,
    audience: str,
    length: str,
    style: str,
    instruction: str,
    k: int = 6,
) -> tuple[StructuredOutput, list[dict]]:
    """Like :func:`generate` but returns a parsed :class:`StructuredOutput`."""
    raw_text, retrieved = generate(index_dir, audience, length, style, instruction, k)
    return StructuredOutput.from_raw(raw_text), retrieved


def apply_change(
    index_dir: str,
    previous_output: str,
    change_instruction: str,
    k: int = 6,
) -> tuple[str, list[str], str]:
    """Re-generate based on an edit instruction.

    Returns:
        A tuple of (new_output, delta_lines, why_changed).
    """
    retriever = Retriever(index_dir)
    retrieved = retriever.search(change_instruction, k=k)
    context = format_context(retrieved)

    prompt = CHANGE_TEMPLATE.format(
        context=context, previous_output=previous_output,
        change_instruction=change_instruction,
    )
    new_output = _call_gemini(prompt)

    # Compute unified diff
    delta = list(difflib.unified_diff(
        previous_output.splitlines(),
        new_output.splitlines(),
        lineterm="",
        fromfile="before",
        tofile="after",
    ))

    # Ask the model for a one-line "why changed" explanation
    why_prompt = (
        f"In one short sentence, explain what changed between this BEFORE and AFTER "
        f"text and why (based on the instruction: '{change_instruction}').\n\n"
        f"BEFORE:\n{previous_output}\n\nAFTER:\n{new_output}"
    )
    try:
        why_changed = _call_gemini(why_prompt).strip()
    except RuntimeError:
        why_changed = "(Could not generate explanation — see diff above.)"

    return new_output, delta, why_changed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--audience", default="grad students")
    parser.add_argument("--length", default="90s")
    parser.add_argument("--style", default="technical")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    structured, retrieved = generate_structured(
        args.index, args.audience, args.length, args.style, args.instruction, k=args.k,
    )
    print("=" * 60)
    print("GENERATED OUTPUT")
    print("=" * 60)
    print(structured.raw)
    print("=" * 60)
    print(f"Citation coverage: {citation_coverage(structured.raw)}%")
    print(f"Retrieved {len(retrieved)} chunks: {[r['chunk_id'] for r in retrieved]}")

    if structured.tweet_abstract:
        print(f"\nTweet abstract ({len(structured.tweet_abstract)} chars):")
        print(f"  {structured.tweet_abstract}")