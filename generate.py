"""
generate.py
Pipeline stage 3: Given a user instruction (audience/length/style + task),
retrieve relevant chunks, build a parameterized prompt, and call Gemini to
produce the script/bullets. Also implements:
  - provenance tagging ([Cx] chunk-id citations, mapped to page numbers)
  - iterative change-tracking: re-generate a targeted section on user edit
    requests ("make #2 less technical") and show an old-vs-new delta

Usage (standalone CLI demo):
    python generate.py --index data/index --audience "grad students" \
        --length 5min --style technical \
        --instruction "Make a 7-slide talk focusing on methods; include 3 speaker notes per slide"

Requires GEMINI_API_KEY in a .env file (see .env.example).
"""

import argparse
import difflib
import os

import google.generativeai as genai
from dotenv import load_dotenv

from retrieve import Retriever, format_context, instruction_wants_math

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.6-flash"  # fast + cheap, good enough for this task

SYSTEM_TEMPLATE = """You are SARAL's script & bullet generator. You turn research paper
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
7. Output ONLY the requested content (script/bullets) - no meta-commentary about
   what you are doing.

SOURCE CHUNKS:
{context}

TASK:
{instruction}
"""

CHANGE_TEMPLATE = """You are SARAL's iterative editor. Below is a PREVIOUS OUTPUT you
generated, and a CHANGE INSTRUCTION from the user asking you to revise part of it.

RULES:
1. Apply the change instruction precisely - only touch what the user asked you to change,
   keep everything else as close to the original as reasonably possible.
2. Keep citation tags ([Cx]) on any claim that still comes from a source chunk. If you
   add or change a claim, cite it from the SOURCE CHUNKS below (do not invent new claims
   without a chunk backing them - if you can't find support, simplify/remove the claim
   instead).
3. Preserve LaTeX/math notation exactly.
4. Output ONLY the revised content - no meta-commentary.

SOURCE CHUNKS:
{context}

PREVIOUS OUTPUT:
{previous_output}

CHANGE INSTRUCTION:
{change_instruction}
"""


def _get_model():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=<your key>."
        )
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def build_prompt(audience, length, style, instruction, context):
    return SYSTEM_TEMPLATE.format(
        audience=audience, length=length, style=style,
        context=context, instruction=instruction,
    )


def generate(index_dir, audience, length, style, instruction, k=6):
    """Full pipeline: retrieve -> prompt -> generate. Returns (output_text, retrieved_chunks).

    Automatically enables math-aware retrieval boosting when the instruction
    implies equations/math should be preserved (see retrieve.py Retriever.search).
    """
    retriever = Retriever(index_dir)
    boost_math = instruction_wants_math(instruction)
    retrieved = retriever.search(instruction, k=k, boost_math=boost_math)
    context = format_context(retrieved)

    prompt = build_prompt(audience, length, style, instruction, context)
    model = _get_model()
    response = model.generate_content(prompt)
    return response.text, retrieved


def apply_change(index_dir, previous_output, change_instruction, k=6):
    """Re-generate based on an edit instruction, return (new_output, delta_lines, why_changed)."""
    retriever = Retriever(index_dir)
    retrieved = retriever.search(change_instruction, k=k)
    context = format_context(retrieved)

    prompt = CHANGE_TEMPLATE.format(
        context=context, previous_output=previous_output,
        change_instruction=change_instruction,
    )
    model = _get_model()
    response = model.generate_content(prompt)
    new_output = response.text

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
    why_response = model.generate_content(why_prompt)
    why_changed = why_response.text.strip()

    return new_output, delta, why_changed


def citation_coverage(text):
    """% of non-empty lines that contain at least one [Cx] citation tag."""
    import re
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    cited = sum(1 for l in lines if re.search(r"\[C\d+\]", l))
    return round(100 * cited / len(lines), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--audience", default="grad students")
    parser.add_argument("--length", default="90s")
    parser.add_argument("--style", default="technical")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    output, retrieved = generate(
        args.index, args.audience, args.length, args.style, args.instruction, k=args.k
    )
    print("=" * 60)
    print("GENERATED OUTPUT")
    print("=" * 60)
    print(output)
    print("=" * 60)
    print(f"Citation coverage: {citation_coverage(output)}%")
    print(f"Retrieved {len(retrieved)} chunks: {[r['chunk_id'] for r in retrieved]}")