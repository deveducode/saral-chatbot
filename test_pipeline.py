"""
test_pipeline.py
Lightweight unit tests for the pure-logic parts of the pipeline (chunking's
math-protection, citation-tag extraction/coverage, structured output parsing).

These deliberately avoid loading the embedding model or calling the LLM,
so they run in under a second and can be used as a quick sanity check
after any change.

Usage:
    pip install pytest
    pytest test_pipeline.py -v
"""

import re

from ingest import protect_math, restore_math, chunk_text
from generate import citation_coverage, StructuredOutput, _split_sections
from eval import extract_cited_sentences


# ---------- ingest.py: math protection ----------

def test_protect_and_restore_math_roundtrip():
    text = "The result is $E=mc^2$ and also $$F = ma$$ in this paper."
    protected, placeholder_map = protect_math(text)
    # no raw math should remain in the protected text
    assert "$E=mc^2$" not in protected
    assert "$$F = ma$$" not in protected
    restored = restore_math(protected, placeholder_map)
    assert restored == text


def test_chunking_never_splits_inline_math():
    # construct a long text where math sits right at a chunk boundary
    filler = "word " * 80
    text = filler + "The formula $Pr = \\frac{TP}{TP+FP}$ is important." + filler
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    full_math = "$Pr = \\frac{TP}{TP+FP}$"
    # the full equation must appear intact in at least one chunk
    assert any(full_math in c for c in chunks), (
        "Math expression was split across chunk boundaries"
    )


def test_chunking_never_splits_equation_block():
    filler = "word " * 80
    text = filler + "\\begin{equation}F1 = 2PR/(P+R)\\end{equation}" + filler
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    full_eq = "\\begin{equation}F1 = 2PR/(P+R)\\end{equation}"
    assert any(full_eq in c for c in chunks)


def test_chunking_produces_nonempty_chunks():
    text = "Sentence one. " * 50
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunking_empty_text_returns_empty():
    """Edge case: empty or whitespace-only text should return no chunks."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


# ---------- generate.py: citation coverage ----------

def test_citation_coverage_all_lines_cited():
    text = "Claim one is true [C1].\nClaim two is also true [C2][C3]."
    assert citation_coverage(text) == 100.0


def test_citation_coverage_partial():
    text = "### Header with no citation\nClaim with a citation [C5]."
    cov = citation_coverage(text)
    assert 0 < cov < 100


def test_citation_coverage_empty_text():
    assert citation_coverage("") == 0.0


def test_citation_coverage_no_citations():
    """Text with no [Cx] tags at all should return 0.0."""
    text = "This is a claim.\nAnother claim here."
    assert citation_coverage(text) == 0.0


# ---------- generate.py: structured output parsing ----------

def test_structured_output_parses_sections():
    """StructuredOutput.from_raw correctly splits ## sections."""
    raw = (
        "## Slide Bullets\n"
        "- Point A [C1]\n- Point B [C2]\n\n"
        "## Speaker Script\n"
        "This is the full spoken script [C1].\n\n"
        "## Speaker Notes\n"
        "Note: emphasize point A.\n\n"
        "## Tweet Abstract\n"
        "New study shows X via method Y — key takeaway here."
    )
    s = StructuredOutput.from_raw(raw)
    assert "Point A" in s.slide_bullets
    assert "Point B" in s.slide_bullets
    assert "full spoken script" in s.speaker_script
    assert "emphasize" in s.speaker_notes
    assert len(s.tweet_abstract) <= 280


def test_structured_output_fallback_no_headers():
    """When the LLM output has no ## headers, everything goes into speaker_script."""
    raw = "This is a plain text output without any section headers [C1]."
    s = StructuredOutput.from_raw(raw)
    assert s.speaker_script == raw.strip()
    assert s.slide_bullets == ""
    assert s.speaker_notes == ""


def test_split_sections_empty():
    """Empty text should return a speaker_script fallback."""
    sections = _split_sections("")
    assert "speaker script" in sections


# ---------- eval.py: sentence/citation extraction ----------

def test_extract_cited_sentences_finds_chunk_ids():
    text = "The model performs well [C12]. It also generalizes [C7][C8]."
    pairs = extract_cited_sentences(text)
    assert len(pairs) == 2
    assert pairs[0][1] == ["C12"]
    assert pairs[1][1] == ["C7", "C8"]


def test_extract_cited_sentences_strips_tags_from_text():
    text = "A clean claim [C1]."
    pairs = extract_cited_sentences(text)
    sentence, chunk_ids = pairs[0]
    assert "[C1]" not in sentence
    assert chunk_ids == ["C1"]


def test_extract_cited_sentences_ignores_uncited_text():
    text = "This sentence has no citation at all."
    pairs = extract_cited_sentences(text)
    assert pairs == []


def test_extract_cited_sentences_skips_citation_only():
    """A 'sentence' that is only citation tags with no text should be skipped."""
    text = "[C1][C2]."
    pairs = extract_cited_sentences(text)
    assert pairs == []


if __name__ == "__main__":
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
