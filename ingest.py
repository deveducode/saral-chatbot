r"""
ingest.py
Pipeline stage 1: Ingest a paper (PDF) -> extract text per page -> chunk
(preserving inline/block LaTeX math) -> embed with sentence-transformers ->
store in a FAISS index + a parallel metadata JSON (chunk text, page number,
chunk id) so we can return provenance later.

Usage:
    python ingest.py --pdf data/sample_paper.pdf --out data/index

Design notes (for README / architecture writeup):
- We keep chunk size ~350-450 characters with ~50 char overlap. Small papers
  don't need huge chunks; smaller chunks -> more precise provenance.
- Math protection: before splitting, we find $...$, $$...$$, and
  \begin{equation}...\end{equation} blocks and replace them with placeholder
  tokens so the splitter never cuts through the middle of an equation. After
  splitting, placeholders are restored.
- We store page number per chunk (from PyMuPDF) for provenance/page citation.
"""

import argparse
import json
import logging
import os
import re
import uuid
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 420
CHUNK_OVERLAP = 60

# Patterns used to locate LaTeX math blocks so they can be protected during
# chunking (and also reused by retrieve.py for math-aware scoring — import
# from here to avoid duplication).
MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),                     # $$ ... $$
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),                      # \[ ... \]
    re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL),  # $ ... $
]

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples, 1-indexed pages.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If the PDF contains zero extractable text pages.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    doc.close()

    if not pages:
        raise ValueError(
            f"PDF has no extractable text pages: {pdf_path}. "
            "The file may be image-only (scanned) or corrupted."
        )
    return pages


# ---------------------------------------------------------------------------
# Math-safe chunking
# ---------------------------------------------------------------------------


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    """Replace math blocks with placeholders so they never get split.

    Returns:
        A tuple of (protected_text, placeholder_map) where placeholder_map
        maps placeholder tokens back to the original math expression.
    """
    placeholder_map: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        key = f"__MATH_{uuid.uuid4().hex[:8]}__"
        placeholder_map[key] = match.group(0)
        return key

    for pattern in MATH_PATTERNS:
        text = pattern.sub(_replace, text)
    return text, placeholder_map


def restore_math(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore placeholder tokens back to their original math expressions."""
    for key, original in placeholder_map.items():
        text = text.replace(key, original)
    return text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Sliding-window chunker that preserves LaTeX math blocks intact.

    The text is first *protected* (math expressions replaced with fixed-length
    placeholder tokens) so the character-based window never slices through the
    middle of an equation.  After splitting, the original math is restored.
    """
    if not text or not text.strip():
        return []

    protected, placeholder_map = protect_math(text)
    chunks: list[str] = []
    start = 0
    n = len(protected)
    while start < n:
        end = min(start + chunk_size, n)
        piece = protected[start:end]
        chunks.append(restore_math(piece, placeholder_map))
        if end == n:
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_index(
    pdf_path: str,
    out_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Ingest a PDF and build a FAISS index with chunk metadata.

    Args:
        pdf_path: Path to the source PDF.
        out_dir: Directory to write ``index.faiss`` and ``metadata.json``.
        model_name: Sentence-transformer model used for embedding.

    Returns:
        A list of chunk record dicts (chunk_id, page, text).

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If the PDF yields zero text chunks.
    """
    os.makedirs(out_dir, exist_ok=True)

    logger.info("[1/4] Extracting text from %s ...", pdf_path)
    pages = extract_pages(pdf_path)  # raises on missing/empty PDF
    logger.info("      -> %d pages with text", len(pages))

    logger.info("[2/4] Chunking (preserving LaTeX math blocks) ...")
    records: list[dict] = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            records.append({
                "chunk_id": f"C{len(records)+1}",
                "page": page_num,
                "text": chunk,
            })

    if not records:
        raise ValueError(
            f"Chunking produced zero chunks from {pdf_path}. "
            "The PDF may contain only images or non-textual content."
        )
    logger.info("      -> %d chunks", len(records))

    logger.info("[3/4] Embedding with %s ...", model_name)
    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    logger.info("[4/4] Building FAISS index ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim (embeddings are L2-normalized)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "source_pdf": os.path.basename(pdf_path),
            "embedding_model": model_name,
            "records": records,
        }, f, indent=2)

    logger.info("Done. Index + metadata written to %s/", out_dir)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="data/index", help="Output dir for index+metadata")
    args = parser.parse_args()
    build_index(args.pdf, args.out)
