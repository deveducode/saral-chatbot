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
import os
import re
import uuid

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 420
CHUNK_OVERLAP = 60

MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),                     # $$ ... $$
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),                      # \[ ... \]
    re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL),  # $ ... $
]


def extract_pages(pdf_path):
    """Return list of (page_number, text) tuples, 1-indexed pages."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages


def protect_math(text):
    """Replace math blocks with placeholders so they never get split.
    Returns (protected_text, placeholder_map)."""
    placeholder_map = {}

    def _replace(match):
        key = f"__MATH_{uuid.uuid4().hex[:8]}__"
        placeholder_map[key] = match.group(0)
        return key

    for pattern in MATH_PATTERNS:
        text = pattern.sub(_replace, text)
    return text, placeholder_map


def restore_math(text, placeholder_map):
    for key, original in placeholder_map.items():
        text = text.replace(key, original)
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker on the (math-protected) text."""
    protected, placeholder_map = protect_math(text)
    chunks = []
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


def build_index(pdf_path, out_dir, model_name="all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Extracting text from {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"      -> {len(pages)} pages with text")

    print("[2/4] Chunking (preserving LaTeX math blocks) ...")
    records = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            records.append({
                "chunk_id": f"C{len(records)+1}",
                "page": page_num,
                "text": chunk,
            })
    print(f"      -> {len(records)} chunks")

    print(f"[3/4] Embedding with {model_name} ...")
    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("[4/4] Building FAISS index ...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim since embeddings are normalized
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "source_pdf": os.path.basename(pdf_path),
            "embedding_model": model_name,
            "records": records,
        }, f, indent=2)

    print(f"Done. Index + metadata written to {out_dir}/")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="data/index", help="Output dir for index+metadata")
    args = parser.parse_args()
    build_index(args.pdf, args.out)
