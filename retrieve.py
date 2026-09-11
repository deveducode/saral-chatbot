"""
retrieve.py
Pipeline stage 2: given a natural-language query, embed it with the same
model used at ingest time, search the FAISS index, and return the top-k
chunks along with their provenance (chunk_id, page number, similarity score).

This is imported by generate.py — but you can also run it standalone to
sanity-check retrieval quality before wiring up the LLM:

    python retrieve.py --index data/index --query "what does the RAG layer do?" --k 4
"""

import argparse
import json
import os
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}

# Same math-detection patterns as ingest.py's protect_math, used here to
# identify chunks that contain LaTeX/math content so they can be boosted
# in retrieval when the user's request implies preserving equations.
_MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL),
]

# If the user's instruction contains any of these, we treat it as a signal
# that math-preserving retrieval should be favored.
_MATH_INTENT_KEYWORDS = (
    "equation", "equations", "formula", "formulas", "math", "mathematical",
    "preserve equations", "keep the math", "latex",
)


def _math_score(text):
    """Return a count of math-block matches in text (0 if none)."""
    return sum(len(p.findall(text)) for p in _MATH_PATTERNS)


def instruction_wants_math(instruction):
    """Heuristic: does the user's instruction imply math/equations matter?"""
    lower = instruction.lower()
    return any(kw in lower for kw in _MATH_INTENT_KEYWORDS)


def _get_model(model_name):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Retriever:
    def __init__(self, index_dir):
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.records = meta["records"]
        self.embedding_model = meta["embedding_model"]
        self.model = _get_model(self.embedding_model)

    def search(self, query, k=4, boost_math=False, math_boost_weight=0.08):
        """
        Top-k retrieval. If boost_math is True, chunks containing LaTeX/math
        blocks get a small additive bonus to their similarity score before
        ranking, so equation-bearing chunks surface more reliably when the
        user cares about preserving math (e.g. "keep the equations intact").
        This is the "math-aware retrieval" improvement proposed for SARAL
        (see Part B) - implemented here so Part A demonstrates it directly.

        We over-fetch candidates (k * 3) before boosting/re-ranking so that
        a math bonus can actually change which chunks land in the final
        top-k, rather than just re-ordering an already-truncated k-list.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        fetch_k = k * 3 if boost_math else k
        fetch_k = min(fetch_k, len(self.records)) or 1
        scores, idxs = self.index.search(q_emb, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            adj_score = float(score)
            math_hits = 0
            if boost_math:
                math_hits = _math_score(rec["text"])
                if math_hits > 0:
                    adj_score += math_boost_weight * min(math_hits, 3)
            candidates.append({
                "chunk_id": rec["chunk_id"],
                "page": rec["page"],
                "text": rec["text"],
                "score": adj_score,
                "math_hits": math_hits,
            })

        if boost_math:
            candidates.sort(key=lambda c: c["score"], reverse=True)

        return candidates[:k]


def format_context(results):
    """Format retrieved chunks for insertion into an LLM prompt, tagged by id."""
    lines = []
    for r in results:
        lines.append(f"[{r['chunk_id']} | page {r['page']}]\n{r['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, k=args.k)
    print(format_context(results))