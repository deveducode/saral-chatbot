"""
retrieve.py
Pipeline stage 2: given a natural-language query, embed it with the same
model used at ingest time, search the FAISS index, and return the top-k
chunks along with their provenance (chunk_id, page number, similarity score).

This is imported by generate.py — but you can also run it standalone to
sanity-check retrieval quality before wiring up the LLM:

    python retrieve.py --index data/index --query "what does the RAG layer do?" --k 4

Features:
- Math-aware retrieval boosting (up-weights equation-bearing chunks when the
  user's instruction implies math should be preserved)
- LRU result cache to avoid redundant embedding + FAISS search during
  iterative editing sessions (scalable-architecture requirement)
"""

import argparse
import functools
import hashlib
import json
import logging
import os
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import math patterns from ingest.py (single source of truth — DRY)
from ingest import MATH_PATTERNS

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}

# If the user's instruction contains any of these, we treat it as a signal
# that math-preserving retrieval should be favored.
_MATH_INTENT_KEYWORDS = (
    "equation", "equations", "formula", "formulas", "math", "mathematical",
    "preserve equations", "keep the math", "latex",
)


def _math_score(text: str) -> int:
    """Return a count of math-block matches in *text* (0 if none)."""
    return sum(len(p.findall(text)) for p in MATH_PATTERNS)


def instruction_wants_math(instruction: str) -> bool:
    """Heuristic: does the user's instruction imply math/equations matter?"""
    lower = instruction.lower()
    return any(kw in lower for kw in _MATH_INTENT_KEYWORDS)


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for *model_name*."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Retriever:
    """Semantic retrieval over a FAISS index built by ``ingest.py``.

    Provides LRU-cached search to avoid redundant embedding + FAISS lookups
    during iterative editing sessions where the user repeatedly refines the
    same query or asks follow-up questions against the same paper.
    """

    _CACHE_MAX_SIZE = 128

    def __init__(self, index_dir: str) -> None:
        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")

        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                "Run `python ingest.py --pdf <file> --out <dir>` first."
            )

        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.records: list[dict] = meta["records"]
        self.embedding_model: str = meta["embedding_model"]
        self.model = _get_model(self.embedding_model)

        # Cache statistics (for observability / scalability evidence)
        self._cache_hits = 0
        self._cache_misses = 0

        # Build the per-instance LRU cache (functools.lru_cache on a method
        # requires wrapping because `self` isn't hashable by default).
        self._cached_search = functools.lru_cache(maxsize=self._CACHE_MAX_SIZE)(
            self._search_impl
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 4,
        boost_math: bool = False,
        math_boost_weight: float = 0.08,
    ) -> list[dict]:
        """Top-k retrieval with optional math-aware boosting.

        If *boost_math* is ``True``, chunks containing LaTeX/math blocks get a
        small additive bonus to their similarity score before ranking, so
        equation-bearing chunks surface more reliably when the user cares about
        preserving math (e.g. "keep the equations intact").

        Results are LRU-cached by (query, k, boost_math) to avoid redundant
        work during iterative editing sessions.  Call :meth:`cache_stats` to
        inspect hit/miss counts.
        """
        # The LRU cache needs hashable args — all three are already hashable.
        results = self._cached_search(query, k, boost_math, math_boost_weight)

        # Update stats (we can't easily distinguish hit/miss from lru_cache
        # directly, so we track total calls and compare to cache_info).
        info = self._cached_search.cache_info()
        self._cache_hits = info.hits
        self._cache_misses = info.misses

        return results

    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss counts for observability."""
        info = self._cached_search.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "max_size": info.maxsize,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_impl(
        self,
        query: str,
        k: int,
        boost_math: bool,
        math_boost_weight: float,
    ) -> tuple[dict, ...]:
        """Core search logic (returns a tuple for LRU-cache hashability).

        We over-fetch candidates (k * 3) before boosting/re-ranking so that
        a math bonus can actually change which chunks land in the final
        top-k, rather than just re-ordering an already-truncated k-list.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return ()

        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        fetch_k = k * 3 if boost_math else k
        fetch_k = min(fetch_k, self.index.ntotal) or 1
        scores, idxs = self.index.search(q_emb, fetch_k)

        candidates: list[dict] = []
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

        # Return as tuple (hashable) for LRU cache compatibility
        return tuple(candidates[:k])


def format_context(results: list[dict] | tuple[dict, ...]) -> str:
    """Format retrieved chunks for insertion into an LLM prompt, tagged by id."""
    lines: list[str] = []
    for r in results:
        lines.append(f"[{r['chunk_id']} | page {r['page']}]\n{r['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, k=args.k)
    print(format_context(results))
    stats = retriever.cache_stats()
    logger.info("Cache stats: %s", stats)