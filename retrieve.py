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

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}


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

    def search(self, query, k=4):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            results.append({
                "chunk_id": rec["chunk_id"],
                "page": rec["page"],
                "text": rec["text"],
                "score": float(score),
            })
        return results


def format_context(results):
    """Format retrieved chunks for insertion into an LLM prompt, tagged by id."""
    lines = []
    for r in results:
        lines.append(f"[{r['chunk_id']} | page {r['page']}]\n{r['text']}\n")
    return "\n".join(lines)


def retrieve(query, k=5, index_dir="data/index"):
    """Retrieve the top-k chunks from the project's default index."""
    return Retriever(index_dir).search(query, k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, k=args.k)
    print(format_context(results))
