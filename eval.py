"""
eval.py
Automatic evaluation for the SARAL chatbot prototype, as specified in the
assignment brief:
  - Citation coverage: what % of generated content has retrieved provenance
  - Factuality proxy: semantic similarity between each generated sentence
    and the source chunk(s) it cites (higher = better grounded)

This does NOT implement ROUGE/BERTScore-vs-human-script or human eval
(3 raters) - those require a human-authored reference script and human
annotators, which are out of scope for a 1-day prototype. They are the
natural next step (see README).

Usage:
    python eval.py --index data/index --audience "grad students" --length 90s \
        --style technical --instruction "Summarize the methodology"
"""

import argparse
import re

from sentence_transformers import SentenceTransformer, util

from generate import generate, citation_coverage
from retrieve import Retriever


def extract_cited_sentences(text):
    """Split generated text into sentences and pair each with its cited chunk_ids."""
    # crude sentence splitter - good enough for eval purposes
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    pairs = []
    for sent in sentences:
        chunk_ids = re.findall(r"\[C(\d+)\]", sent)
        if chunk_ids:
            clean_sent = re.sub(r"\[C\d+\]", "", sent).strip()
            pairs.append((clean_sent, [f"C{cid}" for cid in chunk_ids]))
    return pairs


def factuality_proxy(text, retrieved_chunks, model_name="all-MiniLM-L6-v2"):
    """
    For each cited sentence, compute cosine similarity between the sentence's
    embedding and the embedding of the chunk(s) it cites. Returns the average
    similarity across all cited sentences (0-1, higher = better grounded).
    """
    model = SentenceTransformer(model_name)
    chunk_lookup = {c["chunk_id"]: c["text"] for c in retrieved_chunks}

    pairs = extract_cited_sentences(text)
    if not pairs:
        return {"avg_similarity": None, "n_sentences_checked": 0, "details": []}

    details = []
    for sent, chunk_ids in pairs:
        valid_chunks = [chunk_lookup[cid] for cid in chunk_ids if cid in chunk_lookup]
        if not valid_chunks:
            continue
        sent_emb = model.encode(sent, normalize_embeddings=True)
        chunk_embs = model.encode(valid_chunks, normalize_embeddings=True)
        sims = util.cos_sim(sent_emb, chunk_embs)[0]
        max_sim = float(sims.max())
        details.append({"sentence": sent[:80] + "...", "chunk_ids": chunk_ids, "similarity": round(max_sim, 3)})

    avg = sum(d["similarity"] for d in details) / len(details) if details else None
    return {
        "avg_similarity": round(avg, 3) if avg is not None else None,
        "n_sentences_checked": len(details),
        "details": details,
    }


def run_eval(index_dir, audience, length, style, instruction, k=6):
    print(f"Generating output for eval (audience={audience}, length={length}, style={style})...")
    output, retrieved = generate(index_dir, audience, length, style, instruction, k=k)

    cov = citation_coverage(output)
    fact = factuality_proxy(output, retrieved)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Citation coverage: {cov}%  (% of non-empty lines with at least one [Cx] tag)")
    print(f"Factuality proxy (avg sentence-to-source similarity): {fact['avg_similarity']}")
    print(f"  (based on {fact['n_sentences_checked']} cited sentences checked)")
    print(f"Retrieved chunks: {[r['chunk_id'] for r in retrieved]}")

    print("\nNote: ROUGE/BERTScore-vs-human-script and 3-rater human eval are")
    print("out of scope for this automatic script (require human-authored")
    print("reference scripts / human annotators) - see README for next steps.")

    return {
        "citation_coverage_pct": cov,
        "factuality_proxy": fact,
        "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
        "output": output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--audience", default="grad students")
    parser.add_argument("--length", default="90s")
    parser.add_argument("--style", default="technical")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    run_eval(args.index, args.audience, args.length, args.style, args.instruction, k=args.k)