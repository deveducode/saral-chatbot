"""
eval_multi.py
Runs the ingest -> generate -> eval pipeline across multiple papers and
produces a markdown comparison table, as specified in the brief's
"Quality: ... for a small test set (3 papers)" requirement.

For each paper:
  1. Builds a fresh FAISS index at data/index_<paper_stem>/
  2. Generates a summary with the given audience/length/style/instruction
  3. Computes citation coverage + factuality proxy
Then prints (and saves) a markdown comparison table across all papers.

Usage:
    python eval_multi.py --papers data/paper1.pdf data/paper2.pdf data/paper3.pdf \
        --instruction "Summarize the methodology used" \
        --audience "grad students" --length 90s --style technical
"""

import argparse
import os
from pathlib import Path

from ingest import build_index
from eval import run_eval


def run_multi_eval(paper_paths, audience, length, style, instruction, k=6,
                    out_md="eval_comparison.md"):
    rows = []
    for paper_path in paper_paths:
        stem = Path(paper_path).stem
        index_dir = f"data/index_{stem}"

        print(f"\n{'#' * 60}")
        print(f"# Paper: {paper_path}")
        print(f"{'#' * 60}")

        if not os.path.exists(os.path.join(index_dir, "index.faiss")):
            print(f"Building index at {index_dir} ...")
            build_index(paper_path, index_dir)
        else:
            print(f"Reusing existing index at {index_dir}")

        result = run_eval(index_dir, audience, length, style, instruction, k=k)
        rows.append({
            "paper": stem,
            "citation_coverage_pct": result["citation_coverage_pct"],
            "factuality_avg_similarity": result["factuality_proxy"]["avg_similarity"],
            "n_sentences_checked": result["factuality_proxy"]["n_sentences_checked"],
        })

    # Build markdown comparison table
    lines = []
    lines.append(f"## Multi-paper evaluation comparison\n")
    lines.append(f"Instruction: \"{instruction}\" | audience={audience}, length={length}, style={style}\n")
    lines.append("| Paper | Citation coverage (%) | Factuality proxy (avg similarity) | Sentences checked |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['paper']} | {r['citation_coverage_pct']}% | "
            f"{r['factuality_avg_similarity']} | {r['n_sentences_checked']} |"
        )

    avg_cov = round(sum(r["citation_coverage_pct"] for r in rows) / len(rows), 1)
    fact_vals = [r["factuality_avg_similarity"] for r in rows if r["factuality_avg_similarity"] is not None]
    avg_fact = round(sum(fact_vals) / len(fact_vals), 3) if fact_vals else None
    lines.append(f"\n**Average citation coverage: {avg_cov}%**")
    lines.append(f"**Average factuality proxy: {avg_fact}**")

    md = "\n".join(lines)
    print("\n" + "=" * 60)
    print(md)

    with open(out_md, "w") as f:
        f.write(md + "\n")
    print(f"\nSaved comparison table to {out_md}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", required=True, help="Paths to 2-3 PDF papers")
    parser.add_argument("--audience", default="grad students")
    parser.add_argument("--length", default="90s")
    parser.add_argument("--style", default="technical")
    parser.add_argument("--instruction", default="Summarize the methodology used")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    run_multi_eval(args.papers, args.audience, args.length, args.style,
                    args.instruction, k=args.k)