"""
test_change.py
Quick manual demo for the iterative change-tracking feature (apply_change in
generate.py). Runs the full pipeline end-to-end: generate -> edit -> show diff.

This is NOT a pytest test — it's a manual script that requires:
  - A built FAISS index at data/index/
  - A valid GEMINI_API_KEY in .env

Usage:
    python test_change.py
"""

from generate import generate, generate_structured, apply_change

INDEX_DIR = "data/index"


def main():
    print("=" * 60)
    print("STEP 1: Initial generation (structured)")
    print("=" * 60)
    structured, retrieved = generate_structured(
        INDEX_DIR,
        audience="grad students",
        length="90s",
        style="technical",
        instruction="Summarize the main methodology used for landslide susceptibility assessment",
    )
    print(structured.raw)
    print(f"\nRetrieved chunks: {[r['chunk_id'] for r in retrieved]}")

    if structured.tweet_abstract:
        print(f"\nTweet abstract ({len(structured.tweet_abstract)} chars):")
        print(f"  {structured.tweet_abstract}")

    print("\n" + "=" * 60)
    print("STEP 2: Applying change instruction")
    print("=" * 60)
    change_instruction = "Make this less technical, use plain-English style instead"
    new_output, delta, why_changed = apply_change(
        INDEX_DIR, structured.raw, change_instruction
    )

    print("\n--- NEW OUTPUT ---")
    print(new_output)

    print("\n--- DELTA (unified diff) ---")
    for line in delta:
        print(line)

    print("\n--- WHY CHANGED ---")
    print(why_changed)


if __name__ == "__main__":
    main()
