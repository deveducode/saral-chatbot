"""
test_change.py
Quick manual test for the iterative change-tracking feature (apply_change in generate.py).
"""

from generate import generate, apply_change

INDEX_DIR = "data/index"

def main():
    print("=" * 60)
    print("STEP 1: Initial generation")
    print("=" * 60)
    output, retrieved = generate(
        INDEX_DIR,
        audience="grad students",
        length="90s",
        style="technical",
        instruction="Summarize the main methodology used for landslide susceptibility assessment",
    )
    print(output)
    print(f"\nRetrieved chunks: {[r['chunk_id'] for r in retrieved]}")

    print("\n" + "=" * 60)
    print("STEP 2: Applying change instruction")
    print("=" * 60)
    change_instruction = "Make this less technical, use plain-English style instead"
    new_output, delta, why_changed = apply_change(
        INDEX_DIR, output, change_instruction
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
