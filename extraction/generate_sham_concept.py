"""
generate_sham_concept.py — Create a sham concept dataset with no semantic contrast.

Draws 200 texts at random from the existing concept pair pool and assigns
them to 100 arbitrary pairs with random pos/neg labels.  If the permutation
null model is working correctly, the sham concept should produce p(sep) ≈ 0.5
— the pipeline should not find meaningful separation between random halves.

Usage
-----
    python src/generate_sham_concept.py
"""

import json
import random
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"

CONCEPT_FILES = [
    "credibility_pairs.jsonl",
    "negation_pairs.jsonl",
    "sentiment_pairs.jsonl",
    "causation_pairs.jsonl",
    "certainty_pairs.jsonl",
    "moral_valence_pairs.jsonl",
    "temporal_order_pairs.jsonl",
]


def main():
    # Collect all texts
    all_texts = []
    for fname in CONCEPT_FILES:
        with open(DATA_ROOT / fname) as f:
            for line in f:
                rec = json.loads(line)
                all_texts.append(rec["text"])

    # Deterministic shuffle
    rng = random.Random(42)
    rng.shuffle(all_texts)

    # Take first 200, pair them arbitrarily
    texts = all_texts[:200]
    out_path = DATA_ROOT / "sham_pairs.jsonl"

    with open(out_path, "w") as f:
        for i in range(100):
            pair_id = f"sham_{i+1:03d}"
            # Even index = label 1, odd index = label 0
            pos_rec = {
                "pair_id": pair_id,
                "label": 1,
                "domain": "sham",
                "model_name": "random_assignment",
                "text": texts[2 * i],
                "topic": "no_concept",
                "concept": "sham",
            }
            neg_rec = {
                "pair_id": pair_id,
                "label": 0,
                "domain": "sham",
                "model_name": "random_assignment",
                "text": texts[2 * i + 1],
                "topic": "no_concept",
                "concept": "sham",
            }
            f.write(json.dumps(pos_rec) + "\n")
            f.write(json.dumps(neg_rec) + "\n")

    print(f"Wrote {out_path} — 100 sham pairs (200 texts, random assignment)")


if __name__ == "__main__":
    main()
