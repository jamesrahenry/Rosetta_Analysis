"""
generate_sham_concept.py — Create a sham concept dataset with no semantic contrast.

Draws 200 texts at random from the full C=17 concept pair pool (via
rosetta_tools.dataset.load_concept_pairs) and assigns them to 100 arbitrary
pairs with random pos/neg labels.  If the permutation null model is working
correctly, the sham concept should produce p(sep) ≈ 0.5 — the pipeline should
not find meaningful separation between random halves of a semantically
incoherent pairing.

The output is written to <rosetta_analysis>/data/sham_pairs.jsonl and
committed to the repo so the GPU host does not need ROSETTA_CONCEPTS_ROOT
for the sham control specifically.

Usage
-----
    python extraction/generate_sham_concept.py
    python extraction/generate_sham_concept.py --n-pairs 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ALL_CONCEPTS = [
    "credibility",
    "negation",
    "sentiment",
    "causation",
    "certainty",
    "moral_valence",
    "temporal_order",
    "specificity",
    "plurality",
    "agency",
    "formality",
    "threat_severity",
    "authorization",
    "urgency",
    "sarcasm",
    "deception",
    "exfiltration",
]

OUT_PATH = Path(__file__).parent.parent / "data" / "sham_pairs.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Number of sham pairs to generate (default: 100)")
    parser.add_argument("--texts-per-concept", type=int, default=20,
                        help="Texts to draw per concept (default: 20; "
                             "17 concepts × 20 = 340 texts available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic output (default: 42)")
    args = parser.parse_args()

    from rosetta_tools.dataset import load_concept_pairs

    # Collect texts from all 17 concepts
    all_texts: list[str] = []
    for concept in ALL_CONCEPTS:
        try:
            pairs = load_concept_pairs(concept, n=args.texts_per_concept)
            for pair in pairs:
                all_texts.append(pair.pos_text)
                all_texts.append(pair.neg_text)
        except Exception as e:
            print(f"  Warning: could not load {concept}: {e}")

    n_available = len(all_texts)
    n_needed = args.n_pairs * 2
    if n_available < n_needed:
        raise ValueError(
            f"Need {n_needed} texts for {args.n_pairs} pairs but only "
            f"collected {n_available} from {len(ALL_CONCEPTS)} concepts. "
            f"Increase --texts-per-concept or reduce --n-pairs."
        )

    # Deterministic shuffle, then take the first n_needed texts
    rng = random.Random(args.seed)
    rng.shuffle(all_texts)
    texts = all_texts[:n_needed]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i in range(args.n_pairs):
            pair_id = f"sham_{i + 1:03d}"
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

    print(
        f"Wrote {OUT_PATH} — {args.n_pairs} sham pairs "
        f"({n_needed} texts drawn from C=17, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
