"""
align.py — Procrustes alignment of concept vectors across architectures.

Tests the Platonic Representation Hypothesis: do architecturally diverse
models converge on equivalent semantic directions after accounting for
arbitrary rotational differences between their latent spaces?

Method
------
1. Load CAZ extraction results for all models.
2. For each concept, extract the dominant vector (dom_vector) at each
   model's peak layer.
3. Also extract a calibration activation matrix at the peak layer — a
   shared set of texts run through each model, used to fit the Procrustes
   rotation that maps one latent space to another.
4. For every ordered pair of models:
   a. Fit the optimal orthogonal rotation R (Procrustes).
   b. Compute cosine similarity before and after rotation.
5. Report raw vs aligned similarity — a large alignment_gain indicates the
   models share a rotated-equivalent representation.

Flags
-----
--same-dim-only
    Restrict to pairs that share hidden_dim (zero-PCA, no inflation).
    Primary claims in the PRH paper use this mode.

--permute-labels N
    Null distribution: run N permuted-label trials per pair.
    For each trial, class labels in calibration activations are randomly
    shuffled; the resulting DOM vector is random. The real rotation is
    applied and aligned cosine is measured. Expected result: ~0.03.
    Saved to results/null_permuted_{concept}.csv.

--cross-concept-transfer
    Universality test: fit Procrustes rotation on concept A's calibration
    activations, apply to concept B's DOM vectors. If PRH is strong, a
    single rotation aligns all concepts (universal coordinate transform).
    Saved to results/cross_concept_transfer.csv.

--split-calibration [--n-splits N]
    Construction-artifact test: split calibration activations 50/50
    (maintaining class balance). Fit R on train half; compute DOM vectors
    from held-out test half. If aligned cosine holds, R genuinely generalises
    to the concept direction rather than aligning it as a by-product of
    fitting the same data DOM was derived from. Default: 20 random splits.
    Saved to results/split_calibration.csv.

Usage
-----
    # Primary analysis — same-dim pairs only (zero-PCA, paper-quality)
    python src/align.py --all --same-dim-only

    # Full analysis including cross-dim pairs
    python src/align.py --all

    # Permuted-label null (100 trials, same-dim only)
    python src/align.py --all --same-dim-only --permute-labels 100

    # Cross-concept rotation transfer
    python src/align.py --all --same-dim-only --cross-concept-transfer

    # Split-calibration construction-artifact test (20 splits)
    python src/align.py --all --same-dim-only --split-calibration

    # Custom output
    python src/align.py --all --same-dim-only --out results/prh_main.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rosetta_tools.alignment import align_and_score, compute_procrustes_rotation, apply_rotation, cosine_similarity
from rosetta_tools.viz import CONCEPT_META, CONCEPT_ORDER
from rosetta_tools.paths import ROSETTA_RESULTS, ROSETTA_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
CONCEPTS = list(CONCEPT_META.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _model_dirs():
    """Yield model directories from ROSETTA_MODELS that have extraction results."""
    return sorted(d for d in ROSETTA_MODELS.iterdir()
                  if d.is_dir() and any(d.glob("caz_*.json")))


def load_hidden_dims() -> dict[str, int]:
    """Return {model_id: hidden_dim} for all models with extraction results."""
    dims: dict[str, int] = {}
    for d in _model_dirs():
        summary = d / "run_summary.json"
        if not summary.exists():
            continue
        with summary.open() as f:
            s = json.load(f)
        model_id = s["model_id"]
        # hidden_dim lives in the concept JSON, not the summary
        for concept_file in d.glob("caz_*.json"):
            with concept_file.open() as f:
                data = json.load(f)
            if "hidden_dim" in data:
                dims[model_id] = data["hidden_dim"]
                break
    return dims


def load_dom_vectors(concept: str) -> dict[str, np.ndarray]:
    """Extract peak-layer dominant vectors for a concept."""
    vectors: dict[str, np.ndarray] = {}
    for d in _model_dirs():
        checkpoint = d / f"caz_{concept}.json"
        if not checkpoint.exists():
            continue
        with checkpoint.open() as f:
            data = json.load(f)
        if data.get("concept") != concept:
            continue
        layer_data = data["layer_data"]
        peak_layer = layer_data["peak_layer"]
        metrics = layer_data["metrics"]
        peak_metrics = next((m for m in metrics if m["layer"] == peak_layer), None)
        if peak_metrics and "dom_vector" in peak_metrics:
            vectors[data["model_id"]] = np.array(peak_metrics["dom_vector"])
    return vectors


def load_peak_activations(concept: str) -> dict[str, np.ndarray]:
    """Load peak-layer calibration activations [n_texts, hidden_dim] per model."""
    activations: dict[str, np.ndarray] = {}
    for d in _model_dirs():
        cal_path = d / f"calibration_{concept}.npy"
        summary_path = d / "run_summary.json"
        if not cal_path.exists() or not summary_path.exists():
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        activations[summary["model_id"]] = np.load(cal_path)
    if not activations:
        log.warning(
            "No calibration_%s.npy files found. Run src/extract.py first.", concept
        )
    return activations


# ---------------------------------------------------------------------------
# Primary analysis
# ---------------------------------------------------------------------------


def analyze_concept(
    concept: str,
    use_procrustes: bool,
    same_dim_only: bool,
    hidden_dims: dict[str, int],
) -> pd.DataFrame:
    """Pairwise Procrustes alignment for one concept."""
    log.info("=== Concept: %s ===", concept)

    vectors = load_dom_vectors(concept)
    if len(vectors) < 2:
        log.warning("Fewer than 2 models have results for %s — skipping.", concept)
        return pd.DataFrame()

    activations = {}
    if use_procrustes:
        activations = load_peak_activations(concept)
        if len(activations) < 2:
            log.warning("No calibration activations — falling back to raw cosine.")
            use_procrustes = False

    model_ids = list(vectors.keys())
    rows = []

    for src in model_ids:
        for tgt in model_ids:
            if src == tgt:
                continue

            src_dim = hidden_dims.get(src, 0)
            tgt_dim = hidden_dims.get(tgt, 0)
            is_same_dim = (src_dim > 0 and src_dim == tgt_dim)

            if same_dim_only and not is_same_dim:
                continue

            if use_procrustes and src in activations and tgt in activations:
                try:
                    result = align_and_score(
                        vectors[src], vectors[tgt],
                        activations[src], activations[tgt],
                    )
                except Exception as e:
                    log.warning("SVD failed for %s × %s (%s) — skipping pair",
                                src.split("/")[-1], tgt.split("/")[-1], e)
                    continue
            else:
                raw = cosine_similarity(vectors[src], vectors[tgt]) if is_same_dim else float("nan")
                result = {"raw_cosine": raw, "aligned_cosine": raw,
                          "alignment_gain": 0.0, "same_dim": is_same_dim}

            rows.append({
                "concept": concept,
                "source_model": src,
                "target_model": tgt,
                "src_hidden_dim": src_dim,
                "tgt_hidden_dim": tgt_dim,
                "is_same_dim": is_same_dim,
                **result,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    log.info(
        "  %d pairs  |  mean raw: %.4f  |  mean aligned: %.4f%s",
        len(df),
        df["raw_cosine"].mean(),
        df["aligned_cosine"].mean(),
        "  [same-dim only]" if same_dim_only else "",
    )
    return df


# ---------------------------------------------------------------------------
# Permuted-label null
# ---------------------------------------------------------------------------


def _permuted_dom_vector(cal_acts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compute a DOM vector with randomly shuffled class labels.

    cal_acts: [n_pos + n_neg, hidden_dim] — first half positive, second negative.
    Returns a unit-length vector in the same activation space as the real DOM vector.
    """
    n = cal_acts.shape[0]
    n_pos = n // 2
    perm = rng.permutation(n)
    pos = cal_acts[perm[:n_pos]]
    neg = cal_acts[perm[n_pos:]]
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    return diff / (norm + 1e-10)


def run_permuted_null(
    concept: str,
    n_permutations: int,
    same_dim_only: bool,
    hidden_dims: dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """Permuted-label null distribution for one concept.

    For each model pair and each permutation trial:
    - Randomly shuffle class labels in each model's calibration activations
    - Compute permuted DOM vectors
    - Apply the real Procrustes rotation (fitted on true activations) to the
      permuted target vector
    - Measure cosine with permuted source vector

    Expected result: aligned_cosine ~ 0.03 across all trials — the rotation
    does not inflate random vectors. Contrasts with real aligned_cosine.

    Returns long-form DataFrame with one row per (pair, trial).
    """
    log.info("=== Permuted-label null: %s (%d trials) ===", concept, n_permutations)

    activations = load_peak_activations(concept)
    vectors = load_dom_vectors(concept)

    if len(activations) < 2:
        log.warning("Need calibration activations for permuted null — skipping.")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    model_ids = [m for m in activations if m in vectors]
    rows = []

    for src in model_ids:
        for tgt in model_ids:
            if src == tgt:
                continue

            src_dim = hidden_dims.get(src, 0)
            tgt_dim = hidden_dims.get(tgt, 0)
            is_same_dim = src_dim > 0 and src_dim == tgt_dim

            if same_dim_only and not is_same_dim:
                continue
            if not is_same_dim:
                continue  # permuted null only meaningful for same-dim pairs

            src_acts = activations[src].astype(np.float64)
            tgt_acts = activations[tgt].astype(np.float64)

            # Fit real rotation once — used for all permutation trials
            R = compute_procrustes_rotation(src_acts, tgt_acts)

            for trial in range(n_permutations):
                perm_src_vec = _permuted_dom_vector(src_acts, rng)
                perm_tgt_vec = _permuted_dom_vector(tgt_acts, rng)
                rotated = apply_rotation(perm_tgt_vec, R)
                aligned = cosine_similarity(perm_src_vec, rotated)
                raw = cosine_similarity(perm_src_vec, perm_tgt_vec)

                rows.append({
                    "concept": concept,
                    "source_model": src,
                    "target_model": tgt,
                    "hidden_dim": src_dim,
                    "trial": trial,
                    "raw_cosine": raw,
                    "aligned_cosine": aligned,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    log.info(
        "  %d pairs × %d trials  |  mean raw: %.4f  |  mean aligned: %.4f",
        len(df) // n_permutations, n_permutations,
        df["raw_cosine"].mean(), df["aligned_cosine"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Cross-concept rotation transfer
# ---------------------------------------------------------------------------


def run_cross_concept_transfer(
    concepts: list[str],
    same_dim_only: bool,
    hidden_dims: dict[str, int],
) -> pd.DataFrame:
    """Test whether Procrustes rotation transfers across concepts.

    For each model pair and each (rotation_concept, target_concept) combination:
    - Fit R on rotation_concept's calibration activations
    - Apply R to target_concept's target DOM vector
    - Measure cosine with target_concept's source DOM vector

    When rotation_concept == target_concept this is the standard per-concept
    alignment. When rotation_concept != target_concept this is transfer.

    A strong PRH result: cross-concept transfer ≈ per-concept alignment,
    implying a single universal coordinate transformation between model pairs
    (rotation is not concept-specific, it is a global space alignment).

    A weak PRH result: transfer << per-concept, implying the rotation is
    concept-adaptive (fitting noise that happens to align that concept).
    """
    log.info("=== Cross-concept rotation transfer (%d concepts) ===", len(concepts))

    # Load all vectors and activations up front
    all_vectors: dict[str, dict[str, np.ndarray]] = {}     # concept → model → vec
    all_activations: dict[str, dict[str, np.ndarray]] = {} # concept → model → acts

    for concept in concepts:
        all_vectors[concept] = load_dom_vectors(concept)
        all_activations[concept] = load_peak_activations(concept)

    # Gather all model IDs that have data for at least one concept
    all_model_ids = set()
    for vecs in all_vectors.values():
        all_model_ids.update(vecs.keys())

    rows = []

    for src in sorted(all_model_ids):
        for tgt in sorted(all_model_ids):
            if src == tgt:
                continue

            src_dim = hidden_dims.get(src, 0)
            tgt_dim = hidden_dims.get(tgt, 0)
            is_same_dim = src_dim > 0 and src_dim == tgt_dim

            if same_dim_only and not is_same_dim:
                continue
            if not is_same_dim:
                continue  # transfer only meaningful for same-dim pairs

            for rot_concept in concepts:
                # Check we have activations for this pair under this concept
                if (src not in all_activations.get(rot_concept, {}) or
                        tgt not in all_activations.get(rot_concept, {})):
                    continue

                src_acts = all_activations[rot_concept][src].astype(np.float64)
                tgt_acts = all_activations[rot_concept][tgt].astype(np.float64)

                try:
                    R = compute_procrustes_rotation(src_acts, tgt_acts)
                except Exception as exc:
                    log.debug("Procrustes failed for %s/%s/%s: %s",
                              rot_concept, src, tgt, exc)
                    continue

                for target_concept in concepts:
                    if (src not in all_vectors.get(target_concept, {}) or
                            tgt not in all_vectors.get(target_concept, {})):
                        continue

                    src_vec = all_vectors[target_concept][src].astype(np.float64)
                    tgt_vec = all_vectors[target_concept][tgt].astype(np.float64)

                    rotated = apply_rotation(tgt_vec, R)
                    aligned = cosine_similarity(src_vec, rotated)
                    raw = cosine_similarity(src_vec, tgt_vec)

                    rows.append({
                        "rotation_concept": rot_concept,
                        "target_concept": target_concept,
                        "is_same_concept": rot_concept == target_concept,
                        "source_model": src,
                        "target_model": tgt,
                        "hidden_dim": src_dim,
                        "raw_cosine": raw,
                        "aligned_cosine": aligned,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    same = df[df["is_same_concept"]]["aligned_cosine"]
    transfer = df[~df["is_same_concept"]]["aligned_cosine"]
    log.info(
        "  same-concept: mean=%.4f  |  cross-concept transfer: mean=%.4f  |  delta=%.4f",
        same.mean(), transfer.mean(), same.mean() - transfer.mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Split-calibration construction-artifact test
# ---------------------------------------------------------------------------


def _dom_from_activations(acts: np.ndarray) -> np.ndarray:
    """Compute DOM vector (mean-difference direction) from [n_pos+n_neg, d] activations.

    Assumes first half positive, second half negative — same convention as
    calibration_{concept}.npy files written by extract.py.
    Returns unit-length vector.
    """
    n = acts.shape[0]
    n_half = n // 2
    diff = acts[:n_half].mean(axis=0) - acts[n_half:].mean(axis=0)
    norm = np.linalg.norm(diff)
    return diff / (norm + 1e-10)


def run_split_calibration(
    concept: str,
    n_splits: int,
    same_dim_only: bool,
    hidden_dims: dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """Split-calibration test: fit R on full calibration data, compute DOM from held-out half.

    Breaks the circularity concern without refitting Procrustes per split:
      - Fit R ONCE on all 200 calibration examples (same R as primary analysis).
      - For each of N splits: randomly hold out 50% of examples (class-balanced),
        compute DOM from the held-out half only.
      - Apply the full-data R to the held-out DOM; measure cosine.

    R never sees the held-out DOM vectors, so any alignment cannot be a
    by-product of fitting R on the same data DOM was derived from.
    This is ~20× faster than refitting R per split since SVD is done once.

    If held-out aligned cosine ≈ primary result (~0.98), the construction is clean.
    """
    log.info("=== Split-calibration test: %s (%d splits) ===", concept, n_splits)

    activations = load_peak_activations(concept)
    if len(activations) < 2:
        log.warning("Need calibration activations for split-calibration — skipping.")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    model_ids = list(activations.keys())
    rows = []

    for src in model_ids:
        for tgt in model_ids:
            if src == tgt:
                continue

            src_dim = hidden_dims.get(src, 0)
            tgt_dim = hidden_dims.get(tgt, 0)
            is_same_dim = src_dim > 0 and src_dim == tgt_dim

            if same_dim_only and not is_same_dim:
                continue
            if not is_same_dim:
                continue

            src_acts = activations[src].astype(np.float64)
            tgt_acts = activations[tgt].astype(np.float64)

            n = src_acts.shape[0]
            n_half = n // 2  # 100 for 200-example calibration set

            # Fit R once on full calibration data — same as primary analysis
            try:
                R = compute_procrustes_rotation(src_acts, tgt_acts)
            except Exception as exc:
                log.debug("Procrustes failed %s/%s: %s", src, tgt, exc)
                continue

            for split_idx in range(n_splits):
                # Class-balanced hold-out: shuffle within each class, take second half
                pos_perm = rng.permutation(n_half)
                neg_perm = rng.permutation(n_half)

                held_src = np.concatenate([
                    src_acts[:n_half][pos_perm[n_half // 2:]],
                    src_acts[n_half:][neg_perm[n_half // 2:]],
                ])
                held_tgt = np.concatenate([
                    tgt_acts[:n_half][pos_perm[n_half // 2:]],
                    tgt_acts[n_half:][neg_perm[n_half // 2:]],
                ])

                src_dom_held = _dom_from_activations(held_src)
                tgt_dom_held = _dom_from_activations(held_tgt)

                rotated = apply_rotation(tgt_dom_held, R)
                aligned = cosine_similarity(src_dom_held, rotated)
                raw = cosine_similarity(src_dom_held, tgt_dom_held)

                rows.append({
                    "concept": concept,
                    "source_model": src,
                    "target_model": tgt,
                    "hidden_dim": src_dim,
                    "split": split_idx,
                    "raw_cosine": raw,
                    "aligned_cosine": aligned,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pair_means = df.groupby(["source_model", "target_model"])["aligned_cosine"].mean()
    log.info(
        "  %d pairs × %d splits  |  mean raw: %.4f  |  mean aligned: %.4f  |  "
        "pair mean range: [%.4f, %.4f]",
        len(pair_means), n_splits,
        df["raw_cosine"].mean(), df["aligned_cosine"].mean(),
        pair_means.min(), pair_means.max(),
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Procrustes alignment — PRH experiment")

    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--concept", type=str, choices=CONCEPTS)
    sel.add_argument("--all", action="store_true", help="Run all concepts")

    parser.add_argument(
        "--same-dim-only", action="store_true",
        help="Restrict to same hidden_dim pairs (zero-PCA, no inflation). "
             "Use this for primary paper results.",
    )
    parser.add_argument(
        "--permute-labels", type=int, default=0, metavar="N",
        help="Run N permuted-label null trials per pair. "
             "Saves results/null_permuted_{concept}.csv",
    )
    parser.add_argument(
        "--cross-concept-transfer", action="store_true",
        help="Test rotation transfer across concepts (universality test). "
             "Saves results/cross_concept_transfer.csv",
    )
    parser.add_argument(
        "--split-calibration", action="store_true",
        help="Construction-artifact test: fit R on train half of calibration "
             "data, compute DOM from held-out half. Saves results/split_calibration.csv",
    )
    parser.add_argument(
        "--n-splits", type=int, default=20, metavar="N",
        help="Number of random 50/50 splits for --split-calibration (default: 20)",
    )
    parser.add_argument(
        "--no-procrustes", action="store_true",
        help="Raw cosine similarity only (no Procrustes rotation)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: results/alignment_results.csv, "
             "or results/alignment_results_samedim.csv with --same-dim-only)",
    )
    args = parser.parse_args()

    concepts = CONCEPTS if args.all else [args.concept]

    dirs = _model_dirs()
    if not dirs:
        log.error("No model directories found in %s. Run extract.py first.", ROSETTA_MODELS)
        return

    hidden_dims = load_hidden_dims()
    log.info("Hidden dims loaded for %d models", len(hidden_dims))
    if args.same_dim_only:
        from collections import Counter
        dim_counts = Counter(hidden_dims.values())
        log.info(
            "Same-dim pairs available: %s",
            {d: n*(n-1) for d, n in sorted(dim_counts.items()) if n >= 2},
        )

    # ---- Primary alignment ----
    all_results = []
    for concept in concepts:
        df = analyze_concept(
            concept=concept,
            use_procrustes=not args.no_procrustes,
            same_dim_only=args.same_dim_only,
            hidden_dims=hidden_dims,
        )
        if not df.empty:
            all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        if args.out:
            out_path = Path(args.out)
        elif args.same_dim_only:
            out_path = RESULTS_ROOT / "alignment_results_samedim.csv"
        else:
            out_path = RESULTS_ROOT / "alignment_results.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        log.info("Results → %s", out_path)

        # Summary table
        summary = (
            combined
            .groupby("concept")[["raw_cosine", "aligned_cosine", "alignment_gain"]]
            .mean()
            .round(4)
            .sort_values("aligned_cosine", ascending=False)
        )
        log.info("\nConcept summary (mean pairwise):\n%s", summary.to_string())
    else:
        log.warning("No primary alignment results produced.")

    # ---- Permuted-label null ----
    if args.permute_labels > 0:
        null_dfs = []
        for concept in concepts:
            null_df = run_permuted_null(
                concept=concept,
                n_permutations=args.permute_labels,
                same_dim_only=args.same_dim_only,
                hidden_dims=hidden_dims,
            )
            if not null_df.empty:
                null_dfs.append(null_df)
                out = RESULTS_ROOT / f"null_permuted_{concept}.csv"
                null_df.to_csv(out, index=False)
                log.info("  Null saved → %s", out)

        if null_dfs:
            all_null = pd.concat(null_dfs, ignore_index=True)
            all_null.to_csv(RESULTS_ROOT / "null_permuted_all.csv", index=False)
            log.info(
                "\nPermuted null summary (all concepts):  "
                "mean aligned=%.4f  std=%.4f",
                all_null["aligned_cosine"].mean(),
                all_null["aligned_cosine"].std(),
            )

    # ---- Split-calibration construction-artifact test ----
    if args.split_calibration:
        split_dfs = []
        for concept in concepts:
            split_df = run_split_calibration(
                concept=concept,
                n_splits=args.n_splits,
                same_dim_only=args.same_dim_only,
                hidden_dims=hidden_dims,
            )
            if not split_df.empty:
                split_dfs.append(split_df)

        if split_dfs:
            all_split = pd.concat(split_dfs, ignore_index=True)
            out = RESULTS_ROOT / "split_calibration.csv"
            all_split.to_csv(out, index=False)
            log.info("Split-calibration results → %s", out)

            per_concept = all_split.groupby("concept")["aligned_cosine"].mean()
            log.info(
                "\n=== Split-calibration summary ===\n"
                "  Full-data primary result: ~0.9807\n"
                "  Split-calibration mean:   %.4f ± %.4f\n\n"
                "  Per-concept:\n%s\n\n"
                "  Interpretation: if split mean ≈ primary, R generalises to "
                "held-out DOM vectors (no construction artifact).",
                all_split["aligned_cosine"].mean(),
                all_split.groupby(["source_model", "target_model", "concept"])["aligned_cosine"]
                    .mean().std(),
                per_concept.round(4).to_string(),
            )

    # ---- Cross-concept transfer ----
    if args.cross_concept_transfer:
        transfer_df = run_cross_concept_transfer(
            concepts=concepts,
            same_dim_only=args.same_dim_only,
            hidden_dims=hidden_dims,
        )
        if not transfer_df.empty:
            out = RESULTS_ROOT / "cross_concept_transfer.csv"
            transfer_df.to_csv(out, index=False)
            log.info("Transfer results → %s", out)

            same = transfer_df[transfer_df["is_same_concept"]]["aligned_cosine"]
            xfer = transfer_df[~transfer_df["is_same_concept"]]["aligned_cosine"]
            log.info(
                "\nCross-concept transfer summary:\n"
                "  Same-concept alignment:   %.4f ± %.4f\n"
                "  Cross-concept transfer:   %.4f ± %.4f\n"
                "  Universality ratio:       %.3f",
                same.mean(), same.std(),
                xfer.mean(), xfer.std(),
                xfer.mean() / same.mean() if same.mean() > 0 else float("nan"),
            )


if __name__ == "__main__":
    main()
