#!/usr/bin/env python3
"""
prh_permuted_null_streaming.py — Streaming permuted-label null for PRH paper (P4 §3.2).

Downloads one source model at a time from HF; cycles through all target partners
before deleting and moving to the next source. This groups N_pairs downloads into
N_unique_src + N_pairs total instead of 2 × N_pairs.

Peak disk usage: ~2 × largest_model_calibration_files ≈ 50-200 MB.

Resume-safe: pairs already fully written to the output CSV are skipped.

Usage:
    python alignment/prh_permuted_null_streaming.py
    python alignment/prh_permuted_null_streaming.py --n-trials 100 --out path/to/output.csv

Output columns: concept, source_model, target_model, hidden_dim, trial,
                raw_cosine, aligned_cosine
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from rosetta_tools.alignment import compute_procrustes_rotation, apply_rotation, cosine_similarity

try:
    from rosetta_tools.paths import ROSETTA_RESULTS, ROSETTA_MODELS
except ImportError:
    ROSETTA_RESULTS = Path.home() / "rosetta_data" / "results"
    ROSETTA_MODELS = Path.home() / "rosetta_data" / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HF_REPO = "james-ra-henry/Rosetta-Activations"
PAPER_PREFIX = "paper_n250"

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

DEFAULT_PAIRS_CSV = ROSETTA_RESULTS / "PRH" / "alignment_results_samedim_n250.csv"
DEFAULT_OUT = ROSETTA_RESULTS / "PRH" / "null_permuted_all_n250.csv"
DEFAULT_SCRATCH = Path.home() / "rosetta_data" / "null_scratch"


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------


def build_model_dir_map(scratch_dir: Path) -> dict[str, str]:
    """
    Download run_summary.json for all paper_n250 models; return {model_id: dir_name}.
    These files are tiny (~1 KB each) and stay in scratch for the duration of the run.
    """
    log.info("Building model→dir map from HF run_summary.json files...")
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        local_dir=scratch_dir,
        allow_patterns=[f"{PAPER_PREFIX}/*/run_summary.json"],
    )
    mapping: dict[str, str] = {}
    for summary_path in sorted((scratch_dir / PAPER_PREFIX).glob("*/run_summary.json")):
        dir_name = summary_path.parent.name
        with summary_path.open() as f:
            summary = json.load(f)
        model_id = summary.get("model_id", "")
        if model_id:
            mapping[model_id] = dir_name
    log.info("Found %d models in paper_n250.", len(mapping))
    return mapping


def download_model_files(dir_name: str, scratch_dir: Path) -> None:
    """Download caz_*.json + peak-layer calibration_*.npy for one model."""
    allow_patterns = [f"{PAPER_PREFIX}/{dir_name}/caz_*.json"] + [
        f"{PAPER_PREFIX}/{dir_name}/calibration_{c}.npy" for c in CONCEPTS
    ]
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        local_dir=scratch_dir,
        allow_patterns=allow_patterns,
    )


def delete_model_files(dir_name: str, scratch_dir: Path) -> None:
    """Remove one model's downloaded files from scratch dir."""
    model_path = scratch_dir / PAPER_PREFIX / dir_name
    if model_path.exists():
        shutil.rmtree(model_path)
        log.debug("Deleted scratch: %s", model_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_data(
    dir_name: str, scratch_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """
    Load caz vectors and calibration activations for all concepts.
    Returns: (vectors, activations, hidden_dim)
    """
    model_dir = scratch_dir / PAPER_PREFIX / dir_name
    vectors: dict[str, np.ndarray] = {}
    activations: dict[str, np.ndarray] = {}
    hidden_dim = 0

    for concept in CONCEPTS:
        caz_path = model_dir / f"caz_{concept}.json"
        cal_path = model_dir / f"calibration_{concept}.npy"

        if caz_path.exists():
            with caz_path.open() as f:
                data = json.load(f)
            if not hidden_dim and "hidden_dim" in data:
                hidden_dim = data["hidden_dim"]
            layer_data = data.get("layer_data", {})
            peak_layer = layer_data.get("peak_layer")
            metrics = layer_data.get("metrics", [])
            peak_m = next((m for m in metrics if m["layer"] == peak_layer), None)
            if peak_m and "dom_vector" in peak_m:
                vectors[concept] = np.array(peak_m["dom_vector"])

        if cal_path.exists():
            activations[concept] = np.load(cal_path)

    return vectors, activations, hidden_dim


# ---------------------------------------------------------------------------
# Null computation
# ---------------------------------------------------------------------------


def _permuted_dom_vector(cal_acts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DOM vector with randomly shuffled class labels (same convention as align.py)."""
    n = cal_acts.shape[0]
    n_pos = n // 2
    perm = rng.permutation(n)
    pos = cal_acts[perm[:n_pos]]
    neg = cal_acts[perm[n_pos:]]
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    return diff / (norm + 1e-10)


def run_pair_null(
    src_id: str,
    tgt_id: str,
    src_acts: dict[str, np.ndarray],
    tgt_acts: dict[str, np.ndarray],
    hidden_dim: int,
    n_trials: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Run permuted-label null for all concepts for one ordered model pair."""
    rows: list[dict] = []
    for concept in CONCEPTS:
        if concept not in src_acts or concept not in tgt_acts:
            continue

        sa = src_acts[concept].astype(np.float64)
        ta = tgt_acts[concept].astype(np.float64)

        try:
            R = compute_procrustes_rotation(sa, ta)
        except Exception as e:
            log.warning("  %s: Procrustes failed %s × %s: %s — skip",
                        concept, src_id.split("/")[-1], tgt_id.split("/")[-1], e)
            continue

        for trial in range(n_trials):
            perm_src = _permuted_dom_vector(sa, rng)
            perm_tgt = _permuted_dom_vector(ta, rng)
            rotated = apply_rotation(perm_tgt, R)
            rows.append({
                "concept": concept,
                "source_model": src_id,
                "target_model": tgt_id,
                "hidden_dim": hidden_dim,
                "trial": trial,
                "raw_cosine": float(cosine_similarity(perm_src, perm_tgt)),
                "aligned_cosine": float(cosine_similarity(perm_src, rotated)),
            })

    return rows


def append_rows(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    if out_path.exists():
        df.to_csv(out_path, mode="a", index=False, header=False)
    else:
        df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    if not args.pairs_csv.exists():
        raise FileNotFoundError(
            f"Pairs CSV not found: {args.pairs_csv}\n"
            "Download it first:\n"
            "  hf download james-ra-henry/Rosetta-Activations --repo-type dataset "
            "--local-dir ~/rosetta_data --include 'results/PRH/alignment_results_samedim_n250.csv'"
        )

    model_dir_map = build_model_dir_map(args.scratch)

    pairs_df = pd.read_csv(args.pairs_csv)
    all_pairs = (
        pairs_df[["source_model", "target_model", "src_hidden_dim"]]
        .drop_duplicates()
        .sort_values("source_model")
        .values.tolist()
    )
    log.info("Total same-dim pairs: %d", len(all_pairs))

    # Resume: which (src, tgt) pairs already have all n_trials rows per concept?
    done_pairs: set[tuple[str, str]] = set()
    if args.out.exists():
        existing = pd.read_csv(args.out)
        counts = (
            existing.groupby(["source_model", "target_model", "concept"])
            .size()
            .reset_index(name="n")
        )
        fully_done = counts[counts["n"] >= args.n_trials]
        pair_concept_counts = fully_done.groupby(["source_model", "target_model"]).size()
        for (src, tgt), n_concepts_done in pair_concept_counts.items():
            # Mark done only if all expected concepts are present
            done_pairs.add((src, tgt))
        log.info("Resume: %d pairs already fully written.", len(done_pairs))

    rng = np.random.default_rng(args.seed)
    n_new = 0
    pair_idx = 0

    # Group by source model to minimize HF re-downloads
    for src_id, group_iter in groupby(all_pairs, key=lambda x: x[0]):
        group = list(group_iter)
        remaining = [(tgt_id, hdim) for (_, tgt_id, hdim) in group
                     if (src_id, tgt_id) not in done_pairs]

        if not remaining:
            pair_idx += len(group)
            log.info("Source %s: all %d pairs already done — skip",
                     src_id.split("/")[-1], len(group))
            continue

        src_dir = model_dir_map.get(src_id)
        if src_dir is None:
            log.warning("No HF dir for source %s — skip all its pairs", src_id)
            pair_idx += len(group)
            continue

        log.info("=== Source: %s (%d pairs to run) ===",
                 src_id.split("/")[-1], len(remaining))

        log.info("  Downloading src %s...", src_id)
        download_model_files(src_dir, args.scratch)
        src_vecs, src_acts, src_dim = load_model_data(src_dir, args.scratch)

        if not src_acts:
            log.warning("  No calibration data for %s — skip all its pairs", src_id)
            delete_model_files(src_dir, args.scratch)
            pair_idx += len(group)
            continue

        for tgt_id, hidden_dim in remaining:
            pair_idx += 1
            tgt_dir = model_dir_map.get(tgt_id)
            if tgt_dir is None:
                log.warning("[%d/%d] No HF dir for target %s — skip",
                            pair_idx, len(all_pairs), tgt_id)
                continue

            log.info("[%d/%d] %s × %s (dim=%d)",
                     pair_idx, len(all_pairs),
                     src_id.split("/")[-1], tgt_id.split("/")[-1], hidden_dim)

            log.info("  Downloading tgt %s...", tgt_id)
            download_model_files(tgt_dir, args.scratch)
            _, tgt_acts, _ = load_model_data(tgt_dir, args.scratch)

            n_concepts = sum(1 for c in CONCEPTS if c in src_acts and c in tgt_acts)
            log.info("  %d concepts with calibration on both sides", n_concepts)

            rows = run_pair_null(
                src_id, tgt_id,
                src_acts, tgt_acts,
                int(hidden_dim), args.n_trials, rng,
            )

            if rows:
                append_rows(rows, args.out)
                n_new += 1
                log.info("  Wrote %d rows (%d concepts × %d trials)",
                         len(rows), len(rows) // args.n_trials, args.n_trials)
            else:
                log.warning("  No rows produced for this pair")

            delete_model_files(tgt_dir, args.scratch)

        delete_model_files(src_dir, args.scratch)

    log.info("Done. Ran %d new pairs. Output: %s", n_new, args.out)

    if args.out.exists():
        final = pd.read_csv(args.out)
        n_pairs = len(final.groupby(["source_model", "target_model"]))
        log.info("Final CSV: %d rows, %d pairs, mean aligned=%.4f, mean raw=%.4f",
                 len(final), n_pairs,
                 final["aligned_cosine"].mean(),
                 final["raw_cosine"].mean())


if __name__ == "__main__":
    main()
