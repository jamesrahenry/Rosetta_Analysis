#!/usr/bin/env python3
"""
peak_vs_nonpeak_xval.py — Cross-validated peak layer selection to break circularity.

The original peak_vs_nonpeak.py has a potential circularity: the CAZ peak layer
is identified using the same 250 concept pairs used to compute the DOM vector,
so comparing DOM alignment at "the peak" vs other layers might be partly true
by construction.

This script breaks that circularity using a held-out design:

  Half A (pairs 0:125, texts [0:125] and [250:375]):
      → Identify the xval peak: the layer where per-pair direction coherence
        is highest (i.e. the layer that *would* have been called the peak if
        you only had these pairs).  Specifically, for each layer compute the
        mean cosine of each pair's difference vector to the half-A DOM.  Peak
        = argmax over layers.

  Half B (pairs 125:250, texts [125:250] and [375:500]):
      → Compute DOM from half-B activations at the xval peak layer.
      → Fit Procrustes R using half-B calibration activations at the xval
        peak layer.
      → Score: cosine(src_dom_B @ R, tgt_dom_B).

For comparison, also score using the stored CAZ peak (identified on ALL 250
pairs, i.e. the "same-data peak" — the one with the potential circularity).

Output columns per (src_model, tgt_model, concept):
  xval_peak_layer      — peak found from half-A
  same_peak_layer      — peak from caz_{concept}.json (all-data)
  xval_peak_align      — aligned cosine at xval_peak_layer (half-B DOM, half-B R)
  same_peak_align      — aligned cosine at same_peak_layer (half-B DOM, half-B R)
  non_peak_mean        — mean aligned cosine at all OTHER layers (half-B DOM, half-B R)
  non_peak_best        — max aligned cosine at all other layers

The key question: does xval_peak_align still beat non_peak_mean?  If yes, the
peak advantage is NOT an artefact of circular evaluation.

Usage:
    uv run python alignment/peak_vs_nonpeak_xval.py
    uv run python alignment/peak_vs_nonpeak_xval.py --dim 768
    uv run python alignment/peak_vs_nonpeak_xval.py --out-dir /tmp/prh_xval

Requires calibration_alllayer_{concept}.npy files (same as Method B of
peak_vs_nonpeak.py).  Only Cluster A (768-dim) models have these by default.

Output:
    ~/rosetta_data/results/PRH/peak_vs_nonpeak_xval.json
"""
from __future__ import annotations

# Set single-threaded BLAS BEFORE numpy/scipy import.
# This prevents the WSL2 OpenBLAS deadlock that causes ~20× SVD slowdown.
import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import orthogonal_procrustes

import importlib
if importlib.util.find_spec("rosetta_tools") is None:
    import sys as _sys
    for _p in [Path.home() / "rosetta_tools",
               Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"]:
        if _p.exists():
            _sys.path.insert(0, str(_p))
            break

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

OUT_DIR = ROSETTA_RESULTS / "PRH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug(mid: str) -> str:
    return mid.replace("/", "_").replace("-", "_")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def fit_R(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: find R s.t. A @ R ≈ B (both row-centered)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    R, _ = orthogonal_procrustes(A, B)
    return R


def dom_from_acts(high_acts: np.ndarray, low_acts: np.ndarray) -> np.ndarray:
    """Compute normalised DOM vector from paired high/low activations."""
    diff = high_acts.mean(0) - low_acts.mean(0)
    norm = np.linalg.norm(diff)
    if norm < 1e-10:
        return diff
    return diff / norm


def xval_peak_layer(
    acts: np.ndarray,
    half_a_pairs: slice,
    half_a_low: slice,
) -> int:
    """Identify peak layer using half-A pairs only.

    Strategy: for each layer, compute DOM from half-A pairs, then measure
    the mean cosine of each pair's (high - low) direction to that DOM.
    This is a self-consistency score — the layer where the concept direction
    is most coherent across the half-A held-out pairs.

    acts: [n_layers, 500, hidden_dim]
    half_a_pairs: slice for high texts (e.g. slice(0, 125))
    half_a_low:   slice for low  texts (e.g. slice(250, 375))
    """
    n_layers = acts.shape[0]
    scores = np.zeros(n_layers)
    for layer in range(n_layers):
        layer_acts = acts[layer]
        high_a = layer_acts[half_a_pairs]   # [125, H]
        low_a  = layer_acts[half_a_low]     # [125, H]

        dom = dom_from_acts(high_a, low_a)  # [H]

        # Per-pair cosine of (high_k - low_k) to DOM
        pair_dirs = high_a - low_a          # [125, H]
        norms = np.linalg.norm(pair_dirs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pair_dirs_unit = pair_dirs / norms
        per_pair_cos = pair_dirs_unit @ dom   # [125]
        scores[layer] = float(per_pair_cos.mean())

    return int(np.argmax(scores))


def load_caz_peak(model_dir: Path, concept: str) -> Optional[int]:
    """Load the stored all-data CAZ peak_layer from caz_{concept}.json."""
    p = model_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return int(data["layer_data"]["peak_layer"])


def load_alllayer_acts(model_dir: Path, concept: str) -> Optional[np.ndarray]:
    """Load calibration_alllayer_{concept}.npy → [n_layers, 500, hidden_dim]."""
    p = model_dir / f"calibration_alllayer_{concept}.npy"
    if not p.exists():
        return None
    return np.load(p)


def model_dirs_by_dim() -> dict[int, list[tuple[str, Path]]]:
    """Group model directories by hidden_dim."""
    groups: dict[int, list[tuple[str, Path]]] = {}
    for d in sorted(ROSETTA_MODELS.iterdir()):
        if not d.is_dir():
            continue
        caz = d / "caz_credibility.json"
        rs  = d / "run_summary.json"
        if not caz.exists() or not rs.exists():
            continue
        try:
            summary  = json.loads(rs.read_text())
            mid      = summary["model_id"]
            caz_data = json.loads(caz.read_text())
            hdim     = caz_data["hidden_dim"]
        except Exception:
            continue
        groups.setdefault(hdim, []).append((mid, d))
    return groups


# ---------------------------------------------------------------------------
# Core xval evaluation
# ---------------------------------------------------------------------------

def run_xval(
    pairs: list[tuple[str, Path, str, Path]],
    concepts: list[str] | None = None,
) -> list[dict]:
    """Run cross-validated peak evaluation for all model pairs × concepts.

    For each (src, tgt, concept):
      1. Find xval peak using half-A activations from BOTH src and tgt
         (we use the src model's half-A to find its own peak, and similarly
         for tgt; then take the mean as the representative — but we also
         record them separately).
      2. Compute half-B DOM and fit half-B Procrustes R at every layer.
      3. Score xval_peak_layer, same_peak_layer, and all other layers.
    """
    if concepts is None:
        concepts = CONCEPTS

    rows = []

    for src_mid, src_dir, tgt_mid, tgt_dir in pairs:
        src_short = src_mid.split("/")[-1]
        tgt_short = tgt_mid.split("/")[-1]

        for concept in concepts:
            src_acts = load_alllayer_acts(src_dir, concept)
            tgt_acts = load_alllayer_acts(tgt_dir, concept)
            if src_acts is None or tgt_acts is None:
                continue

            n_layers = min(src_acts.shape[0], tgt_acts.shape[0])

            # Determine usable pair count from the actual text count.
            # Layout: texts [0:n_pairs] = high, [n_pairs:2*n_pairs] = low.
            # Require at least 250 texts (125 pairs) in each half.
            n_texts = min(src_acts.shape[1], tgt_acts.shape[1])
            n_pairs_total = n_texts // 2  # integer pairs (ignore orphan text)
            if n_pairs_total < 250:
                log.warning("%s × %s — %s: only %d usable pairs (need 250), skipping",
                            src_short, tgt_short, concept, n_pairs_total)
                continue

            # Recompute half boundaries based on actual pair count.
            # Use a clean 50/50 split of available pairs.
            half = n_pairs_total // 2        # e.g. 124 when n_pairs_total=249
            half_a_high = slice(0, half)
            half_a_low  = slice(n_pairs_total, n_pairs_total + half)
            half_b_high = slice(half, n_pairs_total)
            half_b_low  = slice(n_pairs_total + half, 2 * n_pairs_total)

            # ── Step 1: identify xval peaks from half-A ──────────────────
            src_xval_peak = xval_peak_layer(src_acts[:n_layers], half_a_high, half_a_low)
            tgt_xval_peak = xval_peak_layer(tgt_acts[:n_layers], half_a_high, half_a_low)

            # Combined xval peak: layer that maximises AVERAGE half-A
            # self-consistency across both models.
            #
            # We recompute the combined score rather than just averaging the
            # two argmaxes — gives a single shared layer to evaluate.
            src_scores_a = _layer_coherence_scores(src_acts[:n_layers], half_a_high, half_a_low)
            tgt_scores_a = _layer_coherence_scores(tgt_acts[:n_layers], half_a_high, half_a_low)
            combined_xval_peak = int(np.argmax((src_scores_a + tgt_scores_a) / 2))

            # ── Step 2: load same-data CAZ peaks ─────────────────────────
            src_same_peak = load_caz_peak(src_dir, concept)
            tgt_same_peak = load_caz_peak(tgt_dir, concept)
            if src_same_peak is None or tgt_same_peak is None:
                continue

            # For same-data comparison we use the src peak (matches Method B
            # convention in peak_vs_nonpeak.py).  We also store a combined
            # same-data peak (average of the two, rounded) for symmetry.
            combined_same_peak = int(round((src_same_peak + tgt_same_peak) / 2))

            log.info("  %s × %s — %s: xval_peak=%d same_peak=%d",
                     src_short, tgt_short, concept,
                     combined_xval_peak, combined_same_peak)

            # ── Step 3: fit half-B Procrustes R at EVERY layer, score DOM ─
            #
            # DOM at layer L from half-B pairs: dom_B_L
            # R_L fitted from half-B calibration activations at layer L
            # score = cos(src_dom_B_L @ R_L, tgt_dom_B_L)

            layer_scores: dict[int, float] = {}
            for layer in range(n_layers):
                try:
                    # Stack half-B high and low texts to get ~250 rows for Procrustes
                    src_b_stack = np.concatenate(
                        [src_acts[layer][half_b_high], src_acts[layer][half_b_low]], axis=0
                    )
                    tgt_b_stack = np.concatenate(
                        [tgt_acts[layer][half_b_high], tgt_acts[layer][half_b_low]], axis=0
                    )
                    R = fit_R(src_b_stack, tgt_b_stack)
                except Exception as e:
                    log.debug("SVD failed at layer %d (%s × %s, %s): %s",
                              layer, src_short, tgt_short, concept, e)
                    continue

                # half-B DOM for this layer
                src_dom_b = dom_from_acts(
                    src_acts[layer][half_b_high],
                    src_acts[layer][half_b_low],
                )
                tgt_dom_b = dom_from_acts(
                    tgt_acts[layer][half_b_high],
                    tgt_acts[layer][half_b_low],
                )

                aligned = src_dom_b @ R
                cos = cosine(aligned, tgt_dom_b)
                layer_scores[layer] = round(float(cos), 6)

            if not layer_scores:
                continue

            # ── Step 4: extract per-condition scores ──────────────────────
            xval_peak_score = layer_scores.get(combined_xval_peak)
            same_peak_score = layer_scores.get(combined_same_peak)

            non_peak_layers  = [l for l in layer_scores if l != combined_xval_peak]
            non_peak_scores  = [layer_scores[l] for l in non_peak_layers]
            non_peak_mean    = float(np.mean(non_peak_scores)) if non_peak_scores else None
            non_peak_best    = float(np.max(non_peak_scores))  if non_peak_scores else None

            # Separate non-peak excluding BOTH peaks (xval and same)
            non_either_layers = [l for l in layer_scores
                                 if l != combined_xval_peak and l != combined_same_peak]
            non_either_scores = [layer_scores[l] for l in non_either_layers]
            non_either_mean   = float(np.mean(non_either_scores)) if non_either_scores else None

            rows.append({
                "src_model":            src_mid,
                "tgt_model":            tgt_mid,
                "concept":              concept,
                "n_layers":             n_layers,
                # xval peaks (individual + combined)
                "src_xval_peak":        src_xval_peak,
                "tgt_xval_peak":        tgt_xval_peak,
                "combined_xval_peak":   combined_xval_peak,
                # same-data peaks
                "src_same_peak":        src_same_peak,
                "tgt_same_peak":        tgt_same_peak,
                "combined_same_peak":   combined_same_peak,
                # alignment scores at half-B DOM
                "xval_peak_align":      xval_peak_score,
                "same_peak_align":      same_peak_score,
                "non_peak_mean":        round(non_peak_mean, 6)  if non_peak_mean  is not None else None,
                "non_peak_best":        round(non_peak_best, 6)  if non_peak_best  is not None else None,
                "non_either_mean":      round(non_either_mean, 6) if non_either_mean is not None else None,
                # full layer scores (for diagnostic inspection)
                "layer_scores":         layer_scores,
            })

    return rows


def _layer_coherence_scores(
    acts: np.ndarray,
    half_high: slice,
    half_low: slice,
) -> np.ndarray:
    """Return per-layer mean cosine of pair dirs to DOM (half coherence scores)."""
    n_layers = acts.shape[0]
    scores = np.zeros(n_layers)
    for layer in range(n_layers):
        layer_acts = acts[layer]
        high = layer_acts[half_high]
        low  = layer_acts[half_low]
        dom  = dom_from_acts(high, low)
        pair_dirs = high - low
        norms = np.linalg.norm(pair_dirs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        per_pair_cos = (pair_dirs / norms) @ dom
        scores[layer] = float(per_pair_cos.mean())
    return scores


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(rows: list[dict]) -> None:
    if not rows:
        print("\nNo results to summarize.")
        return

    xval_scores  = [r["xval_peak_align"] for r in rows if r["xval_peak_align"]  is not None]
    same_scores  = [r["same_peak_align"]  for r in rows if r["same_peak_align"]  is not None]
    non_peak_m   = [r["non_peak_mean"]    for r in rows if r["non_peak_mean"]    is not None]
    non_either_m = [r["non_either_mean"]  for r in rows if r["non_either_mean"]  is not None]

    print("\n" + "=" * 68)
    print("  Cross-Validated Peak vs Non-Peak Summary")
    print("=" * 68)
    print(f"  Rows (concept × pair):      {len(rows)}")
    print()

    if xval_scores:
        print(f"  xval_peak_align:   mean={np.mean(xval_scores):.4f}  "
              f"median={np.median(xval_scores):.4f}  n={len(xval_scores)}")
    if same_scores:
        print(f"  same_peak_align:   mean={np.mean(same_scores):.4f}  "
              f"median={np.median(same_scores):.4f}  n={len(same_scores)}")
    if non_peak_m:
        print(f"  non_peak_mean:     mean={np.mean(non_peak_m):.4f}  "
              f"median={np.median(non_peak_m):.4f}  n={len(non_peak_m)}")
    if non_either_m:
        print(f"  non_either_mean:   mean={np.mean(non_either_m):.4f}  "
              f"(excl both peaks)")

    print()
    # Key question: does xval peak beat non-peak?
    paired = [(r["xval_peak_align"], r["non_peak_mean"]) for r in rows
              if r["xval_peak_align"] is not None and r["non_peak_mean"] is not None]
    if paired:
        xval_arr  = np.array([p[0] for p in paired])
        nonp_arr  = np.array([p[1] for p in paired])
        delta     = xval_arr - nonp_arr
        n_wins    = int((delta > 0).sum())
        mean_d    = float(delta.mean())
        print(f"  xval_peak > non_peak_mean:  {n_wins}/{len(paired)} "
              f"({n_wins / len(paired):.1%})  mean Δ={mean_d:+.4f}")

    # Also compare xval peak vs same-data peak (circularity check)
    paired2 = [(r["xval_peak_align"], r["same_peak_align"]) for r in rows
               if r["xval_peak_align"] is not None and r["same_peak_align"] is not None]
    if paired2:
        xval2    = np.array([p[0] for p in paired2])
        same2    = np.array([p[1] for p in paired2])
        delta2   = xval2 - same2
        mean_d2  = float(delta2.mean())
        n_wins2  = int((delta2 > 0).sum())
        print(f"  xval_peak vs same_peak:     xval {n_wins2}/{len(paired2)} wins  "
              f"mean Δ={mean_d2:+.4f}")
        if abs(mean_d2) < 0.01:
            print("  → Both peaks perform similarly: minimal circularity inflation.")
        elif mean_d2 < 0:
            print("  → same_peak is somewhat higher: small circularity boost detected.")
        else:
            print("  → xval_peak is higher: peak selection generalises.")

    # Per-concept breakdown
    print("\n  Per-concept (xval_peak_align — non_peak_mean):")
    print(f"  {'concept':<20}  {'xval':>6}  {'same':>6}  {'nonpk':>6}  {'Δxval-np':>9}  {'wins':>8}")
    print(f"  {'-'*20}  {'------':>6}  {'------':>6}  {'------':>6}  {'---------':>9}  {'--------':>8}")
    concepts = sorted({r["concept"] for r in rows})
    for c in concepts:
        c_rows = [r for r in rows if r["concept"] == c]
        c_xval = [r["xval_peak_align"] for r in c_rows if r["xval_peak_align"] is not None]
        c_same = [r["same_peak_align"]  for r in c_rows if r["same_peak_align"]  is not None]
        c_nonp = [r["non_peak_mean"]    for r in c_rows if r["non_peak_mean"]    is not None]
        if not c_xval or not c_nonp:
            continue
        mx = np.mean(c_xval)
        ms = np.mean(c_same) if c_same else float("nan")
        mn = np.mean(c_nonp)
        wins = sum(1 for xv, np_ in zip(c_xval, c_nonp) if xv > np_)
        print(f"  {c:<20}  {mx:6.4f}  {ms:6.4f}  {mn:6.4f}  "
              f"{mx - mn:+9.4f}  {wins}/{len(c_xval)}")

    print()
    if xval_scores and non_peak_m:
        overall_delta = np.mean(xval_scores) - np.mean(non_peak_m)
        verdict = (
            "CONFIRMED — xval peak outperforms non-peak. Peak advantage is not "
            "an artefact of circular evaluation."
            if overall_delta > 0.02
            else (
                "MARGINAL — small xval peak advantage. Circularity may inflate "
                "results slightly."
                if overall_delta > 0
                else "INCONCLUSIVE — xval peak does not clearly outperform non-peak."
            )
        )
        print(f"  Verdict: {verdict}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim", type=int, default=None,
        help="Restrict to models with this hidden_dim (e.g. 768 for Cluster A)",
    )
    parser.add_argument(
        "--concepts", nargs="+", default=None, metavar="CONCEPT",
        help="Only evaluate these concepts (default: all 17). Useful for fast dev runs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--no-layer-scores", action="store_true",
        help="Omit per-layer scores from output JSON (smaller file)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dim_tag = f"_dim{args.dim}" if args.dim else ""

    # Collect same-dim model pairs that have alllayer files
    dim_groups = model_dirs_by_dim()
    if args.dim:
        dim_groups = {k: v for k, v in dim_groups.items() if k == args.dim}

    all_pairs: list[tuple[str, Path, str, Path]] = []
    for dim, models in sorted(dim_groups.items()):
        if len(models) < 2:
            continue
        # Only keep models that have at least one alllayer file
        models_with_alllayer = [
            (mid, d) for mid, d in models
            if any(d.glob("calibration_alllayer_*.npy"))
        ]
        if len(models_with_alllayer) < 2:
            log.info("dim=%d: fewer than 2 models with alllayer files — skipping", dim)
            continue
        log.info("dim=%d: %d models with alllayer files → %d pairs",
                 dim, len(models_with_alllayer),
                 len(models_with_alllayer) * (len(models_with_alllayer) - 1) // 2)
        for (src_mid, src_dir), (tgt_mid, tgt_dir) in combinations(models_with_alllayer, 2):
            all_pairs.append((src_mid, src_dir, tgt_mid, tgt_dir))

    if not all_pairs:
        log.error(
            "No eligible pairs found.  Make sure calibration_alllayer_*.npy "
            "files are present.\n"
            "Download Cluster A (768-dim) files with:\n"
            "  hf download james-ra-henry/Rosetta-Activations \\\n"
            "    --repo-type dataset --local-dir ~/rosetta_data/models/ \\\n"
            "    --include 'models/EleutherAI_gpt_neo_125m/calibration_alllayer_*' \\\n"
            "    --include 'models/EleutherAI_pythia_160m/calibration_alllayer_*' \\\n"
            "    --include 'models/openai_community_gpt2/calibration_alllayer_*' \\\n"
            "    --include 'models/facebook_opt_125m/calibration_alllayer_*'"
        )
        return

    log.info("Total pairs for xval: %d", len(all_pairs))

    concepts_to_run = args.concepts if args.concepts else CONCEPTS
    if args.concepts:
        unknown = set(args.concepts) - set(CONCEPTS)
        if unknown:
            log.warning("Unknown concepts (will be skipped): %s", ", ".join(sorted(unknown)))
        log.info("Evaluating %d concept(s): %s", len(concepts_to_run), ", ".join(concepts_to_run))

    rows = run_xval(all_pairs, concepts=concepts_to_run)

    if args.no_layer_scores:
        for r in rows:
            r.pop("layer_scores", None)

    out_path = args.out_dir / f"peak_vs_nonpeak_xval{dim_tag}.json"
    out_path.write_text(json.dumps(rows, indent=None, separators=(",", ":")))
    log.info("Wrote %d rows → %s", len(rows), out_path)

    summarize(rows)


if __name__ == "__main__":
    main()
