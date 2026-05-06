"""
p5_permutation_test.py — P5 depth-stratified alignment cross-validation.

*Written: 2026-04-29 UTC*

Tests whether depth-matched cross-architecture alignment (shallow↔shallow,
deep↔deep) exceeds mismatched alignment (shallow↔deep) in a way that
survives Procrustes cross-validation.

The reviewer concern (CAZ paper §5.5) is that fitting Procrustes on 7
concept directions in d >> 7 space may overfit and inflate matched cosines.
The correct answer is leave-one-concept-out (LOCO) cross-validation:

  For each test concept C:
    Fit Procrustes using all-layer dom_vectors from the OTHER 6 concepts.
    Evaluate SS/DD/SD/DS alignment on concept C.

If depth-matched > mismatched persists after LOCO, the result is not a
Procrustes artifact — the calibration data was genuinely independent.

Also reports:
  - Observed (non-CV) Mann-Whitney p on matched vs mismatched
  - Bootstrap 95% CI on grand mean delta (matched - mismatched)

Output: ~/rosetta_data/results/p5_permutation/p5_cv_results.json

Usage:
    python alignment/p5_permutation_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import mannwhitneyu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_DATA_ROOT_CANDIDATES = [
    Path.home() / "rosetta_data" / "models",                          # GPU host
    Path.home() / "Source" / "Rosetta_Program" / "rosetta_data" / "models",  # local
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if p.exists()), None)
if DATA_ROOT is None:
    sys.exit("rosetta_data/models/ not found. Check path.")

OUT_DIR = DATA_ROOT.parent / "results" / "p5_permutation"

CONCEPTS = [
    "credibility", "certainty", "causation",
    "temporal_order", "negation", "sentiment", "moral_valence",
]
N_BOOTSTRAP = 10_000
RNG_SEED = 42

# ------------------------------------------------------------------
# Local imports — handle both installed and source-tree rosetta_tools
# ------------------------------------------------------------------
try:
    from rosetta_tools.caz import find_caz_regions, LayerMetrics
except ImportError:
    _rt = next(
        (p for p in [
            Path.home() / "rosetta_tools",
            Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools",
        ] if p.exists()),
        None,
    )
    if _rt:
        sys.path.insert(0, str(_rt))
    from rosetta_tools.caz import find_caz_regions, LayerMetrics


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def depth_align_rows(va: np.ndarray, vb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the longer matrix to the shorter one's row count."""
    na, nb = len(va), len(vb)
    n = min(na, nb)
    ia = np.round(np.linspace(0, na - 1, n)).astype(int)
    ib = np.round(np.linspace(0, nb - 1, n)).astype(int)
    return va[ia], vb[ib]


def fit_procrustes(cal_a: np.ndarray, cal_b: np.ndarray) -> np.ndarray | None:
    """Fit Procrustes rotation B → A using calibration matrices.

    Same-dim: orthogonal_procrustes directly (in original space).
    Cross-dim: PCA to shared subspace first.
    Returns R such that (vec_b @ R) is in A's comparison space,
    or None on failure.
    """
    da, db = cal_a.shape[1], cal_b.shape[1]
    a_aligned, b_aligned = depth_align_rows(cal_a, cal_b)

    if da == db:
        try:
            R, _ = orthogonal_procrustes(b_aligned, a_aligned)
            return ("same", R, da)
        except Exception:
            return None
    else:
        from sklearn.decomposition import PCA
        k = min(da, db, len(a_aligned))
        try:
            pca_a = PCA(n_components=k).fit(cal_a)
            pca_b = PCA(n_components=k).fit(cal_b)
            a_proj = pca_a.transform(a_aligned)
            b_proj = pca_b.transform(b_aligned)
            R_sh, _ = orthogonal_procrustes(b_proj, a_proj)
            return ("cross", pca_a, pca_b, R_sh)
        except Exception:
            return None


def apply_rotation(rotation, vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_in_comparison_space, rotated_b_in_comparison_space)."""
    if rotation[0] == "same":
        _, R, _ = rotation
        return vec_a, vec_b @ R
    else:
        _, pca_a, pca_b, R_sh = rotation
        a_p = pca_a.transform(vec_a.reshape(1, -1))[0]
        b_r = pca_b.transform(vec_b.reshape(1, -1))[0] @ R_sh
        return a_p, b_r


def four_cosines(rotation, sh_a, dp_a, sh_b, dp_b) -> dict[str, float]:
    """Compute SS/DD/SD/DS cosines using precomputed projected vectors."""
    sh_a_c, sh_b_c = apply_rotation(rotation, sh_a, sh_b)
    dp_a_c, dp_b_c = apply_rotation(rotation, dp_a, dp_b)
    return {
        "SS": cosine(sh_a_c, sh_b_c),
        "DD": cosine(dp_a_c, dp_b_c),
        "SD": cosine(sh_a_c, dp_b_c),
        "DS": cosine(dp_a_c, sh_b_c),
    }


def delta(cos_dict: dict[str, float]) -> float:
    matched = [v for k, v in cos_dict.items() if k in ("SS", "DD") and not np.isnan(v)]
    mismatched = [v for k, v in cos_dict.items() if k in ("SD", "DS") and not np.isnan(v)]
    if not matched or not mismatched:
        return float("nan")
    return float(np.mean(matched) - np.mean(mismatched))


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_concept_data(model_dir: Path, concept: str) -> dict | None:
    """Load per-layer dom_vectors for a concept. Returns dict or None."""
    caz_path = model_dir / f"caz_{concept}.json"
    if not caz_path.exists():
        return None
    with open(caz_path) as f:
        d = json.load(f)
    ld = d["layer_data"]
    raw = ld["metrics"]
    dom_vecs = np.array([m["dom_vector"] for m in raw], dtype=np.float64)
    norms = np.linalg.norm(dom_vecs, axis=1, keepdims=True)
    dom_vecs /= np.where(norms > 1e-10, norms, 1.0)
    return {
        "hidden_dim": d["hidden_dim"],
        "n_layers": ld["n_layers"],
        "dom_vecs": dom_vecs,
    }


def load_model_all_concepts(model_dir: Path) -> dict[str, dict]:
    """Load dom_vecs for all available concepts in a model directory."""
    result = {}
    for c in CONCEPTS:
        data = load_concept_data(model_dir, c)
        if data is not None:
            result[c] = data
    return result


def find_peaks(model_dir: Path, concept: str) -> tuple[int, int] | None:
    """Return (shallow_peak_layer, deep_peak_layer) or None if not multimodal."""
    caz_path = model_dir / f"caz_{concept}.json"
    if not caz_path.exists():
        return None
    with open(caz_path) as f:
        d = json.load(f)
    ld = d["layer_data"]
    raw = ld["metrics"]
    metrics = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
               for m in raw]
    profile = find_caz_regions(metrics)
    if not profile.is_multimodal:
        return None
    sr = sorted(profile.regions, key=lambda r: r.peak)
    return sr[0].peak, sr[-1].peak


# ------------------------------------------------------------------
# LOCO cross-validation
# ------------------------------------------------------------------

def build_calibration_matrix(model_all_data: dict[str, dict], exclude_concept: str) -> np.ndarray:
    """Concatenate depth-normalized dom_vecs from all concepts except the excluded one.

    Rows are depth-resampled to a common count (min across concepts) and then
    stacked vertically, so the combined matrix has shape (n_concepts × n_common, d).
    """
    arrays = []
    for c, data in model_all_data.items():
        if c == exclude_concept:
            continue
        arrays.append(data["dom_vecs"])

    if not arrays:
        return None

    n_common = min(len(a) for a in arrays)
    resampled = []
    for a in arrays:
        idx = np.round(np.linspace(0, len(a) - 1, n_common)).astype(int)
        resampled.append(a[idx])

    return np.vstack(resampled)  # shape: (n_concepts_used × n_common, d)


def run_loco(model_dirs: list[Path]) -> list[dict]:
    """Run leave-one-concept-out cross-validation for all concept × model pairs."""
    # Preload all concept data for all models
    log.info("Loading all concept data for %d models…", len(model_dirs))
    all_model_data = {}
    for md in model_dirs:
        data = load_model_all_concepts(md)
        if data:
            all_model_data[md.name] = {"path": md, "concepts": data}

    results = []

    for test_concept in CONCEPTS:
        log.info("=== LOCO test concept: %s ===", test_concept)

        # Models that are multimodal for this concept
        multimodal = {}
        for name, info in all_model_data.items():
            if test_concept not in info["concepts"]:
                continue
            peaks = find_peaks(info["path"], test_concept)
            if peaks is None:
                continue
            multimodal[name] = {"peaks": peaks, "info": info}

        if len(multimodal) < 2:
            log.info("  < 2 multimodal models — skip")
            continue

        log.info("  %d multimodal models → %d pairs",
                 len(multimodal), len(list(combinations(multimodal, 2))))

        for name_a, name_b in combinations(multimodal.keys(), 2):
            ma, mb = multimodal[name_a], multimodal[name_b]
            sh_a_idx, dp_a_idx = ma["peaks"]
            sh_b_idx, dp_b_idx = mb["peaks"]

            dom_a = ma["info"]["concepts"][test_concept]["dom_vecs"]
            dom_b = mb["info"]["concepts"][test_concept]["dom_vecs"]
            dim_a = ma["info"]["concepts"][test_concept]["hidden_dim"]
            dim_b = mb["info"]["concepts"][test_concept]["hidden_dim"]

            # Build calibration from OTHER concepts
            cal_a = build_calibration_matrix(ma["info"]["concepts"], test_concept)
            cal_b = build_calibration_matrix(mb["info"]["concepts"], test_concept)
            if cal_a is None or cal_b is None:
                log.warning("  SKIP %s × %s: no calibration data", name_a, name_b)
                continue

            # Fit Procrustes on held-out calibration
            rotation = fit_procrustes(cal_a, cal_b)
            if rotation is None:
                log.warning("  SKIP %s × %s: Procrustes failed", name_a, name_b)
                continue

            # Evaluate on held-out test concept
            sh_a = dom_a[sh_a_idx]
            dp_a = dom_a[dp_a_idx]
            sh_b = dom_b[sh_b_idx]
            dp_b = dom_b[dp_b_idx]

            cos = four_cosines(rotation, sh_a, dp_a, sh_b, dp_b)
            d_obs = delta(cos)

            sig = "***" if d_obs > 0.5 else "** " if d_obs > 0.2 else "*  " if d_obs > 0 else "   "
            log.info("  %s %s × %s: delta=%.3f (SS=%.3f DD=%.3f SD=%.3f DS=%.3f)",
                     sig, name_a[:20], name_b[:20], d_obs,
                     cos["SS"], cos["DD"], cos["SD"], cos["DS"])

            results.append({
                "test_concept": test_concept,
                "model_a": name_a,
                "model_b": name_b,
                "dim_a": dim_a,
                "dim_b": dim_b,
                "same_dim": dim_a == dim_b,
                "obs_delta": d_obs,
                **{f"cos_{k}": v for k, v in cos.items()},
            })

    return results


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

def summarize(results: list[dict], rng: np.random.Generator) -> dict:
    all_matched, all_mismatched = [], []
    obs_deltas = []
    concept_summaries = {}

    for r in results:
        d_obs = r["obs_delta"]
        if not np.isnan(d_obs):
            obs_deltas.append(d_obs)
        for k in ("SS", "DD"):
            v = r.get(f"cos_{k}", float("nan"))
            if not np.isnan(v):
                all_matched.append(v)
        for k in ("SD", "DS"):
            v = r.get(f"cos_{k}", float("nan"))
            if not np.isnan(v):
                all_mismatched.append(v)

    # Per-concept
    for concept in CONCEPTS:
        sub = [r for r in results if r["test_concept"] == concept]
        if not sub:
            continue
        d_sub = [r["obs_delta"] for r in sub if not np.isnan(r["obs_delta"])]
        m_sub = [r[f"cos_{k}"] for r in sub for k in ("SS", "DD") if not np.isnan(r.get(f"cos_{k}", float("nan")))]
        mm_sub = [r[f"cos_{k}"] for r in sub for k in ("SD", "DS") if not np.isnan(r.get(f"cos_{k}", float("nan")))]
        if len(m_sub) >= 4 and len(mm_sub) >= 4:
            _, p_mw = mannwhitneyu(m_sub, mm_sub, alternative="greater")
        else:
            p_mw = float("nan")
        concept_summaries[concept] = {
            "n_pairs": len(sub),
            "mean_delta": float(np.mean(d_sub)) if d_sub else float("nan"),
            "n_positive_delta": sum(1 for d in d_sub if d > 0),
            "mannwhitney_p": float(p_mw),
            "mean_matched": float(np.mean(m_sub)) if m_sub else float("nan"),
            "mean_mismatched": float(np.mean(mm_sub)) if mm_sub else float("nan"),
        }

    # Grand Mann-Whitney
    grand = {}
    if len(all_matched) >= 4 and len(all_mismatched) >= 4:
        stat_mw, p_mw = mannwhitneyu(all_matched, all_mismatched, alternative="greater")
        grand["mannwhitney_p"] = float(p_mw)
        grand["mannwhitney_stat"] = float(stat_mw)

    grand["n_pairs_total"] = len(results)
    grand["n_positive_delta"] = sum(1 for d in obs_deltas if d > 0)
    grand["mean_matched"] = float(np.mean(all_matched)) if all_matched else float("nan")
    grand["mean_mismatched"] = float(np.mean(all_mismatched)) if all_mismatched else float("nan")
    grand["mean_delta"] = float(np.mean(obs_deltas)) if obs_deltas else float("nan")

    # Bootstrap CI on grand mean delta
    if obs_deltas:
        arr = np.array(obs_deltas)
        boot = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(N_BOOTSTRAP)]
        grand["bootstrap_ci_95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        grand["bootstrap_p_gt_zero"] = float(np.mean(np.array(boot) > 0))

    return {"grand": grand, "by_concept": concept_summaries}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    model_dirs = sorted(DATA_ROOT.iterdir())
    log.info("Models found: %d in %s", len(model_dirs), DATA_ROOT)

    results = run_loco(model_dirs)

    summary = summarize(results, rng)

    g = summary["grand"]
    log.info("\n=== GRAND SUMMARY (LOCO cross-validation) ===")
    log.info("  Total pairs: %d  (%d positive delta)",
             g["n_pairs_total"], g["n_positive_delta"])
    log.info("  Grand matched: %.3f   mismatched: %.3f   delta: %.3f",
             g["mean_matched"], g["mean_mismatched"], g["mean_delta"])
    if "mannwhitney_p" in g:
        log.info("  Mann-Whitney p (matched > mismatched): %.4e", g["mannwhitney_p"])
    if "bootstrap_ci_95" in g:
        lo, hi = g["bootstrap_ci_95"]
        log.info("  Bootstrap 95%% CI on mean delta: [%.3f, %.3f]  p>0: %.4f",
                 lo, hi, g["bootstrap_p_gt_zero"])

    log.info("\n  Per-concept:")
    for c, cs in summary["by_concept"].items():
        log.info("    %-16s n=%2d  delta=%.3f  matched=%.3f  mis=%.3f  MW_p=%.4f",
                 c, cs["n_pairs"], cs["mean_delta"],
                 cs["mean_matched"], cs["mean_mismatched"], cs["mannwhitney_p"])

    output = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "method": "leave-one-concept-out cross-validation Procrustes",
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "summary": summary,
        "pair_results": results,
    }

    out_path = OUT_DIR / "p5_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results → %s", out_path)


if __name__ == "__main__":
    main()
