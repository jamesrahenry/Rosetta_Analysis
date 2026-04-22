"""
ablate_random_layer_null.py
===========================
Reviewer-response null: matched random-layer control for the 3.67× enrichment claim.

The concern: ablate_noncaz_control.py compares CAZ-peak layers to "non-CAZ
layers" — layers the detector *deliberately excluded*. This introduces
selection bias: the comparison pool is precisely the layers expected to be
weakest.

This script addresses it by sampling N layers uniformly at random from the
*full* network depth (N = number of CAZ peaks detected for that model/concept),
repeating K=10,000 times to build a null distribution, then comparing observed
CAZ-peak mean reduction to that null. If the 3.67× enrichment is real, it
survives here too; if it was an artifact of pool construction, it won't.

Output
------
results/random_layer_null/
  random_layer_null_summary.txt       — human-readable table
  random_layer_null.json              — per-model/concept records
  random_layer_null_distribution.png  — observed vs null density plot

Usage
-----
    cd ~/caz_scaling
    python src/ablate_random_layer_null.py
    python src/ablate_random_layer_null.py --n-permutations 50000
    python src/ablate_random_layer_null.py --concepts sentiment credibility
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import FAMILY_MAP, THEME, CONCEPTS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CAZ_ROOT = Path(__file__).resolve().parents[1]
RESULTS  = CAZ_ROOT / "results"
OUT_DIR  = RESULTS / "random_layer_null"

NON_CAZ_WINDOW = 3   # must match ablate_noncaz_control.py
N_PERMS_DEFAULT = 10_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_sweeps() -> list[dict]:
    """Load all ablation_global_sweep_*.json files, one record per model/concept.

    When a model has multiple extraction runs (duplicate dirs), keep only the
    most recently modified file for each (model_id, concept) pair.
    """
    # Collect all candidates, keyed by (model_id, concept) -> (mtime, path)
    best: dict[tuple[str, str], tuple[float, Path]] = {}
    for f in RESULTS.rglob("ablation_global_sweep_*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        key = (d.get("model_id", "unknown"), d.get("concept", "unknown"))
        mtime = f.stat().st_mtime
        if key not in best or mtime > best[key][0]:
            best[key] = (mtime, f)

    records = []
    for (model_id, concept), (_, f) in sorted(best.items()):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        layers   = d.get("layers", [])
        caz_peak = d.get("caz_peak")
        if caz_peak is None or not layers:
            continue
        reductions = {l["layer"]: l["global_sep_reduction"] for l in layers}
        n_layers = d.get("n_layers", len(layers))

        # All CAZ peak layers for this model/concept from caz_*.json
        caz_file = f.parent / f"caz_{concept}.json"
        caz_peaks = _get_all_peaks(caz_file, caz_peak)

        # Instruct / family
        is_instruct = any(t in model_id for t in ["Instruct", "instruct", "-it"])
        family = _family(model_id)

        records.append({
            "model_id":    model_id,
            "concept":     concept,
            "family":      family,
            "is_instruct": is_instruct,
            "n_layers":    n_layers,
            "caz_peaks":   caz_peaks,
            "reductions":  reductions,   # {layer_idx: reduction}
        })
    return records


def _get_all_peaks(caz_path: Path, fallback_peak: int) -> list[int]:
    if not caz_path.exists():
        return [fallback_peak]
    try:
        raw = json.loads(caz_path.read_text())
    except (json.JSONDecodeError, OSError):
        return [fallback_peak]
    ld = raw.get("layer_data", raw)
    metrics = ld.get("metrics", [])
    seps = [m.get("separation_fisher", 0.0) for m in metrics]
    if not seps:
        return [fallback_peak]
    peaks = [
        i for i in range(1, len(seps) - 1)
        if seps[i] >= seps[i - 1] and seps[i] >= seps[i + 1]
    ]
    return peaks if peaks else [int(np.argmax(seps))]


def _family(model_id: str) -> str:
    for hf_id, (fam, _) in FAMILY_MAP.items():
        if model_id in hf_id or hf_id.endswith(model_id.split("/")[-1]):
            return fam
    return "Unknown"


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def matched_random_null(
    all_reductions: np.ndarray,
    n_peaks: int,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n_peaks layers uniformly at random from all_reductions, compute
    their mean, repeat n_perms times.  Returns array of null means.
    """
    idx = rng.integers(0, len(all_reductions), size=(n_perms, n_peaks))
    return all_reductions[idx].mean(axis=1)


def run_permutation_test(record: dict, n_perms: int, rng: np.random.Generator) -> dict | None:
    caz_peaks  = record["caz_peaks"]
    reductions = record["reductions"]
    n_layers   = record["n_layers"]

    if not caz_peaks or not reductions:
        return None

    all_reds   = np.array([reductions.get(l, 0.0) for l in range(n_layers)], dtype=np.float64)
    peak_reds  = np.array([reductions.get(p, 0.0) for p in caz_peaks], dtype=np.float64)
    observed   = float(peak_reds.mean())

    n = len(caz_peaks)
    null_dist = matched_random_null(all_reds, n, n_perms, rng)

    null_mean  = float(null_dist.mean())
    null_p95   = float(np.percentile(null_dist, 95))
    null_p99   = float(np.percentile(null_dist, 99))
    p_value    = float(np.mean(null_dist >= observed))
    ratio      = observed / max(null_mean, 1e-6)

    # Also compute old-style non-CAZ mean for comparison
    noncaz_reds = [
        reductions[l] for l in range(n_layers)
        if all(abs(l - p) > NON_CAZ_WINDOW for p in caz_peaks)
        and l in reductions
    ]
    old_noncaz_mean = float(np.mean(noncaz_reds)) if noncaz_reds else float("nan")
    old_ratio = observed / max(old_noncaz_mean, 1e-6) if noncaz_reds else float("nan")

    return {
        "model_id":      record["model_id"],
        "concept":       record["concept"],
        "family":        record["family"],
        "is_instruct":   record["is_instruct"],
        "n_layers":      n_layers,
        "n_caz_peaks":   n,
        "observed_mean": round(observed, 4),
        "null_mean":     round(null_mean, 4),
        "null_p95":      round(null_p95, 4),
        "null_p99":      round(null_p99, 4),
        "p_value":       round(p_value, 6),
        "ratio_vs_random": round(ratio, 2),
        "old_noncaz_mean": round(old_noncaz_mean, 4) if not np.isnan(old_noncaz_mean) else None,
        "old_ratio_vs_noncaz": round(old_ratio, 2) if not np.isnan(old_ratio) else None,
        "null_dist": null_dist.tolist(),   # kept for plotting; stripped in summary JSON
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(results: list[dict], out_dir: Path) -> None:
    base = [r for r in results if not r["is_instruct"]]

    # Grand aggregate
    obs_all   = [r["observed_mean"] for r in base]
    null_all  = [r["null_mean"]     for r in base]
    ratio_all = [r["ratio_vs_random"] for r in base]
    old_ratio_all = [r["old_ratio_vs_noncaz"] for r in base if r["old_ratio_vs_noncaz"] is not None]
    sig       = [r for r in base if r["p_value"] < 0.05]

    lines = [
        "Matched random-layer null — CAZ peak enrichment (base models only)",
        f"N records: {len(base)}  |  Significant (p<0.05): {len(sig)}/{len(base)}",
        f"Grand mean observed: {np.mean(obs_all):.4f}  |  Grand mean null: {np.mean(null_all):.4f}",
        f"Grand mean ratio (observed/null): {np.mean(ratio_all):.2f}×",
        f"Old non-CAZ ratio (for comparison): {np.mean(old_ratio_all):.2f}×  "
        f"(N={len(old_ratio_all)})",
        "",
        f"{'Model':<35} {'Concept':<16} {'Obs':>6} {'Null':>6} {'Ratio':>6} {'p':>8} {'OldR':>6}",
        "-" * 90,
    ]

    for r in sorted(base, key=lambda x: (x["model_id"], x["concept"])):
        old_r = f"{r['old_ratio_vs_noncaz']:.2f}×" if r["old_ratio_vs_noncaz"] is not None else "  n/a"
        lines.append(
            f"{r['model_id'].split('/')[-1]:<35} {r['concept']:<16} "
            f"{r['observed_mean']:>6.3f} {r['null_mean']:>6.3f} "
            f"{r['ratio_vs_random']:>5.2f}× {r['p_value']:>8.4f} {old_r:>6}"
        )

    # By concept
    lines += ["", "By concept (base):"]
    by_concept = defaultdict(list)
    for r in base:
        by_concept[r["concept"]].append(r)
    for c in CONCEPTS:
        if c not in by_concept:
            continue
        rs = by_concept[c]
        lines.append(
            f"  {c:<16}  obs={np.mean([r['observed_mean'] for r in rs]):.4f}  "
            f"null={np.mean([r['null_mean'] for r in rs]):.4f}  "
            f"ratio={np.mean([r['ratio_vs_random'] for r in rs]):.2f}×  "
            f"p<0.05: {sum(1 for r in rs if r['p_value']<0.05)}/{len(rs)}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "random_layer_null_summary.txt").write_text(text)
    log.info("Wrote summary to %s", out_dir / "random_layer_null_summary.txt")


def plot_distributions(results: list[dict], out_dir: Path) -> None:
    base = [r for r in results if not r["is_instruct"] and "null_dist" in r]
    if not base:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Aggregate null and observed
    all_null = np.concatenate([r["null_dist"] for r in base])
    all_obs  = np.array([r["observed_mean"] for r in base])

    ax.hist(all_null, bins=80, density=True, color="#9E9E9E", alpha=0.6,
            label="Random N-layer mean (null)", zorder=2)
    for v in all_obs:
        ax.axvline(v, color="#C62828", alpha=0.25, linewidth=0.8, zorder=3)
    ax.axvline(all_obs.mean(), color="#C62828", linewidth=2.0,
               label=f"CAZ peaks (mean={all_obs.mean():.3f})", zorder=4)

    p95 = np.percentile(all_null, 95)
    ax.axvline(p95, color="#FF7043", linewidth=1.2, linestyle="--",
               label=f"Null p95 ({p95:.3f})", zorder=3)

    ax.set_xlabel("Mean separation reduction", color=THEME["text"])
    ax.set_ylabel("Density", color=THEME["text"])
    ax.set_title("CAZ peak enrichment: observed vs. matched random-layer null",
                 color=THEME["text"], fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, facecolor="white", edgecolor=THEME["spine"],
              labelcolor=THEME["text"])
    ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax.tick_params(colors=THEME["dim"])

    fig.tight_layout()
    out_path = out_dir / "random_layer_null_distribution.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--n-permutations", type=int, default=N_PERMS_DEFAULT)
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    records = load_global_sweeps()
    if not records:
        log.error("No global sweep data found. Run ablate_global_sweep.py --all first.")
        sys.exit(1)

    if args.concepts:
        records = [r for r in records if r["concept"] in args.concepts]

    log.info("Loaded %d model/concept records", len(records))

    results = []
    for rec in records:
        r = run_permutation_test(rec, args.n_permutations, rng)
        if r:
            results.append(r)
            log.info("%s / %s  obs=%.3f  null=%.3f  ratio=%.2f×  p=%.4f",
                     rec["model_id"].split("/")[-1], rec["concept"],
                     r["observed_mean"], r["null_mean"],
                     r["ratio_vs_random"], r["p_value"])

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    # Save JSON without null_dist arrays (too large)
    slim = [{k: v for k, v in r.items() if k != "null_dist"} for r in results]
    (OUT_DIR / "random_layer_null.json").write_text(json.dumps(slim, indent=2))

    summarise(results, OUT_DIR)
    plot_distributions(results, OUT_DIR)
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
