"""
relay_funnel_null.py — Permutation null for the §7 relay-features screen.

Paper 2 §7 reports a screening funnel:
    414 persistent features -> 39 multi-concept alignment -> 12 ablation-confirmed

A reviewer flagged that the 414 -> 39 step has no null. This script provides
one, using only data already on disk (no GPU required).

Notes on reproducibility of the "39" number
-------------------------------------------
The paper's reported intermediate count (39) does not reproduce from the
current on-disk feature_map.json data under any rule we can identify:

    Rule A (labeled: any concept has max|cos| >= 0.5)           -> 42
    Rule B (>= 2 concepts have max|cos| >= 0.5 anywhere)        -> 17
    Rule C (>= 2 concepts with |cos| >= 0.5 at some layer)      -> 17
    Rule D (>= 2 different concepts dominate at different       -> 12
            layers, paper's stated "different depths" rule)

The 12 at Rule D matches the "12 ablation-confirmed" figure without any
ablation filter, suggesting the ablation filter was vacuous for Rule D or
that the paper's funnel lost 0 features at the ablation step. Either way,
the "39" number appears to be stale; the reproducible pre-ablation screen
counts are 17 (loose) or 12 (strict). We run the null against all four
rules and let the reader judge.

The null
--------
Two permutation nulls, both using only on-disk data:

 1. Within-feature concept-label shuffle. Per feature, relabel which
    concept each trajectory belongs to. Tests: is the feature-concept
    binding informative, or are 7 generic trajectories in each feature
    enough to pass the screen?

 2. Across-feature trajectory swap. For each feature/concept cell,
    replace the trajectory with a random other feature's trajectory for
    the SAME concept. Destroys the within-feature coherence that
    defines "one feature, multiple concepts" while preserving each
    concept's marginal cosine distribution.

Outputs
-------
results/relay_null/relay_funnel_null.json - observed counts + null
stats + empirical p-values for all four rules under both nulls.
results/relay_null/relay_funnel_null.png  - histograms w/ observed line.

Usage
-----
    python src/relay_funnel_null.py
    python src/relay_funnel_null.py --n-shuffles 10000 --seed 42

Written: 2026-04-18 UTC.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CAZ_ROOT = Path(__file__).resolve().parents[1]
RESULTS  = CAZ_ROOT / "results"
OUT_DIR  = RESULTS / "relay_null"

DARK_MATTER_MODELS = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",  "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",   "Qwen/Qwen2.5-7B",
]

COS_THRESHOLD = 0.5
MIN_CONCEPTS  = 2


def find_deepdive(mid: str) -> Path | None:
    slug   = mid.replace("/", "_").replace("-", "_")
    prefix = f"deepdive_{slug}_"
    matches = sorted(
        (d for d in RESULTS.iterdir()
         if d.name.startswith(prefix) and d.name[len(prefix):len(prefix) + 8].isdigit()),
        reverse=True,
    )
    return matches[0] if matches else None


def load_persistent_features() -> list[dict]:
    features = []
    for mid in DARK_MATTER_MODELS:
        d = find_deepdive(mid)
        if d is None:
            log.warning("No deepdive for %s", mid)
            continue
        fm = json.loads((d / "feature_map.json").read_text())
        for f in fm.get("features", []):
            if not f.get("is_persistent", False):
                continue
            traj_in = f.get("concept_alignment_trajectory", {})
            if not traj_in:
                continue
            traj = {concept: {int(L): float(v) for L, v in by.items()}
                    for concept, by in traj_in.items()}
            features.append({
                "model_id":   mid,
                "feature_id": f.get("feature_id"),
                "trajectory": traj,
                "max_abs":    {c: max((abs(v) for v in t.values()), default=0.0)
                               for c, t in traj.items()},
            })
    log.info("Loaded %d persistent features across %d models",
             len(features), len({f["model_id"] for f in features}))
    return features


# ---- screens -----------------------------------------------------------

def screen_a(f):
    return any(v >= COS_THRESHOLD for v in f["max_abs"].values())

def screen_b(f):
    return sum(v >= COS_THRESHOLD for v in f["max_abs"].values()) >= MIN_CONCEPTS

def screen_c(f):
    hits = {c for c, t in f["trajectory"].items()
            if any(abs(v) >= COS_THRESHOLD for v in t.values())}
    return len(hits) >= MIN_CONCEPTS

def screen_d(f):
    cb = defaultdict(list)
    for c, t in f["trajectory"].items():
        for L, v in t.items():
            if abs(v) >= COS_THRESHOLD:
                cb[L].append(c)
    layer_dom = {L: cs[0] for L, cs in sorted(cb.items()) if cs}
    return len(set(layer_dom.values())) >= MIN_CONCEPTS

SCREENS = [("A_any_labeled",   screen_a),
           ("B_2plus_maxabs",  screen_b),
           ("C_2plus_anywhere",screen_c),
           ("D_diff_layers",   screen_d)]


def counts(features: list[dict]) -> dict[str, int]:
    return {name: sum(s(f) for f in features) for name, s in SCREENS}


# ---- nulls -------------------------------------------------------------

def _rebuild_max_abs(f):
    f["max_abs"] = {c: max((abs(v) for v in t.values()), default=0.0)
                    for c, t in f["trajectory"].items()}

def shuffle_within_feature(features, rng):
    out = []
    for f in features:
        concepts     = list(f["trajectory"].keys())
        trajectories = list(f["trajectory"].values())
        perm = rng.permutation(len(concepts))
        new_traj = {concepts[i]: trajectories[perm[i]] for i in range(len(concepts))}
        nf = {"model_id": f["model_id"], "feature_id": f["feature_id"],
              "trajectory": new_traj}
        _rebuild_max_abs(nf)
        out.append(nf)
    return out

def shuffle_across_feature(features, rng):
    """For each concept, permute which feature each trajectory belongs to."""
    concepts = sorted({c for f in features for c in f["trajectory"]})
    # Pool trajectories per concept
    pool = {c: [f["trajectory"].get(c, {}) for f in features] for c in concepts}
    perm = {c: rng.permutation(len(features)) for c in concepts}
    out = []
    for i, f in enumerate(features):
        new_traj = {c: pool[c][perm[c][i]] for c in concepts if pool[c][perm[c][i]]}
        nf = {"model_id": f["model_id"], "feature_id": f["feature_id"],
              "trajectory": new_traj}
        _rebuild_max_abs(nf)
        out.append(nf)
    return out


def null_distribution(features, shuffler, n_shuffles, seed):
    rng = np.random.default_rng(seed)
    counts_by_rule = {name: np.empty(n_shuffles, dtype=np.int32) for name, _ in SCREENS}
    for i in range(n_shuffles):
        if i and i % 200 == 0:
            log.info("  %s shuffle %d/%d", shuffler.__name__, i, n_shuffles)
        permuted = shuffler(features, rng)
        c = counts(permuted)
        for name in counts_by_rule:
            counts_by_rule[name][i] = c[name]
    return counts_by_rule


def emp_p(obs, null):
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


# ---- plot --------------------------------------------------------------

def plot_all(obs: dict, null_within: dict, null_across: dict, out_path: Path):
    fig, axes = plt.subplots(len(SCREENS), 2, figsize=(11, 2.8 * len(SCREENS)))
    fig.patch.set_facecolor("white")
    for row, (name, _) in enumerate(SCREENS):
        for col, (null, label) in enumerate([
            (null_within[name], "Within-feature concept-label shuffle"),
            (null_across[name], "Across-feature concept trajectory swap"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("white")
            bins = max(8, int(null.max() - null.min() + 1))
            ax.hist(null, bins=bins, color="#9E9E9E", alpha=0.7,
                    edgecolor="white", zorder=2)
            ax.axvline(obs[name], color="#C62828", linewidth=2.2, zorder=3,
                       label=f"Obs = {obs[name]}")
            p = emp_p(obs[name], null)
            ax.set_title(f"Rule {name}  |  {label}\nempirical p = {p:.3f}",
                         fontsize=9.5, fontweight="bold", color="#212121", loc="left")
            ax.set_xlabel("Features passing screen (null)", fontsize=8.5)
            ax.set_ylabel("Null draws", fontsize=8.5)
            ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=1)
            for s in ax.spines.values(): s.set_edgecolor("#BDBDBD")
            ax.legend(fontsize=8, facecolor="white", edgecolor="#BDBDBD")
    fig.suptitle(
        "Paper 2 §7 relay-feature screen — permutation null across 4 rules × 2 nulls",
        color="#212121", fontsize=12, fontweight="bold", y=1.01, va="bottom")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_persistent_features()
    if not features:
        log.error("No persistent features found."); return

    obs = counts(features)
    log.info("Observed counts: %s", obs)

    log.info("Running %d within-feature shuffles...", args.n_shuffles)
    null_within = null_distribution(features, shuffle_within_feature,
                                    args.n_shuffles, args.seed)
    log.info("Running %d across-feature shuffles...", args.n_shuffles)
    null_across = null_distribution(features, shuffle_across_feature,
                                    args.n_shuffles, args.seed + 1)

    result = {
        "n_persistent_features": len(features),
        "n_models":              len({f["model_id"] for f in features}),
        "n_shuffles":            args.n_shuffles,
        "cos_threshold":         COS_THRESHOLD,
        "min_concepts":          MIN_CONCEPTS,
        "rules": {
            name: {
                "observed": obs[name],
                "within_feature_null": {
                    "mean":  float(null_within[name].mean()),
                    "std":   float(null_within[name].std()),
                    "p95":   float(np.percentile(null_within[name], 95)),
                    "max":   int(null_within[name].max()),
                    "emp_p": emp_p(obs[name], null_within[name]),
                },
                "across_feature_null": {
                    "mean":  float(null_across[name].mean()),
                    "std":   float(null_across[name].std()),
                    "p95":   float(np.percentile(null_across[name], 95)),
                    "max":   int(null_across[name].max()),
                    "emp_p": emp_p(obs[name], null_across[name]),
                },
            } for name, _ in SCREENS
        },
    }

    out_json = OUT_DIR / "relay_funnel_null.json"
    out_json.write_text(json.dumps(result, indent=2))
    log.info("Saved %s", out_json)

    plot_all(obs, null_within, null_across, OUT_DIR / "relay_funnel_null.png")
    log.info("Saved %s", OUT_DIR / "relay_funnel_null.png")

    print()
    print("=" * 78)
    print("RELAY-FEATURE SCREEN — PERMUTATION NULL (414 persistent, 20 models)")
    print("=" * 78)
    print(f"{'Rule':<22} {'Obs':>5}  {'WF null':>15}  {'WF p':>6}  {'AF null':>15}  {'AF p':>6}")
    for name, _ in SCREENS:
        w = result["rules"][name]["within_feature_null"]
        a = result["rules"][name]["across_feature_null"]
        print(f"{name:<22} {obs[name]:>5}  "
              f"{w['mean']:>7.2f} ± {w['std']:<4.2f}  {w['emp_p']:>6.3f}  "
              f"{a['mean']:>7.2f} ± {a['std']:<4.2f}  {a['emp_p']:>6.3f}")
    print()
    print("WF = within-feature concept-label shuffle (tests feature-concept binding)")
    print("AF = across-feature concept trajectory swap (tests within-feature coherence)")
    print()


if __name__ == "__main__":
    main()
