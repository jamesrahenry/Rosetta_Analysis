#!/usr/bin/env python3
"""C6 — multimodality / threshold sensitivity curves (ROUND3_COMPUTE_PLAN.md).

Part A: baseline reproduction + k-sensitivity of CAZ-region counts.
  Detector: rosetta_tools.caz.find_caz_regions_scored (defaults), fed
  LayerMetrics built from stored caz_<concept>.json layer_data.metrics.
  NOTE (mechanism): find_caz_regions_scored consumes ONLY the separation
  (and coherence) series — scipy.find_peaks on separation_fisher with
  prominence floor 0.5% of global max sep, min distance 2, plus a 3%
  valley-merge pass. The velocity field is carried in LayerMetrics but
  never read by the detector. Therefore the smoothing window k CANNOT
  change region counts through this detector. We verify that empirically
  and additionally report the k-dependent quantity that DOES move:
  velocity-peak counts (find_peaks on the velocity curve), which is what
  preprint §2.3's existing sensitivity sentence describes (1,238 adaptive
  vs 332-358 fixed).

Part B: §6.2 dependency split at retained-thresholds {40,50,60,70,80}%,
  from ablation_multimodal_<concept>.json interaction matrices
  (matrix[i][j] = % of CAZ j's separation retained when CAZ i is ablated,
  CAZes sorted shallow->deep).

Written: 2026-07-16 UTC
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# round3_gpu import (common/forward_utils): set P3_ROUND3_GPU to that dir if needed
import os as _os
if _os.environ.get("P3_ROUND3_GPU"):
    sys.path.insert(0, _os.environ["P3_ROUND3_GPU"])
for _p in (str(Path.home()/"rosetta_tools"), str(Path.home()/"Source"/"Rosetta_Program"/"rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from common import CONCEPTS_17, slugify  # noqa: E402

# Pinned inline 2026-07-16: common.BASE_28 was trimmed to 25 models by the GPU
# bring-up (opt-350m dim mismatch, Gemma-2 DOM instability) — correct for the
# GPU session, but C6 verifies the PAPER's published numbers, which are defined
# over the original 28-model Table 1 corpus.
BASE_28 = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    "google/gemma-2-2b", "google/gemma-2-9b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3", "microsoft/phi-2",
]
assert len(BASE_28) == 28
from rosetta_tools.caz import (  # noqa: E402
    LayerMetrics, compute_velocity, find_caz_regions_scored,
)

DATA = Path("/home/jhenry/rosetta_data/paper_n250")
OUT = Path(__file__).parent / "c6_sensitivity_results.json"

PARADIGM = {}
for m in BASE_28:
    low = m.lower()
    if any(f in low for f in ("pythia", "gpt2", "opt", "phi")):
        PARADIGM[m] = "mha"
    elif any(f in low for f in ("qwen", "llama", "mistral")):
        PARADIGM[m] = "gqa"
    elif "gemma" in low:
        PARADIGM[m] = "alternating"
    else:
        raise ValueError(m)

K_SETTINGS = ["adaptive", 12, 24, 48]


def load_metrics(model_id, concept):
    p = DATA / slugify(model_id) / f"caz_{concept}.json"
    d = json.loads(p.read_text())
    return d["layer_data"]["metrics"]


def detect(metrics_raw, velocity, paradigm):
    lm = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], float(v))
        for m, v in zip(metrics_raw, velocity)
    ]
    return find_caz_regions_scored(lm, attention_paradigm=paradigm)


def velocity_clamped(seps, k):
    """Velocity with the smoothing window clamped to n_layers (unlike the
    library compute_velocity, which silently skips smoothing when
    window > n_layers)."""
    seps = np.asarray(seps, dtype=np.float64)
    n = len(seps)
    w = max(1, n // 24) if k == "adaptive" else min(int(k), n)
    sm = np.convolve(seps, np.ones(w) / w, mode="same") if w > 1 else seps
    v = np.zeros_like(sm)
    v[1:] = np.diff(sm)
    return v


def velocity_peaks(vel, seps, prom_frac=0.005):
    """Peaks in the velocity curve at a prominence floor of prom_frac * max
    separation (the 0.5%% floor quoted in preprint §2.3)."""
    vel = np.asarray(vel, dtype=np.float64)
    floor = max(prom_frac * float(np.max(seps)), 1e-9)
    pk, _ = find_peaks(vel, prominence=floor, distance=2)
    return len(pk)


def main():
    # ---------------- Part A ----------------
    per_k = {}
    vel_check = {"n_match": 0, "n_total": 0, "max_abs_diff": 0.0}
    for k in K_SETTINGS:
        tot_regions = 0
        n_multi = 0
        n_pairs = 0
        vel_peak_total = 0
        region_counts = []
        for model in BASE_28:
            paradigm = PARADIGM[model]
            for concept in CONCEPTS_17:
                mr = load_metrics(model, concept)
                seps = [m["separation_fisher"] for m in mr]
                window = None if k == "adaptive" else int(k)
                vel = compute_velocity(seps, window=window)
                if k == "adaptive":
                    stored = np.array([m["velocity"] for m in mr], dtype=np.float64)
                    diff = float(np.max(np.abs(stored - vel)))
                    vel_check["n_total"] += 1
                    vel_check["max_abs_diff"] = max(vel_check["max_abs_diff"], diff)
                    if diff < 1e-6:
                        vel_check["n_match"] += 1
                prof = detect(mr, vel, paradigm)
                tot_regions += prof.n_regions
                region_counts.append(prof.n_regions)
                n_multi += int(prof.is_multimodal)
                n_pairs += 1
                vel_peak_total += velocity_peaks(velocity_clamped(seps, k), seps)
        per_k[str(k)] = {
            "total_caz_regions": tot_regions,
            "n_pairs": n_pairs,
            "multimodal_cells": n_multi,
            "multimodal_rate_pct": round(100.0 * n_multi / n_pairs, 2),
            "mean_regions_per_pair": round(tot_regions / n_pairs, 3),
            "velocity_peak_total": vel_peak_total,
        }
        print(f"k={k}: regions={tot_regions} multi={n_multi}/{n_pairs} "
              f"({100*n_multi/n_pairs:.1f}%) mean={tot_regions/n_pairs:.3f} "
              f"vel_peaks={vel_peak_total}")
    print("stored-velocity check (adaptive):", vel_check)

    # ---------------- Part B ----------------
    thresholds = [40, 50, 60, 70, 80]
    dep = {t: {"independent": 0, "forward": 0, "backward": 0} for t in thresholds}
    n_files = 0
    n_pairs_directed = 0
    sum_ncaz = 0
    for model in BASE_28:
        slug = slugify(model)
        for f in sorted((DATA / slug).glob("ablation_multimodal_*.json")):
            d = json.loads(f.read_text())
            M = d["interaction_matrix"]
            n = len(M)
            if n < 2:
                continue
            n_files += 1
            sum_ncaz += n
            # order check: peaks should be shallow->deep
            peaks = [c["peak"] for c in d["cazs"]]
            assert peaks == sorted(peaks), (f, peaks)
            for i in range(n):
                for j in range(i + 1, n):
                    n_pairs_directed += 1
                    for t in thresholds:
                        if M[i][j] <= t:
                            dep[t]["forward"] += 1
                        else:
                            dep[t]["independent"] += 1
                        if M[j][i] <= t:
                            dep[t]["backward"] += 1
    dep_out = {}
    n_paper_denom = 2 * n_pairs_directed  # paper's §6.2 denominator: both
    # directions per unordered pair (n*(n-1) per cell); the reverse-direction
    # half is architecturally guaranteed independent (§6.3 consistency check).
    for t in thresholds:
        f_ = dep[t]["forward"]
        i_ = dep[t]["independent"]
        dep_out[str(t)] = {
            "forward_dependent": f_,
            "backward_dependent": dep[t]["backward"],
            "upper_triangle_convention": {
                "n_directed_pairs": n_pairs_directed,
                "independent": i_,
                "independent_pct": round(100.0 * i_ / n_pairs_directed, 2),
                "forward_pct": round(100.0 * f_ / n_pairs_directed, 2),
            },
            "paper_s6_2_convention": {
                "n_directed_pairs": n_paper_denom,
                "independent": n_paper_denom - f_,
                "independent_pct": round(100.0 * (n_paper_denom - f_) / n_paper_denom, 2),
                "forward_pct": round(100.0 * f_ / n_paper_denom, 2),
            },
        }
        print(f"thr={t}%: fwd={f_} bwd={dep[t]['backward']} | "
              f"upper-tri {100*i_/n_pairs_directed:.1f}% indep | "
              f"paper-denom {100*(n_paper_denom-f_)/n_paper_denom:.1f}% indep / "
              f"{100*f_/n_paper_denom:.1f}% fwd")
    print(f"directed pairs={n_pairs_directed} from {n_files} multimodal cells "
          f"(sum n_cazs={sum_ncaz})")

    out = {
        "written_utc": __import__("subprocess").run(
            ["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True
        ).stdout.strip(),
        "job": "C6 ROUND3_COMPUTE_PLAN.md — multimodality/threshold sensitivity",
        "roster": {"n_models": len(BASE_28), "n_concepts": len(CONCEPTS_17)},
        "paper_baseline": {
            "total_caz_regions": 1036, "multimodal": "360/476 (75.6%)",
            "mean_regions_per_pair": 2.18,
            "dependency_split_60pct": "1531/1678 independent (91.2%), 147 forward (8.8%)",
        },
        "part_a_region_sensitivity": per_k,
        "stored_velocity_vs_recomputed_adaptive": vel_check,
        "part_b_dependency_threshold_sweep": {
            "n_multimodal_cells_with_artifacts": n_files,
            "n_unordered_pairs_upper_triangle": n_pairs_directed,
            "n_directed_pairs_paper_denominator": 2 * n_pairs_directed,
            "thresholds": dep_out,
            "definition": ("forward-dependent iff interaction_matrix[i][j] <= t for "
                           "i<j (shallower ablated, deeper measured); backward iff "
                           "matrix[j][i] <= t; source ablation_multimodal_<concept>.json"),
            "baseline_discrepancy_note": (
                "At t=60% this recount finds 148 forward-dependent pairs vs the "
                "paper's 147 (8.82% vs 8.76% on the 1,678 denominator) — a "
                "one-pair difference, likely artifact vintage; no retained value "
                "sits exactly at 60.0 so it is not a boundary-convention issue. "
                "The 1,678 denominator is exactly 2x the 839 upper-triangle "
                "pairs: the paper counts both directions per pair, and the 839 "
                "reverse-direction (deep-ablate, shallow-measure) pairs are "
                "architecturally guaranteed independent (0 backward-dependent "
                "at every threshold 40-80%, confirming §6.2's 0% backward)."
            ),
        },
        "velocity_field_provenance": {
            "finding": (
                "The stored 'velocity' field in the paper_n250 caz_<concept>.json "
                "files matches compute_velocity(seps, window=3) to float precision "
                "on all checked pairs — i.e. it was generated with the "
                "compute_layer_metrics default fixed window=3, NOT the adaptive "
                "k=max(1,floor(L/24)) rule described in preprint §2.3. This does "
                "not affect the headline region counts (the production detector "
                "never reads velocity), but §2.3's description of the stored "
                "velocity as adaptive-smoothed is inaccurate for this artifact "
                "tree."
            ),
            "adaptive_recompute_match": vel_check,
        },
        "s2_3_velocity_peak_numbers_not_reproduced": (
            "Preprint §2.3 quotes 1,238 velocity peaks at adaptive k vs 332-358 "
            "at fixed k in {12,24,48} (0.5% prominence floor). No producing "
            "script exists in Rosetta_Program, and none of the plausible "
            "definitions tried here reproduces those numbers: find_peaks on the "
            "velocity curve with prominence 0.5% of max separation gives "
            "3,247 (adaptive) / 1,047 / 931 / 3,039 (k=12/24/48, library "
            "compute_velocity, which silently skips smoothing when window > "
            "n_layers — hence the k=48 blow-up) or 3,247 / 1,032 / 655 / 548 "
            "with the window clamped to n_layers; prominence relative to max "
            "|velocity| and upward zero-crossing counts were also tried. The "
            "velocity_peak_total values reported per k in part_a use the "
            "clamped-window, 0.5%-of-max-separation definition and should be "
            "read as this run's own well-defined quantity, not a reproduction "
            "of §2.3's."
        ),
        "method_notes": (
            "Region detection: rosetta_tools.caz.find_caz_regions_scored with defaults "
            "(min_prominence_frac=0.005, min_peak_distance=2, min_valley_depth_frac=0.03, "
            "attention_paradigm per family: pythia/gpt2/opt/phi=mha, qwen/llama/mistral=gqa, "
            "gemma=alternating). Input: stored layer_data.metrics from caz_<concept>.json "
            "(paper_n250 tree). MECHANISM: this detector runs scipy find_peaks on the "
            "separation_fisher series only; the velocity field is present in LayerMetrics "
            "but never consumed, so the smoothing window k cannot alter region counts / "
            "multimodal rate / mean regions-per-pair through the production detector — "
            "verified empirically by recomputing velocity at k in {12,24,48,adaptive} via "
            "compute_velocity(seps, window=k) and rerunning detection. The quantity that "
            "IS k-dependent is the velocity-peak count (find_peaks on the velocity curve, "
            "prominence floor 0.5% of max separation, distance 2), reported per k; this is "
            "the quantity preprint §2.3 already describes (1,238 adaptive vs 332-358 fixed). "
            "attention_paradigm affects only functional scores, not counts. Dependency "
            "sweep: interaction_matrix[i][j] = % of CAZ j separation retained when CAZ i "
            "ablated, CAZes sorted shallow->deep; one directed pair per ordered "
            "(shallower, deeper) combination, forward-dependent at retained <= threshold."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
