"""
analyze_gem_failures.py
=======================
Diagnostic analysis of anomalous GEM ablation results.

Three models in the GEM sweep show highly anomalous behaviour:
  - gpt2 (124M):       14% handoff wins, mean H-retained = 125.9%
  - gpt2-medium (355M): 43% handoff wins, mean H-retained = 147.3%
  - opt-6.7b (6.7B):    0% handoff wins, mean H-retained = 75.2%

For gpt2 and gpt2-medium, H-retained > 100% means ablating the handoff layer
*increases* concept-diagnostic separation.  This is the opposite of suppression.

This script investigates why by examining:
  1. Node structure: where are the handoff layers, and how do they relate to
     the Fisher separation curve?
  2. Direction cosines: what is the angle between settled_direction and the
     peak dom_vector for these models?  A negative cosine would explain
     why ablating the settled direction *helps* the concept.
  3. Global sweep comparison: does ablating ANY layer in these models
     increase separation, or only the handoff layer?
  4. Scale gradient: how does the anomaly evolve across the GPT-2 family
     (gpt2 → medium → large → xl)?  large and xl show normal behaviour,
     which constrains the hypothesis.
  5. OPT comparison: opt-6.7b vs opt-1.3b / opt-2.7b — same family,
     different outcome.

No GPU required.  All analysis is on existing data.

Usage
-----
    cd ~/caz_scaling
    python src/analyze_gem_failures.py
    python src/analyze_gem_failures.py --models openai-community/gpt2
                                                openai-community/gpt2-medium
                                                facebook/opt-6.7b

Written: 2026-04-21 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from rosetta_tools.gem import find_extraction_dir
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = ROSETTA_RESULTS / "gem_failure_analysis"

FAILURE_MODELS = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "facebook/opt-6.7b",
]
CONTRAST_MODELS = [
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
]
CONCEPTS = [
    "causation", "certainty", "credibility", "moral_valence",
    "negation", "sentiment", "temporal_order",
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_model(model_id: str) -> dict:
    short = model_id.split("/")[-1]
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return {"model_id": model_id, "error": "no extraction dir"}

    analysis = {
        "model_id": model_id,
        "extraction_dir": str(extraction_dir),
        "concepts": {},
    }

    for concept in CONCEPTS:
        caz_path = extraction_dir / f"caz_{concept}.json"
        gem_path = extraction_dir / f"gem_{concept}.json"
        abl_path = extraction_dir / f"ablation_gem_{concept}.json"
        sweep_path = extraction_dir / f"ablation_global_sweep_{concept}.json"

        c = {"concept": concept}

        # --- CAZ info ---
        if caz_path.exists():
            caz = json.loads(caz_path.read_text())
            ld = caz.get("layer_data", caz)
            c["caz_peak"] = ld.get("peak_layer")
            metrics = ld.get("metrics", [])
            c["n_layers"] = len(metrics)
            if metrics and c["caz_peak"] is not None:
                c["peak_fisher_sep"] = metrics[c["caz_peak"]].get("separation_fisher")

        # --- GEM node structure ---
        if gem_path.exists():
            gem = json.loads(gem_path.read_text())
            c["n_nodes"] = gem.get("n_nodes", 0)
            c["node_types"] = gem.get("node_types", [])
            c["ablation_targets"] = gem.get("ablation_targets", [])
            nodes_info = []
            for i, node in enumerate(gem.get("nodes", [])):
                settled = np.array(node.get("settled_direction", []), dtype=np.float64)
                caz_peak = c.get("caz_peak")

                # Cosine between settled_direction and peak dom_vector
                dom_cos = None
                if caz_path.exists() and caz_peak is not None:
                    caz_d = json.loads(caz_path.read_text())
                    metrics = caz_d.get("layer_data", caz_d).get("metrics", [])
                    if caz_peak < len(metrics):
                        dom_vec = np.array(
                            metrics[caz_peak].get("dom_vector", []), dtype=np.float64)
                        if settled.size > 0 and dom_vec.size > 0:
                            settled_n = settled / (np.linalg.norm(settled) + 1e-12)
                            dom_n = dom_vec / (np.linalg.norm(dom_vec) + 1e-12)
                            dom_cos = round(float(np.dot(settled_n, dom_n)), 4)

                nodes_info.append({
                    "node_idx": i,
                    "caz_start": node.get("caz_start"),
                    "caz_peak": node.get("caz_peak"),
                    "caz_end": node.get("caz_end"),
                    "handoff_layer": node.get("handoff_layer"),
                    "handoff_cosine": node.get("handoff_cosine"),
                    "settled_vs_peak_dom_cos": dom_cos,
                    "is_target": i in gem.get("ablation_targets", []),
                })
            c["nodes"] = nodes_info

        # --- GEM ablation result ---
        if abl_path.exists():
            abl = json.loads(abl_path.read_text())
            comp = abl.get("comparison", {})
            c["handoff_retained_pct"] = comp.get("handoff_retained_pct")
            c["peak_retained_pct"] = comp.get("peak_retained_pct")
            c["handoff_better"] = comp.get("handoff_better")
            c["handoff_final_sep_reduction"] = abl.get("handoff", {}).get(
                "final_sep_reduction")
            c["peak_final_sep_reduction"] = abl.get("peak", {}).get(
                "final_sep_reduction")
            c["handoff_ablation_layers"] = abl.get("handoff", {}).get(
                "ablation_layers", [])

        # --- Global sweep: are ANY layers amplifying? ---
        if sweep_path.exists():
            sweep = json.loads(sweep_path.read_text())
            layers_data = sweep.get("layers", [])
            # Find layers with negative reduction (amplification)
            amplifying = [l for l in layers_data
                          if l.get("global_sep_reduction", 0) < -0.02]
            c["n_amplifying_layers"] = len(amplifying)
            c["amplifying_layer_indices"] = [l["layer"] for l in amplifying[:10]]
            # Distribution stats
            reds = [l["global_sep_reduction"] for l in layers_data]
            c["sweep_mean_reduction"] = round(float(np.mean(reds)), 4)
            c["sweep_min_reduction"] = round(float(np.min(reds)), 4)
            c["sweep_max_reduction"] = round(float(np.max(reds)), 4)
            # Is the handoff layer in the amplifying set?
            if c.get("handoff_ablation_layers"):
                handoff_idx = c["handoff_ablation_layers"][len(
                    c["handoff_ablation_layers"]) // 2]
                handoff_sweep = next(
                    (l for l in layers_data if l["layer"] == handoff_idx), None)
                c["sweep_reduction_at_handoff"] = (
                    round(handoff_sweep["global_sep_reduction"], 4)
                    if handoff_sweep else None)

        analysis["concepts"][concept] = c

    return analysis


def print_summary(analyses: list[dict]) -> str:
    lines = ["=" * 80,
             "GEM FAILURE MODE ANALYSIS",
             "=" * 80, ""]

    for a in analyses:
        mid = a["model_id"]
        lines.append(f"{'─'*60}")
        lines.append(f"Model: {mid}")
        if "error" in a:
            lines.append(f"  ERROR: {a['error']}")
            continue

        lines.append(f"{'Concept':<16} {'H-ret%':>7} {'P-ret%':>7} "
                     f"{'H>P':>4} {'settled_cos':>11} {'n_amp':>6} "
                     f"{'sweep@H':>8}")
        lines.append("-" * 65)

        for concept in CONCEPTS:
            c = a["concepts"].get(concept, {})
            if not c:
                continue
            h_ret = f"{c.get('handoff_retained_pct', '?'):>7}" if c.get(
                'handoff_retained_pct') is not None else f"{'?':>7}"
            p_ret = f"{c.get('peak_retained_pct', '?'):>7}" if c.get(
                'peak_retained_pct') is not None else f"{'?':>7}"
            h_better = "Y" if c.get("handoff_better") else ("N" if c.get(
                "handoff_better") is False else "?")

            # settled vs peak cos for first target node
            nodes = c.get("nodes", [])
            target_nodes = [n for n in nodes if n.get("is_target")]
            cos = target_nodes[0].get("settled_vs_peak_dom_cos") if target_nodes else None
            cos_str = f"{cos:>11.4f}" if cos is not None else f"{'?':>11}"

            n_amp = c.get("n_amplifying_layers", "?")
            sweep_h = c.get("sweep_reduction_at_handoff")
            sweep_h_str = f"{sweep_h:>8.4f}" if sweep_h is not None else f"{'?':>8}"

            lines.append(
                f"{concept:<16} {h_ret} {p_ret} {h_better:>4} {cos_str} "
                f"{str(n_amp):>6} {sweep_h_str}")

        # Key diagnostic per model
        concepts_data = list(a["concepts"].values())
        all_h_rets = [c.get("handoff_retained_pct") for c in concepts_data
                      if c.get("handoff_retained_pct") is not None]
        target_coses = [
            n.get("settled_vs_peak_dom_cos")
            for c in concepts_data
            for n in c.get("nodes", [])
            if n.get("is_target") and n.get("settled_vs_peak_dom_cos") is not None
        ]
        if all_h_rets:
            lines.append(f"\n  Mean H-retained: {np.mean(all_h_rets):.1f}%")
        if target_coses:
            lines.append(
                f"  Mean settled-vs-peak cosine: {np.mean(target_coses):.4f}  "
                f"(min={min(target_coses):.4f}, max={max(target_coses):.4f})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--models", nargs="+",
                        default=FAILURE_MODELS + CONTRAST_MODELS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    analyses = []
    for model_id in args.models:
        log.info("Analysing %s", model_id)
        a = analyse_model(model_id)
        analyses.append(a)

    summary = print_summary(analyses)
    print(summary)

    (OUT_DIR / "gem_failure_analysis.txt").write_text(summary)
    (OUT_DIR / "gem_failure_analysis.json").write_text(
        json.dumps(analyses, indent=2, default=str))
    log.info("Wrote to %s", OUT_DIR)


if __name__ == "__main__":
    main()
