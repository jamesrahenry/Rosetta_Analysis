"""
prh_handoff_depth.py — PRH cross-model handoff-depth fingerprint analysis.

Tests the PRH (Platonic Representation Hypothesis) prediction that models
share the transformation PATH, not just endpoints, by comparing per-concept
GEM handoff layers as a fraction of network depth across architectures.

If the same concepts hand off at consistent relative depths across Qwen
and pythia (different families, different layer counts), it corroborates
PRH within the existing CIA trajectory experiment at zero extra GPU cost.

Inputs: trajectory_scoring_*.json from cia_trajectory_scoring.py runs.
Output: handoff_depth_fingerprint_<ts>.json with per-concept comparison.

Written: 2026-04-26 UTC
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RESULTS_DIR = Path("~/Source/Rosetta_Program/rosetta_data/cia_trajectory").expanduser()

# Layer counts per model (post-embedding, matching train_acts[1:] indexing)
MODEL_N_LAYERS = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "EleutherAI/pythia-6.9b": 32,
}


def latest_result(model_slug: str) -> Path:
    """Return the most recent trajectory_scoring JSON for a model."""
    candidates = sorted(RESULTS_DIR.glob(f"trajectory_scoring_{model_slug}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory results for {model_slug} in {RESULTS_DIR}")
    return candidates[-1]


def load_handoffs(path: Path, n_layers: int) -> dict[str, dict]:
    """Extract per-concept handoff info from a trajectory_scoring JSON.

    Handoff layer is caz_end + 1 (clamped to last layer) — the GEM convention
    for the settled-product layer following the dominant CAZ region.
    """
    data = json.loads(path.read_text())
    out = {}
    for r in data["results"]:
        if "error" in r or "caz_end" not in r:
            continue
        caz_end = r["caz_end"]
        handoff = min(caz_end + 1, n_layers - 1)
        out[r["concept"]] = {
            "caz_start":     r["caz_start"],
            "caz_end":       caz_end,
            "handoff_layer": handoff,
            "handoff_frac":  handoff / (n_layers - 1),
            "n_regions":     r.get("n_regions"),
            "is_multimodal": r.get("is_multimodal"),
            "n_layers":      n_layers,
        }
    return out


def compare(qwen: dict[str, dict], pythia: dict[str, dict]) -> dict:
    """Compare per-concept handoff fractions across two models."""
    shared = sorted(set(qwen) & set(pythia))
    rows = []
    for concept in shared:
        q = qwen[concept]
        p = pythia[concept]
        delta_frac = q["handoff_frac"] - p["handoff_frac"]
        rows.append({
            "concept":         concept,
            "qwen_handoff":    q["handoff_layer"],
            "qwen_frac":       round(q["handoff_frac"], 4),
            "qwen_n_layers":   q["n_layers"],
            "pythia_handoff":  p["handoff_layer"],
            "pythia_frac":     round(p["handoff_frac"], 4),
            "pythia_n_layers": p["n_layers"],
            "delta_frac":      round(delta_frac, 4),
            "qwen_multimodal": q["is_multimodal"],
            "pythia_multimodal": p["is_multimodal"],
        })

    # Aggregate stats
    deltas = np.array([row["delta_frac"] for row in rows])
    abs_deltas = np.abs(deltas)
    qwen_fracs = np.array([row["qwen_frac"] for row in rows])
    pythia_fracs = np.array([row["pythia_frac"] for row in rows])
    correlation = float(np.corrcoef(qwen_fracs, pythia_fracs)[0, 1]) if len(rows) >= 2 else float("nan")

    return {
        "n_concepts":            len(rows),
        "concepts":              shared,
        "rows":                  rows,
        "mean_abs_delta":        float(np.mean(abs_deltas)),
        "median_abs_delta":      float(np.median(abs_deltas)),
        "max_abs_delta":         float(np.max(abs_deltas)),
        "pearson_r":             correlation,
        "interpretation": (
            "PRH-supportive: handoff fractions correlate strongly across models "
            "(r > 0.7) AND mean absolute delta < 0.10."
            if correlation > 0.7 and np.mean(abs_deltas) < 0.10
            else "PRH-mixed/weak: see per-concept rows for divergences."
        ),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    qwen_path   = latest_result("Qwen_Qwen2.5-7B-Instruct")
    pythia_path = latest_result("EleutherAI_pythia-6.9b")

    log.info("Qwen results:   %s", qwen_path.name)
    log.info("Pythia results: %s", pythia_path.name)

    qwen   = load_handoffs(qwen_path,   MODEL_N_LAYERS["Qwen/Qwen2.5-7B-Instruct"])
    pythia = load_handoffs(pythia_path, MODEL_N_LAYERS["EleutherAI/pythia-6.9b"])

    cmp = compare(qwen, pythia)

    # Print table
    log.info("\n%s", "=" * 90)
    log.info(
        "%-22s %8s %8s %8s %8s %8s",
        "Concept", "Qwen L", "Qwen %", "Pythia L", "Pythia %", "Δ frac"
    )
    log.info("-" * 90)
    for row in cmp["rows"]:
        log.info(
            "%-22s %8d %8.3f %8d %8.3f %+8.3f",
            row["concept"],
            row["qwen_handoff"], row["qwen_frac"],
            row["pythia_handoff"], row["pythia_frac"],
            row["delta_frac"],
        )
    log.info("=" * 90)
    log.info("n_concepts:        %d", cmp["n_concepts"])
    log.info("Pearson r:         %.3f", cmp["pearson_r"])
    log.info("Mean |Δ frac|:     %.3f", cmp["mean_abs_delta"])
    log.info("Median |Δ frac|:   %.3f", cmp["median_abs_delta"])
    log.info("Max |Δ frac|:      %.3f", cmp["max_abs_delta"])
    log.info("\n%s", cmp["interpretation"])

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"handoff_depth_fingerprint_{ts}.json"
    out.write_text(json.dumps({
        "qwen_source":   qwen_path.name,
        "pythia_source": pythia_path.name,
        "timestamp":     ts,
        **cmp,
    }, indent=2))
    log.info("\nResults → %s", out)


if __name__ == "__main__":
    main()
