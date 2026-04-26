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
PROBE_DIR   = Path("~/Source/Concept_Integrity_Auditor/probes").expanduser()

# Layer counts per model (post-embedding, matching train_acts[1:] indexing)
MODEL_N_LAYERS = {
    "Qwen/Qwen2.5-7B-Instruct": 28,
    "EleutherAI/pythia-6.9b": 32,
}


def model_to_probe_slug(model_name: str) -> str:
    return model_name.replace("/", "_").lower()


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


def load_probe_regions(model_name: str, n_layers: int) -> dict[str, list[dict]] | None:
    """Load per-region handoff data from a CIA probe JSON.

    Returns ``concept -> [region_handoff_dict, ...]``, or ``None`` when
    the probe file is missing (e.g. pythia probes not yet rebuilt).
    """
    slug = model_to_probe_slug(model_name)
    path = PROBE_DIR / f"{slug}.json"
    if not path.exists():
        log.warning("Probe file not found: %s", path)
        return None
    data = json.loads(path.read_text())
    out: dict[str, list[dict]] = {}
    for concept, meta in data.get("concepts", {}).items():
        regions = meta.get("caz_regions", [])
        region_handoffs = []
        for i, r in enumerate(regions):
            handoff = min(int(r["end"]) + 1, n_layers - 1)
            region_handoffs.append({
                "region_idx":   i,
                "start":        int(r["start"]),
                "end":          int(r["end"]),
                "peak":         int(r["peak"]),
                "handoff":      handoff,
                "handoff_frac": handoff / (n_layers - 1),
                "caz_score":    float(r.get("caz_score", 0.0)),
            })
        out[concept] = region_handoffs
    log.info("Loaded probe regions: %s (%d concepts)", path.name, len(out))
    return out


def compare_per_region(
    qwen_regions: dict[str, list[dict]],
    pythia_regions: dict[str, list[dict]],
) -> dict:
    """Per-region handoff comparison across two models, matched by ordinal index.

    For each concept, regions are matched by index: region 0 (shallowest) in
    Qwen ↔ region 0 in pythia, etc. Concepts may have different region counts;
    we use min(N_qwen, N_pythia) matched pairs per concept.
    """
    shared = sorted(set(qwen_regions) & set(pythia_regions))
    rows = []
    for concept in shared:
        q_regs = qwen_regions[concept]
        p_regs = pythia_regions[concept]
        n_match = min(len(q_regs), len(p_regs))
        for i in range(n_match):
            q, p = q_regs[i], p_regs[i]
            rows.append({
                "concept":        concept,
                "region_idx":     i,
                "n_qwen_regions": len(q_regs),
                "n_pythia_regions": len(p_regs),
                "qwen_range":     f"L{q['start']}-{q['end']}",
                "qwen_handoff":   q["handoff"],
                "qwen_frac":      round(q["handoff_frac"], 4),
                "pythia_range":   f"L{p['start']}-{p['end']}",
                "pythia_handoff": p["handoff"],
                "pythia_frac":    round(p["handoff_frac"], 4),
                "delta_frac":     round(q["handoff_frac"] - p["handoff_frac"], 4),
            })
    if not rows:
        return {"n_region_pairs": 0, "rows": []}

    qfracs = np.array([r["qwen_frac"] for r in rows])
    pfracs = np.array([r["pythia_frac"] for r in rows])
    abs_deltas = np.abs(qfracs - pfracs)
    correlation = (
        float(np.corrcoef(qfracs, pfracs)[0, 1]) if len(rows) >= 2 else float("nan")
    )

    # Bucket by depth tertile to see whether shallow/mid/deep regions agree at each level
    def bucket(frac: float) -> str:
        if frac < 0.33:  return "shallow"
        if frac < 0.67:  return "mid"
        return "deep"

    bucket_stats: dict[str, dict] = {}
    for b in ("shallow", "mid", "deep"):
        mask = np.array([bucket(f) == b for f in qfracs])
        if mask.any():
            bucket_stats[b] = {
                "n":              int(mask.sum()),
                "mean_abs_delta": float(np.mean(abs_deltas[mask])),
                "max_abs_delta":  float(np.max(abs_deltas[mask])),
            }

    return {
        "n_region_pairs":   len(rows),
        "rows":             rows,
        "mean_abs_delta":   float(np.mean(abs_deltas)),
        "median_abs_delta": float(np.median(abs_deltas)),
        "max_abs_delta":    float(np.max(abs_deltas)),
        "pearson_r":        correlation,
        "bucket_stats":     bucket_stats,
        "interpretation":   (
            "PRH-supportive (per-region): r > 0.7 AND mean |Δ| < 0.10."
            if correlation > 0.7 and float(np.mean(abs_deltas)) < 0.10
            else "PRH-mixed/weak (per-region): see per-row deltas and bucket_stats."
        ),
    }


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

    # ── Per-region analysis (from probe JSONs) ──────────────────────────────
    log.info("\n%s", "=" * 90)
    log.info("PER-REGION ANALYSIS (matched by ordinal index, shallow → deep)")
    log.info("=" * 90)
    qwen_regions   = load_probe_regions(
        "Qwen/Qwen2.5-7B-Instruct", MODEL_N_LAYERS["Qwen/Qwen2.5-7B-Instruct"]
    )
    pythia_regions = load_probe_regions(
        "EleutherAI/pythia-6.9b", MODEL_N_LAYERS["EleutherAI/pythia-6.9b"]
    )

    region_cmp: dict | None = None
    if qwen_regions is None or pythia_regions is None:
        missing = []
        if qwen_regions is None:   missing.append("qwen")
        if pythia_regions is None: missing.append("pythia")
        log.warning(
            "Skipping per-region analysis — missing probe JSON for: %s. "
            "Run probe rebuild jobs first (e.g. t397473a for pythia-6.9b).",
            ", ".join(missing),
        )
    else:
        region_cmp = compare_per_region(qwen_regions, pythia_regions)
        log.info(
            "%-22s %4s %-12s %4s %5s  %-12s %4s %5s  %7s",
            "Concept", "rgn", "Qwen rng", "L", "frac", "Pythia rng", "L", "frac", "Δ frac",
        )
        log.info("-" * 90)
        for row in region_cmp["rows"]:
            log.info(
                "%-22s %4d %-12s %4d %5.3f  %-12s %4d %5.3f  %+7.3f",
                row["concept"], row["region_idx"],
                row["qwen_range"],   row["qwen_handoff"],   row["qwen_frac"],
                row["pythia_range"], row["pythia_handoff"], row["pythia_frac"],
                row["delta_frac"],
            )
        log.info("-" * 90)
        log.info("Region pairs:      %d", region_cmp["n_region_pairs"])
        log.info("Pearson r:         %.3f", region_cmp["pearson_r"])
        log.info("Mean |Δ frac|:     %.3f", region_cmp["mean_abs_delta"])
        log.info("Median |Δ frac|:   %.3f", region_cmp["median_abs_delta"])
        log.info("Max |Δ frac|:      %.3f", region_cmp["max_abs_delta"])
        for bucket, stats in region_cmp["bucket_stats"].items():
            log.info(
                "  %-7s n=%-3d  mean |Δ|=%.3f  max |Δ|=%.3f",
                bucket, stats["n"], stats["mean_abs_delta"], stats["max_abs_delta"],
            )
        log.info("\n%s", region_cmp["interpretation"])

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"handoff_depth_fingerprint_{ts}.json"
    out.write_text(json.dumps({
        "qwen_source":      qwen_path.name,
        "pythia_source":    pythia_path.name,
        "timestamp":        ts,
        "dominant_region":  cmp,
        "per_region":       region_cmp,
    }, indent=2))
    log.info("\nResults → %s", out)


if __name__ == "__main__":
    main()
