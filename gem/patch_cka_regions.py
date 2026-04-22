"""
patch_cka_regions.py — Recompute caz_regions in existing CKA files.

Reads caz_{concept}.json (separation data) and updates the caz_regions
field in cka_{concept}.json using the current find_caz_regions_scored
(with valley-merge pass).  No model loading or GPU needed.

Usage
-----
    python src/patch_cka_regions.py --all
    python src/patch_cka_regions.py --model mistralai/Mistral-7B-v0.3
    python src/patch_cka_regions.py --model google/gemma-2-9b

Written: 2026-04-19 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics, final_global_attention_layer
from rosetta_tools.models import attention_paradigm_of

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"
CONCEPTS    = ["sentiment", "credibility", "negation", "causation",
               "certainty", "moral_valence", "temporal_order"]


def find_run_dir(model_id: str) -> Path | None:
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if d.is_dir() and sf.exists():
            try:
                if json.loads(sf.read_text()).get("model_id") == model_id:
                    return d
            except Exception:
                continue
    return None


def recompute_regions(model_id: str, run_dir: Path) -> dict[str, int]:
    """Recompute and patch caz_regions for all concepts in run_dir.

    Returns counts: {concept: old_n_regions -> new_n_regions change}
    """
    paradigm = attention_paradigm_of(model_id)
    results: dict[str, str] = {}

    for concept in CONCEPTS:
        caz_file = run_dir / f"caz_{concept}.json"
        cka_file = run_dir / f"cka_{concept}.json"
        if not caz_file.exists():
            continue

        caz = json.loads(caz_file.read_text())
        metrics_raw = caz["layer_data"]["metrics"]
        layer_metrics = [
            LayerMetrics(
                layer=m["layer"],
                separation=m["separation_fisher"],
                coherence=m.get("coherence", 0.0),
                velocity=m.get("velocity", 0.0),
            )
            for m in metrics_raw
        ]
        n_layers = len(layer_metrics)

        functional_peak = None
        if paradigm == "alternating":
            functional_peak = final_global_attention_layer(n_layers)

        profile = find_caz_regions_scored(
            layer_metrics,
            attention_paradigm=paradigm,
            functional_peak_layer=functional_peak,
        )
        new_regions = [
            {k: int(v) for k, v in {"start": r.start, "peak": r.peak, "end": r.end}.items()}
            for r in profile.regions
        ]

        if cka_file.exists():
            cka = json.loads(cka_file.read_text())
            old_n = len(cka.get("caz_regions", []))
        else:
            # No CKA file yet — create a minimal stub so the viz can draw Fisher bands.
            # CKA adjacency data requires model loading; leave those fields empty.
            cka = {
                "model_id":         model_id,
                "concept":          concept,
                "n_layers":         n_layers,
                "cka_adjacent":     [],
                "within_caz_indices": [],
                "cross_caz_indices":  [],
                "within_caz_mean":  None,
                "cross_caz_mean":   None,
                "mann_whitney_U":   None,
                "mann_whitney_p":   None,
                "hypothesis_supported": None,
                "caz_regions":      [],
                "stub":             True,
            }
            old_n = 0
            log.info("  Creating stub CKA file for %s / %s", model_id.split("/")[-1], concept)

        new_n = len(new_regions)
        cka["caz_regions"] = new_regions
        cka_file.write_text(json.dumps(cka, indent=2))

        change = f"{old_n}→{new_n}"
        results[concept] = change
        if old_n != new_n:
            log.info("  %s %s: %s regions", model_id.split("/")[-1], concept, change)

    return results


def all_model_ids() -> list[str]:
    ids = []
    for d in sorted(RESULTS_DIR.iterdir()):
        sf = d / "run_summary.json"
        if d.is_dir() and sf.exists():
            try:
                mid = json.loads(sf.read_text()).get("model_id")
                if mid and any((d / f"cka_{c}.json").exists() for c in CONCEPTS):
                    ids.append(mid)
            except Exception:
                continue
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--model", metavar="HF_ID")
    args = ap.parse_args()

    targets = all_model_ids() if args.all else [args.model]

    for mid in targets:
        rd = find_run_dir(mid)
        if rd is None:
            log.warning("No results dir for %s — skipping", mid)
            continue
        log.info("Patching %s (%s)", mid.split("/")[-1], rd.name)
        recompute_regions(mid, rd)

    log.info("Done.")


if __name__ == "__main__":
    main()
