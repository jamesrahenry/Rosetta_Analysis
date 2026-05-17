"""Shared constants and utilities for the P1 claim test suite."""
import json
from pathlib import Path

import numpy as np

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_PAPER_N250, ROSETTA_RESULTS
from rosetta_tools.caz import LayerMetrics, compute_velocity

# The 7 concepts used in the Paper 1 proof-of-concept corpus
P1_CONCEPTS = [
    "credibility", "certainty", "causation",
    "temporal_order", "sentiment", "negation", "moral_valence",
]

# paper_n250/ is the canonical frozen H200 N=250 extraction; models/ is the live rcp_v1 fallback.
P1_MODELS_SEARCH: list[Path] = [p for p in [ROSETTA_PAPER_N250, ROSETTA_MODELS] if p.exists()]
# Convenience reference for skip messages
P1_MODELS_ROOT = P1_MODELS_SEARCH[0] if P1_MODELS_SEARCH else ROSETTA_PAPER_N250

GPT2XL_SLUG = "openai_community_gpt2_xl"

P5_SAMEDIM_FILE = ROSETTA_RESULTS / "CAZ_Framework" / "p5" / "p5_propdepth_samedim_results.json"
P5_BATTERY_FILE = ROSETTA_RESULTS / "CAZ_Framework" / "p5" / "p5_validation_battery.json"


def metrics_from_caz_json(caz_data: dict) -> list:
    """Reconstruct list[LayerMetrics] from a stored caz_*.json dict."""
    raw = caz_data["layer_data"]["metrics"]
    seps = [m["separation_fisher"] for m in raw]
    cohs = [m["coherence"]         for m in raw]
    vels = compute_velocity(seps)
    return [
        LayerMetrics(layer=i, separation=seps[i], coherence=cohs[i], velocity=float(vels[i]))
        for i in range(len(seps))
    ]
