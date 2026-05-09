"""Shared constants and utilities for the P1 claim test suite."""
import json
from pathlib import Path

import numpy as np

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS
from rosetta_tools.caz import LayerMetrics, compute_velocity

# The 7 concepts used in the Paper 1 proof-of-concept corpus
P1_CONCEPTS = [
    "credibility", "certainty", "causation",
    "temporal_order", "sentiment", "negation", "moral_valence",
]

GPT2XL_SLUG = "openai_community_gpt2_xl"

P5_SAMEDIM_FILE = ROSETTA_RESULTS / "p5_permutation" / "p5_propdepth_samedim_results.json"
P5_BATTERY_FILE = ROSETTA_RESULTS / "p5_permutation" / "p5_validation_battery.json"


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
