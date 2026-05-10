"""Session-scoped fixtures for the P2 (GEM) claim test suite.

All data loaded via rosetta_tools.paths — no hardcoded paths.
Missing data causes pytest.skip(), not fixture errors.
"""
import json
import pytest

from validation.p2_gem._helpers import (
    P2_MODELS, P2_CONCEPTS,
    load_ablation_pair, load_gem_nodes,
    GEM_EEC_CORPUS_FILE, GEM_ADAPTIVE_WIDTH_FILE,
)
from rosetta_tools.paths import ROSETTA_MODELS


@pytest.fixture(scope="session")
def gem_corpus() -> list[dict]:
    """All compare-peak ablation results — list of comparison dicts.

    Each entry is ablation_gem_{concept}.json["comparison"] augmented with
    {"model_id": ..., "concept": ...} for downstream filtering.

    Skips if fewer than 200 of the 272 expected pairs are present.
    Assumes Qwen2.5-14B data will arrive (272 total expected).
    """
    records = []
    for model_id in P2_MODELS:
        for concept in P2_CONCEPTS:
            comp = load_ablation_pair(model_id, concept)
            if comp is not None:
                records.append({"model_id": model_id, "concept": concept, **comp})

    if len(records) < 200:
        pytest.skip(
            f"Too few GEM ablation pairs: {len(records)}/272 present. "
            "Run: ./scripts/reproduce_p2.sh --gpu-only"
        )
    return records


@pytest.fixture(scope="session")
def eec_corpus() -> dict:
    """GEM EEC values — {model_id: {concept: eec_value}}.

    Loaded from gem_eec_corpus.json (produced by aggregate_gem_results.py).
    """
    if not GEM_EEC_CORPUS_FILE.exists():
        pytest.skip(
            f"EEC corpus not found: {GEM_EEC_CORPUS_FILE}. "
            "Run: python gem/aggregate_gem_results.py --p2-corpus"
        )
    return json.loads(GEM_EEC_CORPUS_FILE.read_text())


@pytest.fixture(scope="session")
def gem_nodes() -> list[dict]:
    """Node-level GEM data — list of gem_{concept}.json dicts augmented with model_id/concept.

    Used for handoff cosine and per-node EEC tests.
    Skips if fewer than 200 pairs present.
    """
    records = []
    for model_id in P2_MODELS:
        for concept in P2_CONCEPTS:
            data = load_gem_nodes(model_id, concept)
            if data is not None:
                records.append({"model_id": model_id, "concept": concept, **data})

    if len(records) < 200:
        pytest.skip(
            f"Too few GEM node files: {len(records)}/272 present. "
            "Run: python gem/build_gems.py --all"
        )
    return records


@pytest.fixture(scope="session")
def adaptive_width_data() -> dict:
    """GEM adaptive-width ablation summary computed from raw record list.

    Source: gem_adaptive_width/gem_adaptive_width.json (list of per-pair records).
    Returns a summary dict with the fields the test suite expects.

    "Triggered" pairs are those with rule=="near-final" (adaptive_width==1).
    "Depth-corrected" are triggered pairs with p_value < 0.05 (significant
    improvement vs null distribution).
    """
    import statistics as _stats

    if not GEM_ADAPTIVE_WIDTH_FILE.exists():
        pytest.skip(
            f"Adaptive width results not found: {GEM_ADAPTIVE_WIDTH_FILE}. "
            "Run: python gem/ablate_gem_adaptive_width.py"
        )

    records = json.loads(GEM_ADAPTIVE_WIDTH_FILE.read_text())
    if not isinstance(records, list) or not records:
        pytest.skip("adaptive_width JSON is empty or unexpected format")

    triggered  = [r for r in records if r.get("rule") == "near-final"]
    wins       = [r for r in triggered if r.get("adaptive_better")]
    all_diffs  = [r["delta_vs_fixed3_pp"] for r in records if r.get("delta_vs_fixed3_pp") is not None]
    win_diffs  = [r["delta_vs_fixed3_pp"] for r in wins if r.get("delta_vs_fixed3_pp") is not None]
    dc         = [r for r in triggered if r.get("p_value") is not None and r["p_value"] < 0.05]
    dc_wins    = [r for r in dc if r.get("adaptive_better")]
    dc_diffs   = [r["delta_vs_fixed3_pp"] for r in dc_wins if r.get("delta_vs_fixed3_pp") is not None]

    return {
        "total_pairs":             len(records),
        "triggered":               len(triggered),
        "adaptive_wins":           len(wins),
        "mean_diff_triggered_pp":  _stats.mean(win_diffs) if win_diffs else None,
        "mean_diff_overall_pp":    _stats.mean(all_diffs) if all_diffs else None,
        "overall_lift_pp":         _stats.mean(all_diffs) if all_diffs else None,
        "depth_corrected": {
            "total":       len(dc),
            "wins":        len(dc_wins),
            "mean_diff_pp": _stats.mean(dc_diffs) if dc_diffs else None,
        },
    }
