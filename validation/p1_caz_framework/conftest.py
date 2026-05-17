"""Session-scoped fixtures for the P1 claim test suite.

All data is loaded from rosetta_tools.paths — no hardcoded paths.
Missing data causes pytest.skip(), not fixture errors, so tests that
can run do run regardless of which machine or partial dataset is present.

Canonical source: ROSETTA_MODELS (paper_n250 rsync, N=250 pairs).
"""
import json
import pytest

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS
from validation.p1_caz_framework._helpers import (
    P1_CONCEPTS, GPT2XL_SLUG,
    P5_SAMEDIM_FILE, P5_BATTERY_FILE,
    metrics_from_caz_json,
)


def _load_model_concepts(slug: str) -> dict:
    """Load all P1 concepts for one model from ROSETTA_MODELS.
    Returns {concept: (raw_dict, metrics)}, or empty dict if any concept file is missing."""
    result = {}
    slug_dir = ROSETTA_MODELS / slug
    for concept in P1_CONCEPTS:
        path = slug_dir / f"caz_{concept}.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        result[concept] = (data, metrics_from_caz_json(data))
    return result


@pytest.fixture(scope="session")
def gpt2xl_caz() -> dict:
    """Dict[concept] -> (raw_json, list[LayerMetrics]) for GPT-2-XL (canonical N=250).
    Skips if CAZ extraction files are not present locally."""
    d = _load_model_concepts(GPT2XL_SLUG)
    if not d:
        pytest.skip(
            f"GPT-2-XL CAZ data not found at {ROSETTA_MODELS / GPT2XL_SLUG}. "
            "Restore from HF: hf download james-ra-henry/Rosetta-Activations --include 'paper_n250/*'"
        )
    return d


@pytest.fixture(scope="session")
def p5_samedim() -> dict:
    """P5 proportional-depth samedim result JSON (CAZ_Framework/p5/, 280 trials).
    Skips if the result file has not been synced locally."""
    if not P5_SAMEDIM_FILE.exists():
        pytest.skip(
            f"P5 result not found: {P5_SAMEDIM_FILE}. "
            "Run: ./scripts/reproduce_p1.sh  (or sync from mi-host)"
        )
    return json.loads(P5_SAMEDIM_FILE.read_text())


@pytest.fixture(scope="session")
def p5_battery() -> dict:
    """P5 validation battery (null tests) JSON (CAZ_Framework/p5/).
    Skips if the result file has not been synced locally."""
    if not P5_BATTERY_FILE.exists():
        pytest.skip(
            f"P5 battery not found: {P5_BATTERY_FILE}. "
            "Run: ./scripts/reproduce_p1.sh  (or sync from mi-host)"
        )
    return json.loads(P5_BATTERY_FILE.read_text())


_INSTRUCT_MARKERS = ("_Instruct", "_instruct")


@pytest.fixture(scope="session")
def all_p1_caz() -> dict:
    """Base-model P1 data (canonical N=250) -> {slug: {concept: (data, metrics)}}.
    Reads from ROSETTA_MODELS, filtering to base models only.
    Instruct variants are excluded — P1 paper claims are established on base models."""
    if not ROSETTA_MODELS.exists():
        pytest.skip(f"ROSETTA_MODELS not found: {ROSETTA_MODELS}")

    result = {}
    for slug_dir in sorted(ROSETTA_MODELS.iterdir()):
        if not slug_dir.is_dir():
            continue
        if any(m in slug_dir.name for m in _INSTRUCT_MARKERS):
            continue
        model_data = _load_model_concepts(slug_dir.name)
        if model_data:
            result[slug_dir.name] = model_data

    if not result:
        pytest.skip(
            f"No P1 base-model data with full concept coverage found under {ROSETTA_MODELS}. "
            "Restore from HF: hf download james-ra-henry/Rosetta-Activations --include 'paper_n250/*'"
        )
    return result


@pytest.fixture(scope="session")
def all_p1_n200_caz(all_p1_caz: dict) -> dict:
    """Canonical N=250 data — aliases all_p1_caz.

    Previously a separate load of N≥200 extractions from ROSETTA_MODELS; now
    that all canonical data is N=250, this fixture simply returns all_p1_caz.
    Kept for backward compatibility with TestCrossArchOrdering.
    """
    return all_p1_caz
