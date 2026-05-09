"""Session-scoped fixtures for the P1 claim test suite.

All data is loaded from rosetta_tools.paths — no hardcoded paths.
Missing data causes pytest.skip(), not fixture errors, so tests that
can run do run regardless of which machine or partial dataset is present.
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
    """Load all P1 concepts for one model. Returns {concept: (raw_dict, metrics)},
    or empty dict if any concept file is missing."""
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
    """Dict[concept] -> (raw_json, list[LayerMetrics]) for GPT-2-XL.
    Skips if CAZ extraction files are not present locally."""
    d = _load_model_concepts(GPT2XL_SLUG)
    if not d:
        pytest.skip(
            f"GPT-2-XL CAZ data not found at {ROSETTA_MODELS / GPT2XL_SLUG}. "
            "Run: python caz/deep_dive.py --model openai-community/gpt2-xl --n-pairs 200"
        )
    return d


@pytest.fixture(scope="session")
def p5_samedim() -> dict:
    """P5 proportional-depth samedim result JSON.
    Skips if the result file has not been synced locally."""
    if not P5_SAMEDIM_FILE.exists():
        pytest.skip(
            f"P5 result not found: {P5_SAMEDIM_FILE}. "
            "Run: python alignment/p5/run_p5_samedim.py  (or sync from GPU)"
        )
    return json.loads(P5_SAMEDIM_FILE.read_text())


@pytest.fixture(scope="session")
def p5_battery() -> dict:
    """P5 validation battery (null tests) JSON.
    Skips if the result file has not been synced locally."""
    if not P5_BATTERY_FILE.exists():
        pytest.skip(
            f"P5 battery not found: {P5_BATTERY_FILE}. "
            "Run: python alignment/p5/run_p5_battery.py  (or sync from GPU)"
        )
    return json.loads(P5_BATTERY_FILE.read_text())


@pytest.fixture(scope="session")
def all_p1_caz() -> dict:
    """All model slugs with full P1 concept coverage -> {slug: {concept: (data, metrics)}}.
    Skips if ROSETTA_MODELS directory is empty or absent."""
    if not ROSETTA_MODELS.exists():
        pytest.skip(f"ROSETTA_MODELS not found: {ROSETTA_MODELS}")

    result = {}
    for slug_dir in sorted(ROSETTA_MODELS.iterdir()):
        if not slug_dir.is_dir():
            continue
        model_data = _load_model_concepts(slug_dir.name)
        if model_data:
            result[slug_dir.name] = model_data

    if not result:
        pytest.skip(
            f"No models with full P1 concept coverage found under {ROSETTA_MODELS}. "
            "Run CAZ extraction for at least one model to enable these tests."
        )
    return result
