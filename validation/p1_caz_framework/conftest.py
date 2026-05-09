"""Session-scoped fixtures for the P1 claim test suite.

All data is loaded from rosetta_tools.paths — no hardcoded paths.
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
    """Load all P1 concepts for one model. Returns {concept: (raw_dict, metrics)}."""
    result = {}
    slug_dir = ROSETTA_MODELS / slug
    for concept in P1_CONCEPTS:
        path = slug_dir / f"caz_{concept}.json"
        if not path.exists():
            return {}          # reject models missing any concept
        data = json.loads(path.read_text())
        result[concept] = (data, metrics_from_caz_json(data))
    return result


@pytest.fixture(scope="session")
def gpt2xl_caz() -> dict:
    """Dict[concept] -> (raw_json, list[LayerMetrics]) for GPT-2-XL."""
    d = _load_model_concepts(GPT2XL_SLUG)
    assert d, f"GPT-2-XL CAZ data missing at {ROSETTA_MODELS / GPT2XL_SLUG}"
    return d


@pytest.fixture(scope="session")
def p5_samedim() -> dict:
    assert P5_SAMEDIM_FILE.exists(), f"Missing {P5_SAMEDIM_FILE}"
    return json.loads(P5_SAMEDIM_FILE.read_text())


@pytest.fixture(scope="session")
def p5_battery() -> dict:
    assert P5_BATTERY_FILE.exists(), f"Missing {P5_BATTERY_FILE}"
    return json.loads(P5_BATTERY_FILE.read_text())


@pytest.fixture(scope="session")
def all_p1_caz() -> dict:
    """All model slugs with full P1 concept coverage -> {slug: {concept: (data, metrics)}}."""
    result = {}
    for slug_dir in sorted(ROSETTA_MODELS.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        model_data = _load_model_concepts(slug)
        if model_data:
            result[slug] = model_data
    return result
