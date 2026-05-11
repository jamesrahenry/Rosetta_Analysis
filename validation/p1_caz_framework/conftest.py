"""Session-scoped fixtures for the P1 claim test suite.

All data is loaded from rosetta_tools.paths — no hardcoded paths.
Missing data causes pytest.skip(), not fixture errors, so tests that
can run do run regardless of which machine or partial dataset is present.
"""
import json
import pytest

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_MODELS_SNAPSHOTS, ROSETTA_RESULTS
from validation.p1_caz_framework._helpers import (
    P1_CONCEPTS, P1_SNAPSHOT_SUFFIX, GPT2XL_SLUG,
    P5_SAMEDIM_FILE, P5_BATTERY_FILE,
    metrics_from_caz_json,
)


def _load_model_concepts(slug: str) -> dict:
    """Load all P1 concepts for one model from ROSETTA_MODELS_SNAPSHOTS.
    Returns {concept: (raw_dict, metrics)}, or empty dict if any concept file is missing."""
    result = {}
    slug_dir = ROSETTA_MODELS_SNAPSHOTS / slug
    for concept in P1_CONCEPTS:
        path = slug_dir / f"caz_{concept}.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        result[concept] = (data, metrics_from_caz_json(data))
    return result


@pytest.fixture(scope="session")
def gpt2xl_caz() -> dict:
    """Dict[concept] -> (raw_json, list[LayerMetrics]) for GPT-2-XL (P1 N=100 snapshot).
    Skips if CAZ extraction files are not present locally."""
    d = _load_model_concepts(GPT2XL_SLUG)
    if not d:
        pytest.skip(
            f"GPT-2-XL P1 CAZ data not found at {ROSETTA_MODELS_SNAPSHOTS / GPT2XL_SLUG}. "
            "Run: ./scripts/reproduce_p1.sh --gpu-only"
        )
    return d


@pytest.fixture(scope="session")
def p5_samedim() -> dict:
    """P5 proportional-depth samedim result JSON (CAZ_Framework/p5/).
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


_INSTRUCT_MARKERS = ("_Instruct", "_instruct", "_it_p1n100")

# Base slugs for P1 models — derived from _p1n100 snapshot names.
# Used by all_p1_n200_caz to locate the matching N=200/250 extractions.
def _p1_base_slugs() -> list[str]:
    if not ROSETTA_MODELS_SNAPSHOTS.exists():
        return []
    return [
        d.name.replace("_p1n100", "")
        for d in sorted(ROSETTA_MODELS_SNAPSHOTS.iterdir())
        if d.is_dir()
        and d.name.endswith("_p1n100")
        and not any(m in d.name for m in _INSTRUCT_MARKERS)
    ]


@pytest.fixture(scope="session")
def all_p1_caz() -> dict:
    """Base-model P1 snapshots with full concept coverage -> {slug: {concept: (data, metrics)}}.
    Reads from ROSETTA_MODELS_SNAPSHOTS, filtering to _p1n100 base models only.
    Instruct variants are excluded — P1 paper claims are established on base models."""
    if not ROSETTA_MODELS_SNAPSHOTS.exists():
        pytest.skip(f"ROSETTA_MODELS_SNAPSHOTS not found: {ROSETTA_MODELS_SNAPSHOTS}")

    result = {}
    for slug_dir in sorted(ROSETTA_MODELS_SNAPSHOTS.iterdir()):
        if not slug_dir.is_dir():
            continue
        if not slug_dir.name.endswith(P1_SNAPSHOT_SUFFIX):
            continue
        if any(m in slug_dir.name for m in _INSTRUCT_MARKERS):
            continue
        model_data = _load_model_concepts(slug_dir.name)
        if model_data:
            result[slug_dir.name] = model_data

    if not result:
        pytest.skip(
            f"No P1 base-model snapshots (*{P1_SNAPSHOT_SUFFIX}) with full concept coverage found "
            f"under {ROSETTA_MODELS_SNAPSHOTS}. Run: ./scripts/reproduce_p1.sh --gpu-only"
        )
    return result


@pytest.fixture(scope="session")
def all_p1_n200_caz() -> dict:
    """Base-model P1 data at N≥200 -> {slug: {concept: (data, metrics)}}.

    Reads from ROSETTA_MODELS (N=200/250 extractions) using the same model set
    as all_p1_caz. Used for the §5.2 cross-architecture ordering test, where
    N=200 reduces peak-depth noise and gives the paper's stated 81% positive rate.
    """
    base_slugs = _p1_base_slugs()
    if not base_slugs:
        pytest.skip("No P1 base-model snapshots found — cannot derive N=200 slug list")

    result = {}
    for base in base_slugs:
        model_dir = ROSETTA_MODELS / base
        if not model_dir.is_dir():
            continue
        data = {}
        ok = True
        for concept in P1_CONCEPTS:
            path = model_dir / f"caz_{concept}.json"
            if not path.exists():
                ok = False
                break
            raw = json.loads(path.read_text())
            data[concept] = (raw, metrics_from_caz_json(raw))
        if ok:
            result[base] = data

    if not result:
        pytest.skip(
            f"No P1 base-model N=200 data found under {ROSETTA_MODELS}. "
            "Run: ./scripts/reproduce_p2.sh --gpu-only  (covers the P1 model set)"
        )
    return result
