#!/usr/bin/env python3
"""Shared helpers for the round-3 GPU session (G2, G3, G5, G6).

Plan: papers/shared/ROUND3_COMPUTE_PLAN.md ("GPU session implementation plan").
Review: papers/caz-validation/antagonistic-review-2026-07-15-round3.md.

Conventions enforced here (do not bypass in job scripts):
  * per-(job, model) checkpoint shards under CKPT_ROOT — a crash costs one
    model, not the session;
  * every job uploads its outputs to HF *when the job finishes*, not at
    session teardown (the C7/C11 lesson);
  * bf16 forward passes, float64 statistics — matches the papers' pipeline.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("round3")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths (GPU-host layout per Rosetta_Program/CLAUDE.md — never ~/Source here)
# ---------------------------------------------------------------------------
ROSETTA_DATA = Path(os.environ.get("ROSETTA_DATA", Path.home() / "rosetta_data"))
MODELS_ROOT = ROSETTA_DATA / "models"          # canonical local artifact tree
PAPER_TREE = "paper_n250"                       # HF-side prefix
CKPT_ROOT = ROSETTA_DATA / "round3_ckpt"
OUT_ROOT = ROSETTA_DATA / "results" / "round3_gpu"

HF_DATASET = "james-ra-henry/Rosetta-Activations"
HF_DEST_PREFIX = f"{PAPER_TREE}/_round3_gpu"    # <prefix>/<job>/<file>

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
CONCEPTS_17 = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
CONCEPTS_7 = [
    "credibility", "certainty", "causation", "temporal_order",
    "negation", "sentiment", "moral_valence",
]

# ---------------------------------------------------------------------------
# Model rosters
# ---------------------------------------------------------------------------


def slugify(model_id: str) -> str:
    """HF model id -> local/HF artifact directory slug (established convention)."""
    return model_id.replace("/", "_").replace("-", "_")


# P3's base-model evaluation corpus (Table 1 of caz-validation/preprint.md).
# NOTE: facebook/opt-350m EXCLUDED 2026-07-16 (was in the 28-base list) — it is
# the only OPT variant with word_embed_proj_dim (512) != hidden_size (1024),
# i.e. a project_in/project_out around the transformer stack. That makes its
# embedding activation row 512-dim while all block rows are 1024-dim, which
# breaks the uniform-dimension assumption of G5's per-row np.stack extraction
# (and narrowly trips G2's offset-calibration margin). Excluded rather than
# special-cased pending science-owner review. Corpus is now 27 base models.
BASE_28 = [
    # Pythia (8)
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    # GPT-2 (4)
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    # OPT (4) — opt-350m excluded (see header note: non-uniform activation dims)
    "facebook/opt-125m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    # Qwen 2.5 (5)
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    # Gemma 2 — EXCLUDED 2026-07-16 (James): DOM directions don't stabilize
    # across pair subsamples (gemma-2-2b split-half cos 0.52-0.60 vs 0.96
    # for gpt2/qwen controls; extraction itself fully deterministic).
    # Set aside for separate investigation, not silently droppable noise.
    # Llama 3.2 (2)
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    # Mistral (1) / Phi (1)
    "mistralai/Mistral-7B-v0.3", "microsoft/phi-2",
]
assert len(BASE_28) == 25

# G3's permutation-null subset (caz-validation §4.1).
G3_SUBSET = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-1.4b",
    "openai-community/gpt2-xl", "Qwen/Qwen2.5-3B",
    # gemma-2-2b EXCLUDED 2026-07-16 — see Gemma note in BASE_28
]

# Families for cross-family pairing (mirrors p5_propdepth.get_family()).
_FAMILY_PREFIXES = [
    ("EleutherAI_pythia", "pythia"), ("EleutherAI_gpt_neo", "gpt_neo"),
    ("openai_community_gpt2", "gpt2"), ("facebook_opt", "opt"),
    ("Qwen_Qwen2.5", "qwen2"), ("google_gemma", "gemma2"),
    ("meta_llama_Llama_3.1", "llama3"), ("meta_llama_Llama_3.2", "llama3"),
    ("mistralai_Mistral", "mistral"), ("microsoft_phi", "phi"),
    ("tiiuae_falcon", "falcon"),
]


def family_of(slug: str) -> str:
    for prefix, fam in _FAMILY_PREFIXES:
        if slug.startswith(prefix):
            return fam
    raise ValueError(f"unknown family for slug {slug!r}")


# P4's alignment corpus, clusters A-E: the same-dim clusters
# (768/1024/2048/2560/3584/4096/5120 — Table S2). Includes
# gpt-neo-125m (cluster A per P4 §2.1) and base/instruct siblings (§3.8).
# PINNED 2026-07-16. NOTE: facebook/opt-350m EXCLUDED 2026-07-16 from cluster G
# (non-uniform activation dims — see BASE_28 header note); cluster G is now
# {pythia-410m, gpt2-medium}, so its cross-family ordered-pair count drops from
# 6 to 2. Roster is now 29 models. Manifest with dims/layers:
# g5_alignment_roster.json (this directory).
ALIGN_ROSTER_30 = [
    # A (768)
    "EleutherAI/gpt-neo-125m", "EleutherAI/pythia-160m", "facebook/opt-125m",
    "openai-community/gpt2",
    # G (1024) — opt-350m excluded (see BASE_28 header note: non-uniform dims)
    "EleutherAI/pythia-410m", "openai-community/gpt2-medium",
    # B (2048)
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "facebook/opt-1.3b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
    # H (2560)
    "EleutherAI/pythia-2.8b", "facebook/opt-2.7b", "microsoft/phi-2",
    # D (3584) — gemma-2-9b/-it EXCLUDED 2026-07-16 (see Gemma note in
    # BASE_28); cluster D is now Qwen-only
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
    # C (4096)
    "EleutherAI/pythia-6.9b", "facebook/opt-6.7b",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3",
    # E (5120)
    "EleutherAI/pythia-12b", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B",
]
assert len(ALIGN_ROSTER_30) == 27
ALIGN_CLUSTER_DIMS = {768, 1024, 2048, 2560, 3584, 4096, 5120}


def alignment_roster_from_hf() -> list[str]:
    """Returns the pinned 30-slug alignment roster, cross-checked against HF
    calibration_alllayer coverage (warns on drift; never silently diverges).
    The raw HF coverage is broader (extended-corpus and Gemma-4 models carry
    calibration files too) — the pinned list is the clusters A-E population."""
    pinned = [slugify(m) for m in ALIGN_ROSTER_30]
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset")
        covered = {
            f.split("/")[1] for f in files
            if f.startswith(f"{PAPER_TREE}/") and "calibration_alllayer_" in f
        }
        missing = [s for s in pinned if s not in covered]
        if missing:
            log.warning("[roster] pinned slugs missing HF calibration files: %s",
                        missing)
    except Exception as e:  # noqa: BLE001 — cross-check only, offline OK
        log.warning("[roster] HF cross-check skipped (%s)", e)
    return pinned


# ---------------------------------------------------------------------------
# Artifact readers (formats verified against ~/rosetta_data 2026-07-16)
# ---------------------------------------------------------------------------


def load_caz(slug: str, concept: str, root: Path = MODELS_ROOT) -> dict:
    """caz_<concept>.json -> dict with keys model_id, hidden_dim,
    layer_data{n_layers, peak_layer, peak_separation, metrics[...]}."""
    p = root / slug / f"caz_{concept}.json"
    return json.loads(p.read_text())


def dom_matrix(caz: dict, normalize: bool = True) -> np.ndarray:
    """[n_layers, hidden_dim] float64 DOM vectors from a caz JSON."""
    arr = np.array(
        [m["dom_vector"] for m in caz["layer_data"]["metrics"]], dtype=np.float64
    )
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms > 1e-10, norms, 1.0)
    return arr


def peak_layer(caz: dict) -> int:
    return int(caz["layer_data"]["peak_layer"])


def all_region_peaks(slug: str, concept: str, root: Path = MODELS_ROOT) -> list[int]:
    """All detected region peak layers for (model, concept), from gem_<concept>.json
    nodes[].caz_peak. Used for the >3-layers-from-any-peak non-CAZ criterion."""
    p = root / slug / f"gem_{concept}.json"
    if not p.exists():
        return []
    nodes = json.loads(p.read_text()).get("nodes") or []
    return sorted({int(n["caz_peak"]) for n in nodes})


def noncaz_layers(slug: str, concept: str, n_layers: int,
                  min_dist: int = 4, root: Path = MODELS_ROOT) -> list[int]:
    """Layers >3 away from every detected peak of this concept (Table 11 criterion)."""
    peaks = all_region_peaks(slug, concept, root)
    if not peaks:  # fall back to the single caz peak
        peaks = [peak_layer(load_caz(slug, concept, root))]
    return [
        l for l in range(n_layers)
        if all(abs(l - pk) >= min_dist for pk in peaks)
    ]


# ---------------------------------------------------------------------------
# HF transfer
# ---------------------------------------------------------------------------


def hf_download_artifacts(patterns: list[str], local_dir: Path = ROSETTA_DATA) -> None:
    """Pull artifact files matching paper_n250-relative patterns into
    local_dir/paper_n250/..., then expose them under MODELS_ROOT via symlink."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        HF_DATASET, repo_type="dataset", local_dir=str(local_dir),
        allow_patterns=[f"{PAPER_TREE}/{p}" for p in patterns],
    )
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for d in sorted((local_dir / PAPER_TREE).iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            link = MODELS_ROOT / d.name
            if not link.exists():
                link.symlink_to(d)


def hf_upload(job: str, path: Path, retries: int = 5) -> None:
    """Upload one output file to HF under _round3_gpu/<job>/. Retries with
    backoff; raises after exhaustion — callers must NOT swallow the failure
    (upload-before-teardown is a hard rule)."""
    from huggingface_hub import HfApi
    api = HfApi()
    dest = f"{HF_DEST_PREFIX}/{job}/{path.name}"
    for attempt in range(retries):
        try:
            api.upload_file(
                path_or_fileobj=str(path), path_in_repo=dest,
                repo_id=HF_DATASET, repo_type="dataset",
            )
            log.info("[hf] uploaded %s -> %s", path.name, dest)
            return
        except Exception as e:  # noqa: BLE001 — network layer, retry then raise
            wait = 2 ** attempt * 10
            log.warning("[hf] upload failed (%s), retry in %ds", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"HF upload failed after {retries} attempts: {path}")


def hf_verify(job: str, filenames: list[str]) -> None:
    """Re-list the repo and assert every expected file is present (pre-teardown check)."""
    from huggingface_hub import HfApi
    files = set(HfApi().list_repo_files(HF_DATASET, repo_type="dataset"))
    missing = [
        f for f in filenames if f"{HF_DEST_PREFIX}/{job}/{f}" not in files
    ]
    if missing:
        raise RuntimeError(f"HF verification failed, missing: {missing}")
    log.info("[hf] verified %d files for job %s", len(filenames), job)


# ---------------------------------------------------------------------------
# Checkpoint shards
# ---------------------------------------------------------------------------


def shard_path(job: str, key: str) -> Path:
    d = CKPT_ROOT / job
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def shard_done(job: str, key: str) -> dict | None:
    p = shard_path(job, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("[ckpt] corrupt shard %s — recomputing", p)
    return None


def shard_write(job: str, key: str, payload: dict) -> Path:
    p = shard_path(job, key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)
    return p


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def fisher_separation_1d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Fisher separation of 1-D projections (float64)."""
    pos = pos.astype(np.float64)
    neg = neg.astype(np.float64)
    num = abs(pos.mean() - neg.mean())
    den = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float(num / den) if den > 1e-12 else 0.0


def fisher_separation_nd(pos: np.ndarray, neg: np.ndarray) -> float:
    """Trace-normalized Fisher separation, matching caz-validation §2.3:
    ||mu+ - mu-|| / sqrt(0.5*(tr(Sigma+) + tr(Sigma-)))."""
    pos = pos.astype(np.float64)
    neg = neg.astype(np.float64)
    num = float(np.linalg.norm(pos.mean(0) - neg.mean(0)))
    den = float(np.sqrt(0.5 * (pos.var(0, ddof=1).sum() + neg.var(0, ddof=1).sum())))
    return num / den if den > 1e-12 else 0.0


def dom_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Normalised centroid difference (float64 unit vector)."""
    u = pos.mean(0).astype(np.float64) - neg.mean(0).astype(np.float64)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("degenerate DOM direction (zero centroid difference)")
    return u / n
