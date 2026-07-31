#!/usr/bin/env python3
"""Permuted-label Procrustes null for the GEM handoff alignment (P4 Test 5).

C10-null of ../ROUND3_COMPUTE_PLAN.md (Hopper t2d4606e): regenerate
`prh_gem_handoff_null.csv` against the CORRECTED handoff-layer pipeline. This
is the chance baseline for `prh_gem_handoff_primary.csv` (Test 5's cross-family,
same-dimension handoff alignment, grand mean aligned cosine 0.9609). It shows
that when the sample correspondence between two models' calibration activations
is shuffled *before* the Procrustes fit, aligned cosine collapses to ~0 — i.e.
the 0.96 alignment is not an artifact of the underdetermined rotation.

WHY THIS IS A GPU JOB (it was a CPU item): the null refits an orthogonal
Procrustes rotation for every (pair x trial). At 25 trials x ~1660 cross-family
same-dim pairs that is ~41,500 d-dimensional SVDs (d up to 5120). On the CPU
backfill box that ran ~3.5 h for ONE of 17 concepts (~50 h projected). Here the
25 trials of a pair are stacked into a single batched `torch.linalg.svd` on the
GPU, turning the whole run into minutes.

RE-ANALYSIS ONLY — no forward passes, no model weights. Inputs are the stored
`gem_<concept>.json` (deepest-node handoff layer) and
`calibration_alllayer_<concept>.npy` (per-layer calibration activations) on HF
`james-ra-henry/Rosetta-Activations` @ revision `paper-n250` — the exact files
`regen_gem_handoff_primary.py` used, so pairs correspond 1:1 with the primary.

ROSTER NOTE: this uses the handoff-primary's own 30-model roster (PRIMARY_MODELS
below), which INCLUDES google/gemma-2-9b(-it) and facebook/opt-350m. That is
deliberate and differs from `common.ALIGN_ROSTER_30` (which excludes them per
the 2026-07-16 Gemma-instability / opt-350m-dim decisions). A null must cover
exactly the pairs of the primary it calibrates, and the published
`prh_gem_handoff_primary.csv` artifact predates those exclusions and contains
gemma/opt-350m pairs. Do not "fix" this to common's roster without also
rebuilding the primary.

Output: `prh_gem_handoff_null.csv`, schema
  concept, source, target, trial, null_aligned_cosine
uploaded to REPLACE the stale artifact at
  paper_n250/_prh_gem_handoff/prh_gem_handoff_null.csv   (revision main)
with a provenance copy + summary under paper_n250/_round3_gpu/handoff_null/.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from common import (
    CKPT_ROOT, HF_DATASET, OUT_ROOT, PAPER_TREE, hf_upload, hf_verify, log,
    shard_done, shard_write,
)

JOB = "handoff_null"
N_TRIALS = 25              # matches the stale artifact (trial index 0..24)
GLOBAL_SEED = 42           # deterministic, reproducible per pair
INPUT_REVISION = "paper-n250"
CANONICAL_DEST = f"{PAPER_TREE}/_prh_gem_handoff/prh_gem_handoff_null.csv"
OUT_CSV = OUT_ROOT / "prh_gem_handoff_null.csv"
SUMMARY = OUT_ROOT / "handoff_null_summary.json"
ROSETTA_DATA = Path(os.environ.get("ROSETTA_DATA", Path.home() / "rosetta_data"))
LOCAL = ROSETTA_DATA / PAPER_TREE

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# Handoff-primary roster (see ROSTER NOTE in docstring). 30 models.
PRIMARY_MODELS = [
    "openai_community_gpt2", "EleutherAI_gpt_neo_125m",
    "EleutherAI_pythia_160m", "facebook_opt_125m",
    "openai_community_gpt2_medium", "facebook_opt_350m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "facebook_opt_1.3b",
    "meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B",
    "EleutherAI_pythia_2.8b", "facebook_opt_2.7b", "microsoft_phi_2",
    "EleutherAI_pythia_6.9b", "facebook_opt_6.7b",
    "meta_llama_Llama_3.1_8B", "mistralai_Mistral_7B_v0.3",
    "Qwen_Qwen2.5_7B", "google_gemma_2_9b",
    "EleutherAI_pythia_12b", "Qwen_Qwen2.5_14B", "Qwen_Qwen2.5_32B",
    "Qwen_Qwen2.5_3B_Instruct", "Qwen_Qwen2.5_7B_Instruct",
    "meta_llama_Llama_3.2_1B_Instruct", "meta_llama_Llama_3.1_8B_Instruct",
    "mistralai_Mistral_7B_Instruct_v0.3", "google_gemma_2_9b_it",
]

FAMILY = {
    "openai_community_gpt2": "GPT-2", "openai_community_gpt2_medium": "GPT-2",
    "EleutherAI_gpt_neo_125m": "GPT-Neo",
    "EleutherAI_pythia_160m": "Pythia", "EleutherAI_pythia_410m": "Pythia",
    "EleutherAI_pythia_1b": "Pythia", "EleutherAI_pythia_1.4b": "Pythia",
    "EleutherAI_pythia_2.8b": "Pythia", "EleutherAI_pythia_6.9b": "Pythia",
    "EleutherAI_pythia_12b": "Pythia",
    "facebook_opt_125m": "OPT", "facebook_opt_350m": "OPT",
    "facebook_opt_1.3b": "OPT", "facebook_opt_2.7b": "OPT", "facebook_opt_6.7b": "OPT",
    "meta_llama_Llama_3.2_1B": "Llama 3", "meta_llama_Llama_3.2_1B_Instruct": "Llama 3",
    "meta_llama_Llama_3.1_8B": "Llama 3", "meta_llama_Llama_3.1_8B_Instruct": "Llama 3",
    "Qwen_Qwen2.5_3B": "Qwen 2.5", "Qwen_Qwen2.5_3B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_7B": "Qwen 2.5", "Qwen_Qwen2.5_7B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_14B": "Qwen 2.5", "Qwen_Qwen2.5_32B": "Qwen 2.5",
    "microsoft_phi_2": "Phi-2",
    "mistralai_Mistral_7B_v0.3": "Mistral", "mistralai_Mistral_7B_Instruct_v0.3": "Mistral",
    "google_gemma_2_9b": "Gemma 2", "google_gemma_2_9b_it": "Gemma 2",
}

HF_ID = {
    "openai_community_gpt2": "openai-community/gpt2",
    "openai_community_gpt2_medium": "openai-community/gpt2-medium",
    "EleutherAI_gpt_neo_125m": "EleutherAI/gpt-neo-125m",
    "EleutherAI_pythia_160m": "EleutherAI/pythia-160m",
    "EleutherAI_pythia_410m": "EleutherAI/pythia-410m",
    "EleutherAI_pythia_1b": "EleutherAI/pythia-1b",
    "EleutherAI_pythia_1.4b": "EleutherAI/pythia-1.4b",
    "EleutherAI_pythia_2.8b": "EleutherAI/pythia-2.8b",
    "EleutherAI_pythia_6.9b": "EleutherAI/pythia-6.9b",
    "EleutherAI_pythia_12b": "EleutherAI/pythia-12b",
    "facebook_opt_125m": "facebook/opt-125m",
    "facebook_opt_350m": "facebook/opt-350m",
    "facebook_opt_1.3b": "facebook/opt-1.3b",
    "facebook_opt_2.7b": "facebook/opt-2.7b",
    "facebook_opt_6.7b": "facebook/opt-6.7b",
    "meta_llama_Llama_3.2_1B": "meta-llama/Llama-3.2-1B",
    "meta_llama_Llama_3.2_1B_Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta_llama_Llama_3.1_8B": "meta-llama/Llama-3.1-8B",
    "meta_llama_Llama_3.1_8B_Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5_3B": "Qwen/Qwen2.5-3B",
    "Qwen_Qwen2.5_3B_Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5_7B": "Qwen/Qwen2.5-7B",
    "Qwen_Qwen2.5_7B_Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5_14B": "Qwen/Qwen2.5-14B",
    "Qwen_Qwen2.5_32B": "Qwen/Qwen2.5-32B",
    "microsoft_phi_2": "microsoft/phi-2",
    "mistralai_Mistral_7B_v0.3": "mistralai/Mistral-7B-v0.3",
    "mistralai_Mistral_7B_Instruct_v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "google_gemma_2_9b": "google/gemma-2-9b",
    "google_gemma_2_9b_it": "google/gemma-2-9b-it",
}
assert set(HF_ID) == set(PRIMARY_MODELS) == set(FAMILY), "roster dict mismatch"


# ---------------------------------------------------------------------------
# Device / batched Procrustes
# ---------------------------------------------------------------------------


def pick_device(requested: str) -> str:
    import torch
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    log.warning("[handoff_null] CUDA unavailable — falling back to CPU (slow)")
    return "cpu"


def _svd_batched(M):
    """Batched orthogonal factor R = U @ Vh of M = svd(M), with a per-slice
    gesvd/CPU fallback for the rare non-converged slice (mirrors g5's driver
    fallback — same math, robust driver)."""
    import torch
    try:
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return U @ Vh
    except Exception as e:  # noqa: BLE001 — one bad slice shouldn't kill the batch
        log.warning("[handoff_null] batched SVD failed (%s); per-slice fallback", e)
        outs = []
        for i in range(M.shape[0]):
            try:
                U, _, Vh = torch.linalg.svd(M[i], full_matrices=False)
                outs.append(U @ Vh)
            except Exception:
                # last resort: numpy gesvd on this one slice
                from scipy.linalg import svd as _svd
                u, _, vt = _svd(M[i].detach().cpu().numpy(), lapack_driver="gesvd")
                outs.append(torch.from_numpy(u @ vt).to(M.device, M.dtype))
        return torch.stack(outs)


def permuted_null_cosines(src_acts: np.ndarray, tgt_acts: np.ndarray,
                          dom_src: np.ndarray, dom_tgt: np.ndarray,
                          n_trials: int, rng: np.random.Generator,
                          device: str, dtype, trial_chunk: int) -> list[float]:
    """For a same-dim pair, return `n_trials` permuted-label aligned cosines.

    Each trial permutes the row correspondence of the target calibration
    activations, then fits R minimising ||tgt_c[perm] @ R - src_c|| (orthogonal
    Procrustes, R = U @ Vh of tgt_c[perm]^T @ src_c) and reports
    cos(dom_src, dom_tgt @ R). Shuffling correspondence destroys the alignment,
    so the cosines cluster at ~0. Trials are batched through one GPU SVD.
    """
    import torch
    n = min(src_acts.shape[0], tgt_acts.shape[0])
    src = torch.as_tensor(src_acts[:n], device=device, dtype=dtype)
    tgt = torch.as_tensor(tgt_acts[:n], device=device, dtype=dtype)
    src_c = src - src.mean(0, keepdim=True)
    tgt_c = tgt - tgt.mean(0, keepdim=True)
    dsrc = torch.as_tensor(dom_src, device=device, dtype=dtype)
    dtgt = torch.as_tensor(dom_tgt, device=device, dtype=dtype)
    dsrc = dsrc / dsrc.norm().clamp_min(1e-12)

    out: list[float] = []
    for start in range(0, n_trials, trial_chunk):
        k = min(trial_chunk, n_trials - start)
        perms = np.stack([rng.permutation(n) for _ in range(k)])          # [k,n]
        tp = tgt_c[torch.as_tensor(perms, device=device)]                 # [k,n,d]
        M = tp.transpose(-1, -2) @ src_c                                  # [k,d,d]
        R = _svd_batched(M)                                               # [k,d,d]
        aligned = dtgt @ R                                                # [k,d]
        num = (aligned * dsrc).sum(-1)
        den = aligned.norm(dim=-1).clamp_min(1e-12) * dsrc.norm().clamp_min(1e-12)
        out.extend((num / den).detach().cpu().tolist())
    return out


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def ensure(model: str, fname: str) -> Path:
    p = LOCAL / model / fname
    if not p.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_DATASET, f"{PAPER_TREE}/{model}/{fname}",
                        repo_type="dataset", revision=INPUT_REVISION,
                        local_dir=str(ROSETTA_DATA))
    return p


def dom_at(acts: np.ndarray, layer: int) -> np.ndarray:
    sl = acts[layer]
    half = sl.shape[0] // 2
    d = sl[:half].mean(0) - sl[half:].mean(0)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else d


def load_concept_models(concept: str) -> dict:
    """Extract corrected handoff-layer DOM + calibration acts for every roster
    model that has this concept's artifacts. Identical extraction to
    regen_gem_handoff_primary.py (deepest node by caz_end)."""
    per_model = {}
    for model in PRIMARY_MODELS:
        npy_p = None
        try:
            gem = json.loads(ensure(model, f"gem_{concept}.json").read_text())
            npy_p = ensure(model, f"calibration_alllayer_{concept}.npy")
            acts = np.load(npy_p)
            n_layers = acts.shape[0]
            deepest = max(gem["nodes"], key=lambda nd: nd["caz_end"])
            handoff = min(deepest["handoff_layer"], n_layers - 1)
            per_model[model] = {
                "dom_h": dom_at(acts, handoff),
                "acts_h": acts[handoff].astype(np.float32),
                "handoff": handoff, "hidden_dim": int(acts.shape[-1]),
            }
            del acts
        except Exception as e:  # noqa: BLE001 — a missing model is reported, not fatal
            log.warning("[handoff_null] SKIP %s %s: %s", model, concept, e)
        finally:
            if npy_p is not None:
                try:
                    os.remove(npy_p)  # calibration npys are large; don't hoard
                except OSError:
                    pass
    return per_model


# ---------------------------------------------------------------------------
# Per-concept null (checkpointed)
# ---------------------------------------------------------------------------


def run_concept(concept: str, device: str, dtype, n_trials: int,
                trial_chunk: int, pair_base: int) -> tuple[list[dict], int]:
    """Returns (rows, n_pairs). Deterministic per-pair seeds so the whole run is
    reproducible regardless of ordering; pair_base offsets the global counter."""
    per_model = load_concept_models(concept)
    rows, pair_idx = [], pair_base
    for src in PRIMARY_MODELS:
        if src not in per_model:
            continue
        for tgt in PRIMARY_MODELS:
            if src == tgt or tgt not in per_model or FAMILY[src] == FAMILY[tgt]:
                continue
            a, b = per_model[src], per_model[tgt]
            if a["dom_h"].shape[0] != b["dom_h"].shape[0]:
                continue  # same-dim only, matching the primary
            rng = np.random.default_rng(GLOBAL_SEED + pair_idx)
            pair_idx += 1
            cosines = permuted_null_cosines(
                a["acts_h"], b["acts_h"], a["dom_h"], b["dom_h"],
                n_trials, rng, device, dtype, trial_chunk)
            for trial, c in enumerate(cosines):
                rows.append({
                    "concept": concept, "source": HF_ID[src], "target": HF_ID[tgt],
                    "trial": trial, "null_aligned_cosine": float(c),
                })
    return rows, pair_idx - pair_base


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def pair_base_for(concept: str, device: str, dtype, n_trials: int,
                  trial_chunk: int, counts: dict) -> int:
    """Global pair counter up to (not including) `concept`, from cached counts so
    per-pair seeds are stable across resumes."""
    base = 0
    for c in CONCEPTS:
        if c == concept:
            break
        base += counts[c]
    return base


def run(concepts: list[str], device: str, dtype_name: str, n_trials: int,
        trial_chunk: int, smoke: bool) -> None:
    import torch
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # First pass: pair counts per concept (needed for stable per-pair seeds on
    # resume). Cheap — derived from same-dim cross-family roster combinatorics,
    # but computed from the actual loaded models to honour any SKIPs.
    t_start = time.time()
    counts: dict[str, int] = {}
    all_rows: list[dict] = []
    pair_base = 0
    for concept in concepts:
        cached = shard_done(JOB, concept + ("_smoke" if smoke else ""))
        if cached is not None:
            counts[concept] = cached["n_pairs"]
            all_rows.extend(cached["rows"])
            pair_base += cached["n_pairs"]
            log.info("[handoff_null] %s cached (%d pairs) — skipping", concept,
                     cached["n_pairs"])
            continue
        t0 = time.time()
        rows, n_pairs = run_concept(concept, device, dtype, n_trials,
                                    trial_chunk, pair_base)
        counts[concept] = n_pairs
        pair_base += n_pairs
        all_rows.extend(rows)
        shard_write(JOB, concept + ("_smoke" if smoke else ""),
                    {"n_pairs": n_pairs, "n_trials": n_trials, "rows": rows})
        vals = np.array([r["null_aligned_cosine"] for r in rows]) if rows else np.array([0.0])
        log.info("[handoff_null] [%d/%d] %s: %d pairs x %d trials  mean=%.4f sd=%.4f  (%.0fs)",
                 concepts.index(concept) + 1, len(concepts), concept, n_pairs,
                 n_trials, vals.mean(), vals.std(), time.time() - t0)

    # Assemble CSV (schema matches the stale artifact exactly).
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "source", "target",
                                          "trial", "null_aligned_cosine"])
        w.writeheader()
        w.writerows(all_rows)
    vals = np.array([r["null_aligned_cosine"] for r in all_rows], dtype=np.float64)
    summary = {
        "job": JOB, "n_rows": len(all_rows), "n_trials_per_pair": n_trials,
        "n_pairs": int(sum(counts.values())), "n_concepts": len(concepts),
        "roster": "handoff-primary 30-model roster (incl. gemma-2-9b/-it, "
                  "opt-350m) — matches prh_gem_handoff_primary.csv 1:1",
        "method": "permuted-label (shuffled sample correspondence) orthogonal "
                  "Procrustes null on corrected handoff-layer DOM vectors; "
                  "batched torch.linalg.svd",
        "input_revision": INPUT_REVISION, "seed": GLOBAL_SEED, "dtype": dtype_name,
        "grand_mean": float(vals.mean()), "grand_sd": float(vals.std()),
        "abs_max": float(np.abs(vals).max()),
        "primary_reference_mean": 0.9609,
        "elapsed_s": time.time() - t_start,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    log.info("[handoff_null] DONE: %d rows, grand mean=%.5f sd=%.5f abs_max=%.4f (%.0fs)",
             len(all_rows), summary["grand_mean"], summary["grand_sd"],
             summary["abs_max"], summary["elapsed_s"])

    if smoke:
        log.info("[handoff_null] smoke — no HF upload")
        return

    # Sanity gate before publishing (permuted null must sit near zero).
    if abs(summary["grand_mean"]) > 0.05 or summary["grand_sd"] > 0.15:
        raise RuntimeError(
            f"null sanity failed (mean={summary['grand_mean']:.4f}, "
            f"sd={summary['grand_sd']:.4f}); refusing to upload")

    # 1) Replace the stale canonical artifact the paper's S3 cites.
    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=str(OUT_CSV), path_in_repo=CANONICAL_DEST,
        repo_id=HF_DATASET, repo_type="dataset", revision="main",
        commit_message="C10-null: regenerate permuted-label handoff null vs "
                       "fixed pipeline (GPU batched SVD; t2d4606e)")
    log.info("[handoff_null] uploaded canonical -> %s", CANONICAL_DEST)
    # 2) Provenance copy + summary under the round-3 GPU tree.
    hf_upload(JOB, OUT_CSV)
    hf_upload(JOB, SUMMARY)
    hf_verify(JOB, [OUT_CSV.name, SUMMARY.name])
    # 3) Confirm the canonical file is actually there.
    files = set(HfApi().list_repo_files(HF_DATASET, repo_type="dataset", revision="main"))
    if CANONICAL_DEST not in files:
        raise RuntimeError(f"canonical upload not visible: {CANONICAL_DEST}")
    log.info("[handoff_null] verified canonical artifact present")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cuda (default) or cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="SVD/statistics precision (default float64, matches pipeline)")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help=f"trials per pair (default {N_TRIALS}, matches stale artifact)")
    ap.add_argument("--trial-chunk", type=int, default=25,
                    help="trials per batched SVD call (lower for tight GPU memory)")
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="subset of concepts (default all 17)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 concepts, 3 trials, no upload")
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.smoke:
        run(CONCEPTS[:2], device, args.dtype, n_trials=3,
            trial_chunk=min(3, args.trial_chunk), smoke=True)
        return
    concepts = args.concepts or CONCEPTS
    run(concepts, device, args.dtype, args.trials, args.trial_chunk, smoke=False)


if __name__ == "__main__":
    main()
