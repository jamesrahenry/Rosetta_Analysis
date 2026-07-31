#!/usr/bin/env python3
"""Exfiltration multimodal + CKA backfill — the two battery artifacts the
main rerun session missed (P3 review catch, 2026-07-17).

EXFILTRATION_RERUN_SPEC §4-P3 C's "per-CAZ multimodal protocol" is
`gem/ablate_multimodal.py` -> ablation_multimodal_<c>.json (feeds the §6.4
divergence population's 476 cells), and cka_<c>.json comes from
`caz/cka_validation.py`. The session's battery ran global_sweep/ablation/
ablation_gem/random/patch only — this mode regenerates the two missing
artifacts from the corrected data, for EXACTLY the (model, concept) cells
that existed before (coverage is deliberately uneven across the corpus;
expanding it is a science-owner call, not a runner call):

  * ablation_multimodal_exfiltration.json — 12 slugs
  * cka_exfiltration.json                 — 19 slugs   (union: 24)

Pool state: stages the 249-pair replacement pool itself (idempotent), same
as the main session — these artifacts subsample (n=50 / n=100) from the
paper-snapshot pool via the deterministic sampler, exactly like the other
16 concepts'. Safe to run before or after the rcp_v1 full-pool mode: each
mode sets its own pool state at start and the worker serializes runs.

Prereqs per slug (force-downloaded here): corrected caz/gem exfiltration
from HF main — hard-fails if a slug's corrected caz isn't on HF yet.

No weight purging: this is designed to run AFTER the scratch-disk cache
migration (5TB at /workspace/scratch), and its downloads deliberately
pre-warm the cache for the rcp_v1 extraction that follows.

Via Prefect: run_session.py --mm-cka [...] on the round3-gpu-session
deployment. Pinned slug lists below were read from HF 2026-07-17 13:05 UTC.

Written: 2026-07-17 13:10 UTC by claude:exfil-rerun (P3 review follow-up;
Hopper tf08f424 thread)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from common import (
    HF_DATASET, MODELS_ROOT, PAPER_TREE, hf_upload, log, shard_done,
    shard_write,
)
from exfiltration_rerun_session import (
    BASELINES, CONCEPT, ROSETTA_ANALYSIS, TimedOut, run, stage_pool,
    sync_repos, verify_pool,
)

JOB = "exfil_mm_cka"
PROV = "exfiltration_rerun"          # same provenance area as the session

# Pinned from HF 2026-07-17: the cells that existed (defective) and must be
# replaced. Do NOT extend these lists here — coverage is a science call.
MM_SLUGS = [
    "EleutherAI_pythia_12b", "EleutherAI_pythia_2.8b", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_6.9b", "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_14B",
    "Qwen_Qwen2.5_3B", "facebook_opt_350m", "facebook_opt_6.7b",
    "google_gemma_2_2b", "google_gemma_2_9b", "openai_community_gpt2_xl",
]
CKA_SLUGS = [
    "EleutherAI_pythia_1.4b", "EleutherAI_pythia_160m", "EleutherAI_pythia_1b",
    "EleutherAI_pythia_2.8b", "EleutherAI_pythia_410m", "EleutherAI_pythia_70m",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B",
    "Qwen_Qwen2.5_7B", "facebook_opt_1.3b", "facebook_opt_125m",
    "facebook_opt_2.7b", "facebook_opt_350m", "google_gemma_2_2b",
    "meta_llama_Llama_3.2_1B", "openai_community_gpt2",
    "openai_community_gpt2_large", "openai_community_gpt2_medium",
]
STEP_BUDGET_S = 7200


def model_id_of(slug: str) -> str:
    """HF model id from the corrected caz on HF main (also our prereq check)."""
    from huggingface_hub import hf_hub_download
    d = MODELS_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    for fname in (f"caz_{CONCEPT}.json", f"gem_{CONCEPT}.json"):
        for attempt in range(4):
            try:
                p = hf_hub_download(HF_DATASET, f"{PAPER_TREE}/{slug}/{fname}",
                                    repo_type="dataset", force_download=True)
                break
            except Exception as e:  # noqa: BLE001 — HF has been flaky tonight
                if attempt == 3:
                    raise
                wait = 2 ** attempt * 10
                log.warning("%s: download of %s failed (%s), retry in %ds",
                           slug, fname, e, wait)
                time.sleep(wait)
        shutil.copy2(p, d / fname)
    caz = json.loads((d / f"caz_{CONCEPT}.json").read_text())
    baseline = BASELINES / slug / f"caz_{CONCEPT}.json"
    if baseline.exists() and baseline.read_bytes() == (d / f"caz_{CONCEPT}.json").read_bytes():
        raise RuntimeError(f"{slug}: HF main caz is still the defective baseline "
                           "— the rerun session's upload is missing?")
    model_id = caz["model_id"]
    # detect_manifolds.find_extraction_dir (which ablate_multimodal.py uses)
    # locates a model's directory by scanning for run_summary.json{model_id},
    # NOT by slug path — it silently no-ops (exit 0, no output file) for any
    # slug that never got a full local extraction on THIS host (large models
    # whose exfiltration-only extraction here never wrote one). Stub one in
    # if genuinely absent — never overwrite a real one from an actual
    # extraction run, which carries real per-concept results.
    summary_p = d / "run_summary.json"
    if not summary_p.exists():
        summary_p.write_text(json.dumps({
            "model_id": model_id, "concepts": [CONCEPT],
            "note": "stub written by exfiltration_mm_cka_backfill.py — only "
                    "satisfies find_extraction_dir()'s lookup, carries no "
                    "extraction results of its own",
        }))
        log.info("%s: stubbed run_summary.json (no local extraction record "
                "existed for find_extraction_dir to find)", slug)
    return model_id


def mm_region_count(slug: str) -> int:
    """Number of CAZ regions ablate_multimodal.py's own find_caz_regions_scored
    detects for this slug's corrected exfiltration caz. The multimodal
    protocol needs >=2 (it builds an N-region interaction matrix) and
    silently no-ops — exit 0, no output file — for exactly 1 region. That
    silent-no-op is indistinguishable from an actual bug from the outside
    (both look like 'exited 0, file missing'), which is why this needs its
    own explicit check rather than inferring from the subprocess result
    (2026-07-17 incident: two real, unrelated bugs already hid behind that
    exact same symptom for this script)."""
    from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
    caz = json.loads((MODELS_ROOT / slug / f"caz_{CONCEPT}.json").read_text())
    metrics = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"],
                            m["velocity"]) for m in caz["layer_data"]["metrics"]]
    return find_caz_regions_scored(metrics).n_regions


def run_step(slug: str, model_id: str, kind: str) -> Path | None:
    """Returns None for a legitimate not-applicable outcome (mm, <2 regions)
    — never for an actual failure, which always raises."""
    if kind == "mm":
        n_regions = mm_region_count(slug)
        if n_regions < 2:
            log.info("%s: exfiltration has only %d CAZ region post-correction "
                     "(multimodal needs >=2) — not applicable, not a failure",
                     slug, n_regions)
            return None
        cmd = (f"gem/ablate_multimodal.py --model '{model_id}' "
               f"--concepts {CONCEPT} --no-clean-cache")
        out = MODELS_ROOT / slug / f"ablation_multimodal_{CONCEPT}.json"
    else:
        cmd = (f"caz/cka_validation.py --model '{model_id}' "
               f"--concepts {CONCEPT} --overwrite")
        out = MODELS_ROOT / slug / f"cka_{CONCEPT}.json"
    if out.exists():
        out.unlink()   # stale defective copy must not satisfy anything
    r = run(f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python {cmd}",
            timeout=STEP_BUDGET_S)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError(f"{kind} failed for {slug}: "
                           f"{(getattr(r, 'stderr', '') or '')[-500:]}")
    if not out.exists():
        raise RuntimeError(f"{kind} for {slug}: exited 0 but {out.name} missing "
                           "(NOT a known not-applicable case — investigate "
                           "before assuming this is benign)")
    return out


def upload_files(slug: str, files: list[Path]) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    dests = [(f, d) for f in files
             for d in (f"{PAPER_TREE}/{slug}/{f.name}",
                       f"{PAPER_TREE}/_round3_gpu/{PROV}/{slug}/{f.name}")]
    for f, dest in dests:
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(f), path_in_repo=dest,
                                repo_id=HF_DATASET, repo_type="dataset")
                break
            except Exception as e:  # noqa: BLE001
                time.sleep(2 ** attempt * 10)
                log.warning("%s: upload %s failed (%s), retrying", slug, dest, e)
        else:
            raise RuntimeError(f"{slug}: upload failed after retries: {dest}")
    info = {p.path: p for p in api.get_paths_info(
        HF_DATASET, [d for _, d in dests], repo_type="dataset")}
    for f, dest in dests:
        got = info.get(dest)
        size = getattr(got, "size", None) or getattr(getattr(got, "lfs", None), "size", None)
        if got is None or (size is not None and size != f.stat().st_size):
            raise RuntimeError(f"{slug}: size-verification failed for {dest}")
    log.info("%s: %d uploads size-verified", slug, len(dests))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of slugs (default: the pinned union)")
    ap.add_argument("--skip-sync", action="store_true")
    args = ap.parse_args()

    if not args.skip_sync:
        sync_repos()
    stage_pool()
    verify_pool()

    union = sorted(set(MM_SLUGS) | set(CKA_SLUGS))
    if args.models:
        union = [s for s in union if s in set(args.models)]
    log.info("[mm-cka] %d slugs (%d multimodal, %d cka)",
             len(union), len(MM_SLUGS), len(CKA_SLUGS))

    failures: list[tuple[str, str]] = []
    for i, slug in enumerate(union, 1):
        if shard_done(JOB, slug):
            log.info("[mm-cka] %s already complete — skipping", slug)
            continue
        log.info("[mm-cka] === %d/%d %s ===", i, len(union), slug)
        t0 = time.time()
        try:
            model_id = model_id_of(slug)
            produced: list[Path] = []
            not_applicable: list[str] = []
            if slug in MM_SLUGS:
                r = run_step(slug, model_id, "mm")
                (produced.append(r) if r is not None else not_applicable.append("mm"))
            if slug in CKA_SLUGS:
                r = run_step(slug, model_id, "cka")
                (produced.append(r) if r is not None else not_applicable.append("cka"))
            if produced:
                upload_files(slug, produced)
            shard_write(JOB, slug, {"files": [p.name for p in produced],
                                    "not_applicable": not_applicable,
                                    "elapsed_s": time.time() - t0})
        except Exception as e:  # noqa: BLE001 — collect and continue
            log.error("[mm-cka] %s FAILED: %s: %s", slug, type(e).__name__, e)
            failures.append((slug, f"{type(e).__name__}: {e}"))
        log.info("[mm-cka] %s done in %.0fmin", slug, (time.time() - t0) / 60)

    not_applicable_summary = {}
    for slug in union:
        shard = shard_done(JOB, slug)
        if shard and shard.get("not_applicable"):
            not_applicable_summary[slug] = shard["not_applicable"]

    manifest = {
        "job": JOB, "utc": time.strftime("%F %T UTC"),
        "mm_slugs": MM_SLUGS, "cka_slugs": CKA_SLUGS, "failures": failures,
        "not_applicable": not_applicable_summary,
        "note": "regenerates exactly the pre-existing (defective) multimodal/"
                "cka exfiltration cells from corrected data; coverage "
                "unchanged by design (P3 review catch, spec §4-P3 C). "
                "not_applicable: multimodal legitimately requires >=2 CAZ "
                "regions (interaction-matrix protocol) — correction can "
                "reduce a model's region count below that threshold, in "
                "which case the pre-existing (defective-label) cell has no "
                "valid corrected replacement and is intentionally absent, "
                "not a failure.",
    }
    from common import OUT_ROOT
    mp = OUT_ROOT / "exfil_mm_cka_manifest.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1))
    hf_upload(PROV, mp)

    if failures:
        for slug, err in failures:
            log.error("[mm-cka] FAILED: %s — %s", slug, err)
        raise RuntimeError(f"{len(failures)}/{len(union)} slugs failed — "
                           "relaunch resumes from shards")
    log.info("[mm-cka] COMPLETE — all multimodal/cka exfiltration cells "
             "regenerated and verified.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
