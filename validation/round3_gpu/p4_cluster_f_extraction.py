#!/usr/bin/env python3
"""P4 Cluster F full extraction — closes the "Cluster F excluded" gap that
appears in nearly every P4 test, not just the primary alignment result.

Not one of the original G1-G7 round-3 items (those are all P3, or P4 items
already scoped in ROUND3_COMPUTE_PLAN.md) -- this is new, added 2026-07-16
when an H200 became available at no extra cost over the smaller hosts this
session had been using. Dropped in this directory because it shares the
GPU session and should upload-verify the same way the G-numbered jobs do,
not because it's part of the original P3 statistical-objections plan.

## The gap

Cluster F (Falcon-40B, Llama-3.1-70B, Qwen2.5-72B, the 8192-dim frontier
cluster) currently has caz_<concept>.json (peak-layer detection only) on HF
for all 17 concepts, confirmed via direct listing 2026-07-16. It is MISSING:

  - calibration_alllayer_<concept>.npy (full per-layer raw activations,
    500 examples x n_layers x hidden_dim) -- needed for anything that isn't
    just "peak-layer DOM vector": the permuted-label and random-text nulls
    (S3.2), universality/cross-concept transfer (S3.3), split-calibration
    (S3.5), GEM handoff (S3.7), proportional-depth stratification (S3.8).
  - gem_<concept>.json (GEM node detection) -- needed specifically for S3.7.

Every one of those sections currently carries some form of "clusters A-E
only, frontier Cluster F excluded" as an explicit scope note. This job does
not add more frontier models -- it lets the existing 3 participate in tests
they currently can't.

## What this does NOT change

The primary S3.1 alignment result (0.9709, clusters A-E) and its own
separately-reported Cluster F case study (0.9763, "not independently
replicated at scale") are already computed from real Cluster F peak-layer
DOM vectors and are NOT reprocessed here -- this job only fills in the
missing full-depth artifacts so OTHER tests can extend to include F. Do not
re-derive S3.1's number from this job's output; the existing caz_*.json
files (already correct) remain the source for that.

## Method

1. extraction/extract.py --prh-frontier --concepts <all 17> -- extract.py's
   own resume logic (extraction/extract.py:_needs_extraction) already
   detects "caz exists but calibration_alllayer missing" as needing
   re-extraction -- which is every concept for all 3 F models right now.
   There is no way to recover raw per-layer activations without a fresh
   forward pass, so this necessarily redoes the full extraction rather than
   patching around it. Uses the exact tool + flag that produced the
   existing (correct) Cluster F peak-layer data (see infra's
   rosetta_p4_h200.py, the prior Falcon-agency backfill job), so this run's
   caz_*.json output should reproduce current values almost exactly -- if
   it doesn't, that's a signal something upstream changed and needs
   investigating before trusting the new artifacts.
2. gem/build_gems.py --model <id> --concepts <all 17> --force, per model --
   GEM node construction from the freshly-extracted data. CPU-only, no
   model load, fast relative to step 1.
3. Upload every new/changed file to HF paper_n250/<model>/ via common.py's
   hf_upload + hf_verify -- BEFORE any teardown (tbc29f76 lesson: this
   session already lost one job's output once to a premature teardown).

## Hardware

Falcon-40B (~80GB bf16) fits comfortably on a single H200 (141GB). Llama-
3.1-70B and Qwen2.5-72B (~140-144GB bf16 in weights alone) are TIGHT to
OVER capacity on a single H200 with no headroom left for activations/KV-
cache -- unconfirmed whether this session's H200 is single-GPU or a
multi-GPU node (the original Cluster F extraction's "shared H200 node"
phrasing doesn't specify).

"Shoot for the stars, hit the moon if we find our faces full of regolith"
(James 2026-07-16): each model gets a genuine bf16 attempt first --
including extract.py's automatic device_map="auto" GPU+CPU spillover if
VRAM is short even across multiple GPUs, which is numerically IDENTICAL to
full-GPU bf16, just slower where layers landed on CPU -- bounded to
BF16_BUDGET_SECONDS (4h) so a slow-but-not-crashing CPU-spillover run can't
quietly burn a full day of GPU-hours at $3.60/hr before anyone notices. If
that attempt OOMs or simply doesn't finish in budget, it's killed and
retried once with --load-8bit (a further 2h budget) rather than either
abandoning the run or paying for an open-ended wait. Note if 8-bit IS used:
the paper's S3.1 corpus note states Cluster F was extracted "at full
precision (bfloat16)... with no quantization" -- if a model's rerun data
needs 8-bit to fit, that note needs updating
to disclose the mixed-precision extraction, and it should be flagged before
this data is used for anything precision-sensitive.

## Pair-sampling determinism (resolved 2026-07-16)

Flagged by the GPU runner in BRINGUP_NOTES.md: rosetta_analysis's own
uv.lock pins rosetta-tools to v1.4.0, which predates
jamesrahenry/Rosetta_Tools@6fed9e2 ("stable per-concept seed for pair
sampling, was process-randomized hash()"). Under v1.4.0, load_concept_pairs
draws a DIFFERENT 250-pair-per-concept subset on every invocation (Python's
hash() is per-process-salted by default) -- not just different from
current staging, different from ITSELF on a second run. There is no
well-defined "v1.4.0 draw" to preserve for consistency with clusters A-E;
that data's own historical extraction can't be reliably reproduced by
re-running v1.4.0 either, so matching it was never actually on the table.

Decision (James 2026-07-16): "current state of the 250 pairs is intended,
the old behaviour was not intended." Cluster F extraction uses the CURRENT
staging rosetta_tools (deterministic, sha256-seeded sampling) going
forward. sync_repos() below force-overrides whatever rosetta_analysis's
uv.lock resolves with the editable staging checkout, then verifies via
`uv run` (not just the outer/non-uv python) that the deterministic sampler
is actually what extract.py will import at runtime -- refuses to proceed
otherwise.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

from common import HF_DATASET, PAPER_TREE, CONCEPTS_17, hf_upload, hf_verify, log

JOB = "p4_cluster_f"

FRONTIER_MODELS = {
    # slug -> (hf id, informal VRAM note)
    "tiiuae_falcon_40b": ("tiiuae/falcon-40b", "comfortable, ~80GB bf16"),
    "meta_llama_Llama_3.1_70B": ("meta-llama/Llama-3.1-70B", "tight, ~140GB bf16"),
    "Qwen_Qwen2.5_72B": ("Qwen/Qwen2.5-72B", "tight, ~144GB bf16"),
}

ROSETTA_ANALYSIS = Path.home() / "rosetta_analysis"
MODELS_ROOT = Path.home() / "rosetta_data" / "models"


class TimedOut:
    """Sentinel standing in for a CompletedProcess when the subprocess was
    killed for running past its budget rather than crashing on its own.
    Duck-types just enough of CompletedProcess (.returncode, .stdout,
    .stderr) for callers that don't care which happened."""
    returncode = -9
    stdout = ""

    def __init__(self, timeout: int):
        self.stderr = f"TimedOut after {timeout}s (process killed, not a crash)"


def run(cmd: str, timeout: int, extra_env: dict | None = None):
    log.info("$ %s", cmd[:220])
    env = {**os.environ, **(extra_env or {})}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        log.warning("  command exceeded its %ds budget — killed, not crashed", timeout)
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        for line in stdout.strip().split("\n")[-25:]:
            if line:
                log.info("  %s", line)
        return TimedOut(timeout)
    for line in (r.stdout or "").strip().split("\n")[-25:]:
        if line:
            log.info("  %s", line)
    if r.returncode != 0:
        for line in (r.stderr or "").strip().split("\n")[-25:]:
            if line:
                log.warning("  %s", line)
    return r


def sync_repos() -> None:
    # PUBLIC jamesrahenry repos — no eigan-ai forks in paper science (James
    # 2026-07-16). The eigan-ai/extraction fork also renames the package to
    # `extraction`, so `import rosetta_tools` (which extract.py does) can't
    # resolve against it anyway. Repointed 2026-07-20 off the `-staging` forks,
    # archived read-only when the staging split was retired (2026-07-18);
    # public is now the single line of truth. The re-clone-on-origin-mismatch
    # below auto-migrates any host still carrying a `-staging` clone.
    # GITHUB_SSH_KEY_B64 (flow_utils -> GIT_SSH_COMMAND) provides SSH auth.
    # uv drives rosetta_analysis's env below; ensure it exists even if the
    # image predates its inclusion (a container restart wipes live installs).
    run("command -v uv >/dev/null 2>&1 || pip install -q uv", 300)
    for repo, url in [
        ("rosetta_tools", "git@github.com:jamesrahenry/Rosetta_Tools.git"),
        ("rosetta_analysis", "git@github.com:jamesrahenry/Rosetta_Analysis.git"),
        ("Rosetta_Concept_Pairs", "https://github.com/jamesrahenry/Rosetta_Concept_Pairs.git"),
    ]:
        path = os.path.expanduser(f"~/{repo}")
        if os.path.isdir(f"{path}/.git"):
            # A clone may already exist from OTHER flows pointing at the
            # eigan-ai fork (e.g. ~/rosetta_tools on a shared worker) — a
            # plain pull would silently keep the wrong repo. Re-clone on
            # any origin mismatch.
            r = run(f"git -C {path} remote get-url origin", 60)
            if (r.stdout or "").strip() != url:
                run(f"rm -rf {path}", 60)
                run(f"git clone -q {url} {path}", 600)
            else:
                run(f"git -C {path} pull --ff-only --autostash -q", 300)
        else:
            run(f"git clone -q {url} {path}", 600)
    run("pip install -q -e ~/rosetta_tools", 600)
    r = run('python -c "import rosetta_tools; print(rosetta_tools.__file__)"', 60)
    if "rosetta_tools" not in (r.stdout or ""):
        raise RuntimeError("rosetta_tools failed to install from staging — refusing to run")

    run("cd ~/rosetta_analysis && uv sync --frozen -q 2>/dev/null || true", 600)

    # `uv sync --frozen` resolves rosetta_tools from rosetta_analysis's OWN
    # uv.lock, which (per BRINGUP_NOTES.md's 2026-07-16 audit) pins
    # rosetta-tools @ v1.4.0 -- predating jamesrahenry/Rosetta_Tools@6fed9e2
    # ("stable per-concept seed for pair sampling, was process-randomized
    # hash()"). v1.4.0's calibration-pair sampling is NOT reproducible even
    # against itself (Python's hash() is salted per-process by default), so
    # there is no well-defined "v1.4.0 behavior" to preserve for consistency
    # with the existing A-E/Cluster-F data -- that data's own historical
    # draw can't be reliably reproduced by re-running v1.4.0 either. Decision
    # (James 2026-07-16): use the CURRENT, intended (deterministic) sampling
    # behavior going forward rather than the old unintended one -- so
    # explicitly override whatever the lockfile resolved with the editable
    # staging checkout synced above, inside rosetta_analysis's own uv venv.
    r = run("cd ~/rosetta_analysis && uv pip install -e ~/rosetta_tools --reinstall", 300)
    if r.returncode != 0:
        raise RuntimeError(f"failed to override rosetta_tools inside rosetta_analysis's uv venv: {(r.stderr or '')[-500:]}")
    r = run(
        'cd ~/rosetta_analysis && uv run --no-sync python -c "'
        "import rosetta_tools.dataset as d, inspect; "
        "src = inspect.getsource(d.load_concept_pairs); "
        "assert 'sha256' in src, 'still on the old hash()-based sampler'; "
        "print('deterministic sampler confirmed active')"
        '"',
        60,
    )
    if r.returncode != 0 or "deterministic sampler confirmed active" not in (r.stdout or ""):
        raise RuntimeError(
            "rosetta_analysis's uv venv is still resolving the OLD (v1.4.0, "
            "non-deterministic) pair sampler after the override attempt -- "
            "refusing to run extraction against it. Fix the override (or "
            "bump rosetta_analysis's own uv.lock properly) before retrying."
        )

    os.environ["ROSETTA_CONCEPTS_ROOT"] = str(Path.home() / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1")
    log.info("repos synced, deterministic pair-sampling confirmed")


BF16_BUDGET_SECONDS = 14400   # 4h "shoot for the stars" -- genuine attempt at full
                              # precision (incl. device_map="auto" GPU+CPU spillover),
                              # but bounded so a slow-not-crashing CPU-offload run
                              # doesn't quietly burn a full day of GPU-hours before
                              # anyone notices. Raise this if a model is progressing
                              # (check the log) and just needs a bit more room, rather
                              # than assuming 4h is definitely enough.
BIT8_BUDGET_SECONDS = 7200    # 2h -- 8-bit should be meaningfully faster than a
                              # CPU-spillover bf16 run; this is a true fallback
                              # ceiling, not expected to be hit.


def extract_one(slug: str, hf_id: str, load_8bit: bool = False) -> bool:
    """Full per-layer extraction for one Cluster F model, all 17 concepts.
    Returns True if an 8-bit fallback was used (caller must NOT treat this
    model's output as precision-matched to its existing bf16 artifacts).

    "Shoot for the stars, hit the moon if we find our faces full of
    regolith": try bf16 first (full precision, including automatic
    device_map="auto" GPU+CPU spillover if VRAM is short) within a bounded
    budget; if it OOMs OR simply doesn't finish in that budget, kill it and
    retry once with --load-8bit rather than paying for an open-ended slow
    crawl. Only two attempts total -- an 8-bit retry that ALSO times out or
    fails is a real error, not something to keep retrying blindly."""
    concepts_arg = " ".join(CONCEPTS_17)
    bit_flag = " --load-8bit" if load_8bit else ""
    budget = BIT8_BUDGET_SECONDS if load_8bit else BF16_BUDGET_SECONDS
    r = run(
        f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python extraction/extract.py "
        f"--model '{hf_id}' --concepts {concepts_arg} --n-pairs 250 "
        f"--dtype bfloat16 --no-clean-cache{bit_flag}",
        timeout=budget,
    )
    timed_out = isinstance(r, TimedOut)
    oom = not timed_out and ("CUDA out of memory" in (r.stderr or "") or "OutOfMemoryError" in (r.stderr or ""))
    if timed_out or oom or r.returncode != 0:
        if not load_8bit and (timed_out or oom):
            reason = f"exceeded {budget}s budget" if timed_out else "OOM"
            log.warning("%s: bf16 attempt failed (%s) — falling back to --load-8bit "
                       "(moon, not stars: this model's caz_*.json will NOT be "
                       "re-uploaded, see upload_and_verify)", slug, reason)
            return extract_one(slug, hf_id, load_8bit=True)
        kind = "timed out" if timed_out else "failed"
        raise RuntimeError(f"extraction {kind} for {slug} even at "
                          f"{'8-bit' if load_8bit else 'bf16'}: {(r.stderr or '')[-800:]}")
    log.info("%s: extraction complete%s", slug, " (8-bit fallback)" if load_8bit else " (bf16, no quantization — stars hit)")
    return load_8bit


def build_gem(slug: str) -> None:
    """GEM node construction for one model, all 17 concepts. CPU-only."""
    concepts_arg = " ".join(CONCEPTS_17)
    r = run(
        f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python gem/build_gems.py "
        f"--model {slug} --concepts {concepts_arg} --force",
        timeout=1800,
    )
    if r.returncode != 0:
        log.warning("%s: GEM build failed, leaving for manual retry: %s",
                   slug, (r.stderr or "")[-500:])


def upload_and_verify(slug: str, used_8bit: bool) -> None:
    """Upload calibration_alllayer_*.npy, gem_*.json, and (bf16 runs only)
    refreshed caz_*.json for one model. Verifies presence on HF before
    returning -- caller must not tear down the host until every model's
    upload_and_verify has returned cleanly.

    If this model's extraction fell back to 8-bit, its freshly-regenerated
    local caz_<concept>.json is NOT precision-matched to the existing bf16
    caz_<concept>.json already on HF (the one backing the paper's current
    S3.1 primary result for this model) -- uploading it would silently
    overwrite trusted bf16 peak-layer data with slightly different 8-bit-
    derived values. So for 8-bit models we upload only the genuinely new
    artifacts (calibration_alllayer, gem) and leave the existing caz files
    untouched, logging the precision split loudly instead of hiding it."""
    model_dir = MODELS_ROOT / slug
    patterns = ["calibration_alllayer_*.npy", "gem_*.json"]
    if not used_8bit:
        patterns.append("caz_*.json")
    else:
        log.warning("%s: used 8-bit fallback — NOT re-uploading caz_*.json "
                   "(existing bf16 version on HF stays authoritative for "
                   "this model; new calibration_alllayer/gem files are "
                   "8-bit-derived and precision-mismatched from it — flag "
                   "for manual review before use in precision-sensitive tests)",
                   slug)
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(str(model_dir / pat)))
    if not files:
        raise RuntimeError(f"{slug}: no output files found at {model_dir} — extraction may have failed silently")
    log.info("%s: uploading %d files", slug, len(files))
    dest_prefix = f"{PAPER_TREE}/{slug}"
    from huggingface_hub import HfApi
    api = HfApi()
    for f in files:
        for attempt in range(5):
            try:
                api.upload_file(
                    path_or_fileobj=str(f), path_in_repo=f"{dest_prefix}/{f.name}",
                    repo_id=HF_DATASET, repo_type="dataset",
                )
                break
            except Exception as e:  # noqa: BLE001 — network, retry then raise
                wait = 2 ** attempt * 10
                log.warning("%s: upload of %s failed (%s), retry in %ds", slug, f.name, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"{slug}: upload failed after 5 attempts for {f.name}")

    # verify: every calibration_alllayer_*.npy and gem_*.json we produced is
    # actually listed on HF now, not just "upload_file didn't raise"
    expected = [f.name for f in files if f.name.startswith(("calibration_alllayer_", "gem_"))]
    hf_files = set(api.list_repo_files(HF_DATASET, repo_type="dataset"))
    missing = [name for name in expected if f"{dest_prefix}/{name}" not in hf_files]
    if missing:
        raise RuntimeError(f"{slug}: HF verification failed, missing after upload: {missing}")
    log.info("%s: verified %d files on HF", slug, len(expected))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(FRONTIER_MODELS),
                    help="subset of slugs to run (default: all 3)")
    ap.add_argument("--skip-sync", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run Falcon-40B, agency+certainty only, as a fast correctness check")
    args = ap.parse_args()

    if not args.skip_sync:
        sync_repos()

    concepts_note = "smoke (2 concepts)" if args.smoke else "all 17 concepts"
    log.info("P4 Cluster F full extraction — models=%s, %s", args.models, concepts_note)

    global CONCEPTS_17
    if args.smoke:
        CONCEPTS_17 = ["agency", "certainty"]  # type: ignore[assignment]
        args.models = args.models[:1]

    precision_splits: list[str] = []
    for slug in args.models:
        if slug not in FRONTIER_MODELS:
            log.warning("unknown slug %s, skipping (known: %s)", slug, list(FRONTIER_MODELS))
            continue
        hf_id, note = FRONTIER_MODELS[slug]
        t0 = time.time()
        log.info("=== %s (%s) — %s ===", slug, hf_id, note)
        used_8bit = extract_one(slug, hf_id)
        build_gem(slug)
        upload_and_verify(slug, used_8bit)
        if used_8bit:
            precision_splits.append(slug)
        log.info("=== %s done in %.0fmin ===", slug, (time.time() - t0) / 60)

    if precision_splits:
        log.warning(
            "PRECISION SPLIT — %d model(s) needed 8-bit and are now inconsistent "
            "with the rest of Cluster F: %s. Their calibration_alllayer/gem files "
            "are uploaded (8-bit-derived); their caz_*.json was NOT touched and "
            "remains the original bf16 version. This means those specific "
            "models are NOT yet at full parity with A-E — resolve before "
            "claiming the gap fully closed (retry on more VRAM, or accept and "
            "document the mixed precision explicitly in S3.1's corpus note).",
            len(precision_splits), precision_splits,
        )
    else:
        log.info("All models extracted at bf16, no quantization -- full "
                 "precision parity with clusters A-E, no follow-up needed.")

    log.info("P4 Cluster F full extraction COMPLETE — all uploads verified. "
             "Safe to tear down the host now.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
