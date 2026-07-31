#!/usr/bin/env python3
"""Exfiltration full-rerun session — EXFILTRATION_RERUN_SPEC.md §5 steps 1-6.

Runs on a GPU host via the existing `round3-gpu-session` Prefect deployment:
`run_session.py --exfil-rerun [...]` dispatches here (no new flow/deployment
needed; the staged working tree wins over the sparse clone, same as the
round-3 session).

Phases (each shard-checkpointed and --skip-able; safe to resume):

  sync      jamesrahenry staging repos + rosetta_analysis uv env, with the
            deterministic-sampler verification (p4_cluster_f pattern).
  stage     install the corrected 249-pair pool (built by
            build_corrected_exfiltration_pairs.py, shipped in
            exfiltration_rerun_data/) into the host RCP checkout,
            sha256-pinned. Re-verified before every consuming phase.
  download  baseline (defective) caz_exfiltration.json per roster slug +
            caz_causation.json (size ordering + control gate), plus the
            broad caz/gem artifact tree G2/G3/G6 read.
  control   §7.4 sanity gate: fresh gpt2 x causation extraction into a
            scratch --out-root; peak-DOM cosine vs stored must be >= 0.96.
  models    per roster model (union of P3 paper-28 and P4 §2 rosters, §2a):
            purge stale exfiltration artifacts -> extract.py (n=249, bf16,
            Cluster-F budget/8-bit fallback) -> §7.1/§7.2/§7.3 gates ->
            build_gems -> P3 battery (paper-28 only: global_sweep, ablation,
            ablation_gem, random, patch) -> upload (replacement
            paper_n250/<slug>/ + provenance copy, §6/§6-P3) -> purge weights.
  g2        §3 reslice: exfiltration-only split-pair ablation, BASE_28
            roster, merged into the round-3 G2 aggregate on HF.
  g3        §3 reslice: full G3-subset rerun (4 small models, corrected
            artifacts) -> replacement matrix.
  g6        §3 reslice: full C=17 null battery rerun (artifact-only).
  finalize  upload the constructed pool + flip-list sidecar + session
            manifest (§1a.3, §6-P3), final HF verification.

Gate policy (§7): any gate failure raises BEFORE that model's upload —
defective data never lands on HF from this session. Cluster F's stored
exfiltration artifacts were re-extracted 2026-07-16 (corrected labels,
different draw), so the §7.2 anti-correlation expectation INVERTS for
falcon-40b (stored caz already corrected) — encoded below, don't "fix" it.

Written: 2026-07-17 02:55 UTC by claude:exfil-rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    BASE_28, G3_SUBSET, CKPT_ROOT, HF_DATASET, MODELS_ROOT, OUT_ROOT,
    PAPER_TREE, ROSETTA_DATA, dom_matrix, hf_download_artifacts, hf_upload,
    hf_verify, load_caz, log, peak_layer, shard_done, shard_write, slugify,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "exfiltration_rerun_data"
JOB = "exfiltration_rerun"                     # provenance prefix on HF
CONCEPT = "exfiltration"
N_PAIRS = 249

# Pinned by build_corrected_exfiltration_pairs.py (179 flipped / 70 unchanged
# / 1 deleted, twice-verified). If the pool is ever rebuilt, update BOTH.
POOL_SHA256 = "c46957b2835cd0328bceba06acfad36af4bcfc452799b6a6fc0a04cc9c9c99a3"

RCP_ROOT = Path.home() / "Rosetta_Concept_Pairs"
RCP_POOL = RCP_ROOT / "pairs" / "raw" / "v1" / "exfiltration_consensus_pairs.jsonl"
ROSETTA_ANALYSIS = Path.home() / "rosetta_analysis"
BASELINES = ROSETTA_DATA / "exfil_rerun_baselines"
CONTROL_ROOT = ROSETTA_DATA / "exfil_rerun_control"

# --- Rosters (§2/§2a) -------------------------------------------------------
# P3's full paper corpus: BASE_28 (the gemma/opt-350m-excluded round-3 roster,
# 25 models) plus the three exclusions — published tables are 28 rows/concept.
PAPER_28 = BASE_28 + [
    "facebook/opt-350m", "google/gemma-2-2b", "google/gemma-2-9b",
]
assert len(PAPER_28) == 28

CLUSTER_F = {
    "tiiuae/falcon-40b": "comfortable, ~80GB bf16",
    "meta-llama/Llama-3.1-70B": "tight, ~140GB bf16 — 8-bit fallback likely",
    "Qwen/Qwen2.5-72B": "tight, ~144GB bf16 — 8-bit fallback likely",
}

# Models whose stored HF exfiltration caz ALREADY has corrected labels.
# falcon-40b: 2026-07-16 Cluster F backfill (bf16, caz uploaded). The two
# 70B-class models were added 2026-07-17 on gate evidence, not provenance:
# their stored caz shows corrected-level separation (0.42) and +0.99 cosine
# with the fresh corrected extraction — impossible under 71.6%-inverted
# labels — even though BRINGUP_NOTES says their caz "stays bf16" from the
# original extraction. That means the ORIGINAL Cluster F caz extraction
# postdates the RCP label fix. Discrepancy flagged to the science owners
# (Hopper) — do not resolve it by editing this set without checking there.
STORED_ALREADY_CORRECTED = {
    "tiiuae_falcon_40b", "meta_llama_Llama_3.1_70B", "Qwen_Qwen2.5_72B",
}

# §7.1 — relabel-demo expectations (N=250 relabel-in-post; fresh N=249
# extraction must land within 0.05 on each, else stop-and-check).
GATE_SEPARATION = {
    "EleutherAI_pythia_70m": 0.460,
    "openai_community_gpt2_xl": 0.379,
    "Qwen_Qwen2.5_3B": 0.458,
}
GATE_SEPARATION_TOL = 0.05
CONTROL_MODEL, CONTROL_CONCEPT, CONTROL_MIN_COS = "openai-community/gpt2", "causation", 0.96

BF16_BUDGET_S = 14400
BIT8_BUDGET_S = 7200
BATTERY_BUDGET_S = 10800   # per battery script per model; the 7-14B global
                           # sweeps dominate — raise before assuming a hang.

EXFIL_FILES = [
    f"calibration_{CONCEPT}.npy", f"calibration_alllayer_{CONCEPT}.npy",
    f"calibration_{CONCEPT}_meta.json", f"caz_{CONCEPT}.json",
    f"gem_{CONCEPT}.json", f"ablation_global_sweep_{CONCEPT}.json",
    f"ablation_{CONCEPT}.json", f"ablation_gem_{CONCEPT}.json",
    f"ablation_random_{CONCEPT}.json", f"patch_{CONCEPT}.json",
]


def alignment_roster() -> list[str]:
    """P4 §2: pinned ALIGN_ROSTER_30 (27, gemma-excluded) + Cluster F (3)."""
    from common import ALIGN_ROSTER_30
    return list(ALIGN_ROSTER_30) + list(CLUSTER_F)


def union_roster() -> list[str]:
    """§2a (RESOLVED): union of P3 paper-28 and P4 §2 rosters, deduped by slug."""
    seen: dict[str, str] = {}
    for mid in PAPER_28 + alignment_roster():
        seen.setdefault(slugify(mid), mid)
    return list(seen.values())


class TimedOut:
    returncode = -9
    stdout = ""

    def __init__(self, timeout: int):
        self.stderr = f"TimedOut after {timeout}s (killed, not a crash)"


def run(cmd: str, timeout: int, extra_env: dict | None = None):
    log.info("$ %s", cmd[:220])
    env = {**os.environ, **(extra_env or {})}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        log.warning("  exceeded %ds budget — killed", timeout)
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        for line in out.strip().split("\n")[-25:]:
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


# ---------------------------------------------------------------------------
# sync — p4_cluster_f_extraction.py pattern (staging repos, sampler check)
# ---------------------------------------------------------------------------


def sync_repos() -> None:
    for repo, url in [
        ("rosetta_tools", "git@github.com:jamesrahenry/rosetta_tools-staging.git"),
        ("rosetta_analysis", "git@github.com:jamesrahenry/Rosetta_Analysis-staging.git"),
        ("Rosetta_Concept_Pairs", "https://github.com/jamesrahenry/Rosetta_Concept_Pairs.git"),
    ]:
        path = os.path.expanduser(f"~/{repo}")
        if os.path.isdir(f"{path}/.git"):
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
        raise RuntimeError("rosetta_tools failed to install from staging")

    # The runner image doesn't ship uv (yesterday's host had it installed by
    # hand — a "host-side workaround" this script shouldn't depend on).
    r = run("command -v uv || pip install -q uv", 300)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError("uv is unavailable and pip install uv failed")

    run("cd ~/rosetta_analysis && uv sync --frozen -q 2>/dev/null || true", 600)
    r = run("cd ~/rosetta_analysis && uv pip install -e ~/rosetta_tools --reinstall", 300)
    if r.returncode != 0:
        raise RuntimeError("failed to override rosetta_tools in rosetta_analysis venv")
    r = run(
        'cd ~/rosetta_analysis && uv run --no-sync python -c "'
        "import rosetta_tools.dataset as d, inspect; "
        "src = inspect.getsource(d.load_concept_pairs); "
        "assert 'sha256' in src, 'still on the old hash()-based sampler'; "
        "print('deterministic sampler confirmed active')"
        '"', 60)
    if "deterministic sampler confirmed active" not in (r.stdout or ""):
        raise RuntimeError("rosetta_analysis venv still resolves the old "
                           "non-deterministic sampler — refusing to run")
    os.environ["ROSETTA_CONCEPTS_ROOT"] = str(RCP_ROOT / "pairs" / "raw" / "v1")
    log.info("[sync] repos synced, deterministic sampler confirmed")


# ---------------------------------------------------------------------------
# stage — corrected pool into the host RCP checkout (§1a.2)
# ---------------------------------------------------------------------------


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_pool() -> None:
    src = DATA_DIR / "exfiltration_consensus_pairs.jsonl"
    if not src.exists():
        raise RuntimeError(f"{src} missing — run build_corrected_exfiltration_pairs.py "
                           "on the dev box and stage its output here")
    if _sha256(src) != POOL_SHA256:
        raise RuntimeError("staged pool sha256 mismatch vs pinned constant — "
                           "the data files and this script are out of sync")
    RCP_POOL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, RCP_POOL)
    verify_pool()
    log.info("[stage] corrected 249-pair pool installed at %s", RCP_POOL)


def verify_pool() -> None:
    """Re-check before every consuming phase — a git pull --autostash in a
    resumed session can silently revert the replacement file."""
    if not RCP_POOL.exists() or _sha256(RCP_POOL) != POOL_SHA256:
        raise RuntimeError(f"{RCP_POOL} is not the corrected pool (sha256 "
                           "mismatch) — rerun the stage phase")


def expected_manifest() -> list[str]:
    """Composite pair ids in pool order (the §7.3 reference)."""
    ids: list[str] = []
    for line in (DATA_DIR / "exfiltration_consensus_pairs.jsonl").read_text().splitlines():
        rec = json.loads(line)
        cid = f"{rec['pair_id']}__{rec['model_name']}"
        if cid not in ids:
            ids.append(cid)
    assert len(ids) == N_PAIRS
    return ids


# ---------------------------------------------------------------------------
# download — baselines + the artifact tree the reslices read
# ---------------------------------------------------------------------------


def download_baselines(slugs: list[str]) -> None:
    from huggingface_hub import hf_hub_download
    BASELINES.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        for fname in (f"caz_{CONCEPT}.json", "caz_causation.json"):
            dest = BASELINES / slug / fname
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                p = hf_hub_download(HF_DATASET, f"{PAPER_TREE}/{slug}/{fname}",
                                    repo_type="dataset")
                shutil.copy2(p, dest)
            except Exception as e:  # noqa: BLE001
                log.warning("[download] no stored %s for %s (%s) — gates that "
                            "need it will be skipped for this model", fname, slug, e)


def download_artifact_tree() -> None:
    """caz/gem artifacts for the 16 NON-exfiltration concepts only. A
    `caz_*.json` wildcard here clobbered freshly-corrected local
    caz_exfiltration.json files with the stored defective versions on the
    2026-07-17 resume (snapshot_download re-fetches on etag mismatch — and a
    corrected local file always mismatches a defective remote). Exfiltration
    artifacts locally are session-owned: produced by extraction, restored
    only by repair_local_exfiltration()."""
    from common import CONCEPTS_17
    slugs = sorted({slugify(m) for m in BASE_28 + G3_SUBSET + alignment_roster()})
    concepts16 = [c for c in CONCEPTS_17 if c != CONCEPT]
    patterns = [f"{s}/{kind}_{c}.json"
                for s in slugs for c in concepts16 for kind in ("caz", "gem")]
    hf_download_artifacts(patterns)
    # hf_download_artifacts only symlinks a slug dir under MODELS_ROOT when
    # it doesn't exist yet. A slug whose dir was created as a REAL directory
    # by an earlier extraction (e.g. pythia-70m during the smoke run) never
    # gets the link, so downloads land in paper_n250/<slug>/ invisibly —
    # G2's causation-calibration failed on exactly this. Link the individual
    # files across for real dirs.
    snap_root = ROSETTA_DATA / PAPER_TREE
    for slug in slugs:
        dst = MODELS_ROOT / slug
        src = snap_root / slug
        if not src.is_dir() or dst.is_symlink() or not dst.is_dir():
            continue
        for f in list(src.glob("caz_*.json")) + list(src.glob("gem_*.json")):
            if f.name.endswith(f"_{CONCEPT}.json"):
                continue   # exfiltration stays session-owned
            target = dst / f.name
            if not target.exists():
                target.symlink_to(f)


def size_key(slug: str) -> float:
    p = BASELINES / slug / "caz_causation.json"
    if not p.exists():
        return float("inf")   # unknown size: run last, after known-good models
    caz = json.loads(p.read_text())
    return caz["hidden_dim"] * caz["layer_data"]["n_layers"]


# ---------------------------------------------------------------------------
# control — §7.4
# ---------------------------------------------------------------------------


def control_gate() -> None:
    if shard_done(JOB, "control_gate"):
        log.info("[control] already passed — skipping")
        return
    slug = slugify(CONTROL_MODEL)
    baseline = BASELINES / slug / f"caz_{CONTROL_CONCEPT}.json"
    if not baseline.exists():
        raise RuntimeError(f"[control] stored {baseline} missing — cannot gate")
    if CONTROL_ROOT.exists():
        shutil.rmtree(CONTROL_ROOT)
    r = run(
        f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python extraction/extract.py "
        f"--model '{CONTROL_MODEL}' --concepts {CONTROL_CONCEPT} --n-pairs 250 "
        f"--dtype bfloat16 --no-clean-cache --out-root {CONTROL_ROOT}",
        timeout=3600)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError("[control] gpt2 x causation extraction failed")
    fresh = json.loads((CONTROL_ROOT / slug / f"caz_{CONTROL_CONCEPT}.json").read_text())
    stored = json.loads(baseline.read_text())
    cos = _peak_dom_cosine(fresh, stored)
    log.info("[control] gpt2 x causation peak-DOM cosine fresh-vs-stored = %.4f", cos)
    if cos < CONTROL_MIN_COS:
        raise RuntimeError(
            f"[control] §7.4 FAILED: cosine {cos:.4f} < {CONTROL_MIN_COS} — the "
            "env/pipeline does not reproduce the corpus; fix before trusting "
            "any exfiltration delta")
    shard_write(JOB, "control_gate", {"cosine": cos, "utc": time.strftime("%F %T")})


def _peak_dom_cosine(caz_a: dict, caz_b: dict) -> float:
    da = dom_matrix(caz_a)[peak_layer(caz_a)]
    db = dom_matrix(caz_b)[peak_layer(caz_b)]
    return float(np.dot(da, db))


# ---------------------------------------------------------------------------
# models — extract, gate, gem, battery, upload
# ---------------------------------------------------------------------------


def fresh_corrected_present(slug: str) -> bool:
    """True when this model's on-disk exfiltration artifacts already come
    from the reconstructed 249-pair set (a prior session pass extracted them
    but failed later, e.g. at a gate or upload) — re-extracting would waste
    GPU-hours to produce the identical thing."""
    meta_p = MODELS_ROOT / slug / f"calibration_{CONCEPT}_meta.json"
    caz_p = MODELS_ROOT / slug / f"caz_{CONCEPT}.json"
    if not meta_p.exists() or not caz_p.exists():
        return False
    try:
        ids = json.loads(meta_p.read_text())["corpus"]["pair_ids"]
        if set(ids) != set(expected_manifest()):
            return False
        # The meta alone lies if something re-downloaded the stored caz over
        # the corrected one (the 2026-07-17 clobber): a caz byte-identical to
        # the defective baseline is stale regardless of what the meta says.
        # (Legitimately-corrected stored caz — Cluster F — came from a
        # different draw, so byte-identity still means stale.)
        baseline_p = BASELINES / slug / f"caz_{CONCEPT}.json"
        if baseline_p.exists() and baseline_p.read_bytes() == caz_p.read_bytes():
            return False
        return True
    except Exception:  # noqa: BLE001 — corrupt meta means re-extract
        return False


def purge_stored_exfiltration(slug: str) -> None:
    """extract.py skips any model whose caz_<concept>.json already exists at
    >= requested n_pairs — stale (defective) artifacts must go first. The
    baseline copies for §7 gates live in BASELINES, taken at download time."""
    d = MODELS_ROOT / slug
    if not d.exists():
        return
    for pat in (f"*_{CONCEPT}.json", f"*_{CONCEPT}.npy", f"*_{CONCEPT}_meta.json",
                f"*_{CONCEPT}_2*.json"):
        for p in d.glob(pat):
            p.unlink()


def extract_model(mid: str, load_8bit: bool = False) -> bool:
    slug = slugify(mid)
    bit = " --load-8bit" if load_8bit else ""
    budget = BIT8_BUDGET_S if load_8bit else BF16_BUDGET_S
    if mid not in CLUSTER_F:
        budget = min(budget, 7200)
    r = run(
        f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python extraction/extract.py "
        f"--model '{mid}' --concepts {CONCEPT} --n-pairs {N_PAIRS} "
        f"--dtype bfloat16 --no-clean-cache{bit}",
        timeout=budget)
    timed_out = isinstance(r, TimedOut)
    oom = not timed_out and ("CUDA out of memory" in (r.stderr or "")
                             or "OutOfMemoryError" in (r.stderr or ""))
    if timed_out or oom or r.returncode != 0:
        if not load_8bit and (timed_out or oom) and mid in CLUSTER_F:
            log.warning("%s: bf16 attempt failed (%s) — retrying --load-8bit",
                        slug, "budget" if timed_out else "OOM")
            return extract_model(mid, load_8bit=True)
        raise RuntimeError(f"extraction failed for {slug}: {(r.stderr or '')[-600:]}")
    return load_8bit


def gates_for_model(slug: str, manifest_ref: list[str] | None) -> list[str]:
    """§7.1-§7.3 for one freshly-extracted model. Returns the manifest
    (so the first model becomes the cross-model reference). Raises on any
    hard failure — the caller must not upload."""
    meta = json.loads((MODELS_ROOT / slug / f"calibration_{CONCEPT}_meta.json").read_text())
    ids = meta["corpus"]["pair_ids"]

    # §7.3 manifest: exact set vs the reconstruction; exact order vs the
    # first extracted model (cross-model identity is what the papers need).
    if set(ids) != set(expected_manifest()):
        raise RuntimeError(f"[gate §7.3] {slug}: extracted pair_ids != the "
                           "reconstructed 249-pair set")
    if len(ids) != N_PAIRS:
        raise RuntimeError(f"[gate §7.3] {slug}: {len(ids)} pairs, expected {N_PAIRS}")
    if manifest_ref is not None and ids != manifest_ref:
        raise RuntimeError(f"[gate §7.3] {slug}: pair order differs from the "
                           "session's reference model — cross-model comparison broken")
    if ids != expected_manifest():
        log.warning("[gate §7.3] %s: sampler order differs from pool file order "
                    "(set + cross-model order both verified — acceptable, noting "
                    "for the record)", slug)

    fresh = load_caz(slug, CONCEPT)
    new_sep = float(fresh["layer_data"]["peak_separation"])

    baseline_p = BASELINES / slug / f"caz_{CONCEPT}.json"
    if baseline_p.exists():
        stored = json.loads(baseline_p.read_text())
        old_sep = float(stored["layer_data"]["peak_separation"])
        cos = _peak_dom_cosine(fresh, stored)
        ratio = new_sep / old_sep if old_sep > 1e-9 else float("nan")
        log.info("[gate] %s: sep %.3f -> %.3f (%.2fx), peak-DOM cosine %.3f",
                 slug, old_sep, new_sep, ratio, cos)
        if slug in STORED_ALREADY_CORRECTED:
            # stored artifact already label-corrected (2026-07-16 Cluster F
            # backfill) — expect agreement, not inversion (draw differs, so
            # only warn on weak agreement).
            if cos < 0.5:
                log.warning("[gate §7.2] %s: cosine %.3f vs already-corrected "
                            "stored caz is unexpectedly low — flag for review",
                            slug, cos)
        else:
            if cos > 0:
                raise RuntimeError(
                    f"[gate §7.2] {slug}: corrected DOM correlates POSITIVELY "
                    f"({cos:.3f}) with the defective stored DOM — the label "
                    "overlay didn't take")
            if cos > -0.3:
                log.warning("[gate §7.2] %s: anti-correlation weak (%.3f; "
                            "observed reference was -0.72)", slug, cos)
            if not 1.3 <= ratio <= 3.0:
                log.warning("[gate §7.1] %s: separation ratio %.2fx outside the "
                            "rough 1.5-2.5x expectation", slug, ratio)
    else:
        log.warning("[gate] %s: no stored baseline caz — §7.1/§7.2 skipped", slug)

    if slug in GATE_SEPARATION:
        want = GATE_SEPARATION[slug]
        if abs(new_sep - want) > GATE_SEPARATION_TOL:
            raise RuntimeError(
                f"[gate §7.1] {slug}: peak separation {new_sep:.3f} deviates "
                f">{GATE_SEPARATION_TOL} from the relabel-demo value {want:.3f} "
                "— stop and check before anything downstream consumes this")
        log.info("[gate §7.1] %s: %.3f within %.2f of demo %.3f ✓",
                 slug, new_sep, GATE_SEPARATION_TOL, want)
    return ids


def run_battery(mid: str) -> None:
    """§4-P3 B-E through the original pipeline, exfiltration slice only."""
    slug = slugify(mid)
    cmds = [
        ("global_sweep", f"gem/ablate_global_sweep.py --model '{mid}' "
                         f"--concepts {CONCEPT} --overwrite --no-clean-cache"),
        ("ablation", f"gem/ablate.py --model '{mid}' --concepts {CONCEPT} --force"),
        ("ablation_gem", f"gem/ablate_gem.py --model '{mid}' --concepts {CONCEPT} "
                         "--compare-peak --width 3 --no-clean-cache"),
        ("random", f"gem/ablate_random_direction.py --model '{mid}' "
                   f"--concepts {CONCEPT} --overwrite --no-clean-cache"),
        ("patch", f"gem/patch.py --model '{mid}' --concepts {CONCEPT} "
                  "--force --no-clean-cache"),
    ]
    outputs = {
        "global_sweep": f"ablation_global_sweep_{CONCEPT}.json",
        "ablation": f"ablation_{CONCEPT}.json",
        "ablation_gem": f"ablation_gem_{CONCEPT}.json",
        "random": f"ablation_random_{CONCEPT}.json",
        "patch": f"patch_{CONCEPT}.json",
    }
    for name, cmd in cmds:
        key = f"battery_{slug}_{name}"
        # Shard alone isn't proof: purge_stored_exfiltration deletes battery
        # JSONs before a re-extraction, so a mop-up run must re-run any step
        # whose output file is gone (2026-07-17 resume bug).
        if shard_done(JOB, key) and (MODELS_ROOT / slug / outputs[name]).exists():
            continue
        r = run(f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python {cmd}",
                timeout=BATTERY_BUDGET_S)
        if isinstance(r, TimedOut) or r.returncode != 0:
            raise RuntimeError(f"battery step {name} failed for {slug}: "
                               f"{(getattr(r, 'stderr', '') or '')[-500:]}")
        shard_write(JOB, key, {"utc": time.strftime("%F %T")})


def upload_model(slug: str, used_8bit: bool, battery: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    d = MODELS_ROOT / slug
    expected = EXFIL_FILES if battery else EXFIL_FILES[:5]
    files = [d / f for f in expected if (d / f).exists()]
    missing = [f for f in expected if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"{slug}: expected outputs missing before upload: {missing}")
    if used_8bit:
        log.warning("%s: extracted via 8-bit fallback — uploading anyway (the "
                    "label correction supersedes precision parity; recorded in "
                    "the provenance manifest — disclose per §2's precision-split "
                    "open item before precision-sensitive use)", slug)
    for f in files:
        for dest in (f"{PAPER_TREE}/{slug}/{f.name}",
                     f"{PAPER_TREE}/_round3_gpu/{JOB}/{slug}/{f.name}"):
            for attempt in range(5):
                try:
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=dest,
                                    repo_id=HF_DATASET, repo_type="dataset")
                    break
                except Exception as e:  # noqa: BLE001
                    wait = 2 ** attempt * 10
                    log.warning("%s: upload %s failed (%s), retry in %ds",
                                slug, dest, e, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"{slug}: upload failed after retries: {dest}")
    # Verify by SIZE, not just presence — HF has had network trouble in the
    # last 24h (James 2026-07-17); a truncated upload that still lists would
    # otherwise pass. Nothing local gets cleaned up unless this returns.
    dests = [d for f in files
             for d in (f"{PAPER_TREE}/{slug}/{f.name}",
                       f"{PAPER_TREE}/_round3_gpu/{JOB}/{slug}/{f.name}")]
    info = {p.path: p for p in api.get_paths_info(HF_DATASET, dests, repo_type="dataset")}
    for f in files:
        want = f.stat().st_size
        for dest in (f"{PAPER_TREE}/{slug}/{f.name}",
                     f"{PAPER_TREE}/_round3_gpu/{JOB}/{slug}/{f.name}"):
            got = info.get(dest)
            got_size = getattr(got, "size", None) or getattr(getattr(got, "lfs", None), "size", None)
            if got is None:
                raise RuntimeError(f"{slug}: HF verification failed — {dest} not on repo")
            if got_size is not None and got_size != want:
                raise RuntimeError(f"{slug}: HF verification failed — {dest} is "
                                   f"{got_size} bytes, local is {want} (truncated upload?)")
    log.info("%s: %d files uploaded + verified (replacement + provenance)",
             slug, len(files))


def purge_weights(mid: str) -> None:
    """Free the HF weight cache — 39 models won't fit on one disk.

    The cache location is env-dependent: the gpu-runner job template injects
    HF_HOME under /workspace (so weights persist across flow runs), while
    Path.home() in this process is /root — the 2026-07-17 disk-full incident
    was this function purging the wrong (empty) directory while 641GB piled
    up in /workspace/.cache. Check every plausible location, loudly report
    what was actually freed, and warn if nothing was found anywhere."""
    frag = "models--" + mid.replace("/", "--")
    bases = []
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        bases.append(Path(HF_HUB_CACHE))
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("HF_HOME"):
        bases.append(Path(os.environ["HF_HOME"]) / "hub")
    bases += [Path("/workspace/.cache/huggingface/hub"),
              Path.home() / ".cache" / "huggingface" / "hub"]
    freed = False
    for base in dict.fromkeys(bases):   # dedupe, keep order
        for p in base.glob(frag):
            log.info("[purge] removing %s", p)
            shutil.rmtree(p, ignore_errors=True)
            freed = True
    if not freed:
        log.warning("[purge] no cached weights found for %s in %s — if disk "
                    "fills, find where this env actually caches", mid,
                    [str(b) for b in dict.fromkeys(bases)])


def models_phase(smoke: bool) -> None:
    verify_pool()
    roster = union_roster()
    battery_slugs = {slugify(m) for m in PAPER_28}
    if smoke:
        roster = ["EleutherAI/pythia-70m"]

    # Gate models first (fail fast on §7.1), then ascending size.
    gate_first = [m for m in roster if slugify(m) in GATE_SEPARATION]
    rest = sorted((m for m in roster if slugify(m) not in GATE_SEPARATION),
                  key=lambda m: size_key(slugify(m)))
    ordered = gate_first + rest

    state_p = CKPT_ROOT / JOB / "manifest_ref.json"
    manifest_ref = json.loads(state_p.read_text())["ids"] if state_p.exists() else None

    failures: list[tuple[str, str]] = []
    for i, mid in enumerate(ordered, 1):
        slug = slugify(mid)
        if shard_done(JOB, f"model_{slug}" + ("_smoke" if smoke else "")):
            log.info("[models] %s already complete — skipping", slug)
            continue
        log.info("[models] === %d/%d %s ===", i, len(ordered), mid)
        t0 = time.time()
        try:
            if fresh_corrected_present(slug):
                # bf16 unless a fallback warning appears in the pass that
                # produced these files (none has, as of 2026-07-17).
                log.info("[models] %s: corrected artifacts already on disk "
                         "(prior pass) — skipping re-extraction", slug)
                used_8bit = False
            else:
                purge_stored_exfiltration(slug)
                used_8bit = extract_model(mid)
            ids = gates_for_model(slug, manifest_ref)
            if manifest_ref is None:
                manifest_ref = ids
                shard_write(JOB, "manifest_ref", {"ids": ids, "model": slug})
            r = run(f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python "
                    f"gem/build_gems.py --model {slug} --concepts {CONCEPT} --force",
                    timeout=1800)
            if isinstance(r, TimedOut) or r.returncode != 0:
                raise RuntimeError(f"build_gems failed: {(getattr(r, 'stderr', '') or '')[-400:]}")
            do_battery = slug in battery_slugs and not smoke
            if do_battery:
                run_battery(mid)
            if not smoke:
                upload_model(slug, used_8bit, battery=do_battery)
            shard_write(JOB, f"model_{slug}" + ("_smoke" if smoke else ""),
                        {"used_8bit": used_8bit, "battery": do_battery,
                         "elapsed_s": time.time() - t0})
            # Weight-cache purge ONLY on full success (incl. verified upload):
            # HF is flaky right now (James 2026-07-17) — a failed model keeps
            # its weights so the mop-up relaunch doesn't re-download 10s of GB
            # through the same unreliable network. Local outputs are never
            # deleted regardless.
            if not smoke:
                purge_weights(mid)
        except Exception as e:  # noqa: BLE001 — collect, keep going (G2 lesson)
            log.error("[models] %s FAILED: %s: %s", slug, type(e).__name__, e)
            failures.append((slug, f"{type(e).__name__}: {e}"))
        log.info("[models] %s done in %.0fmin", slug, (time.time() - t0) / 60)

    if failures:
        for slug, err in failures:
            log.error("[models] FAILED: %s — %s", slug, err)
        raise RuntimeError(f"{len(failures)}/{len(ordered)} models failed; "
                           "completed models are checkpointed — fix and resume "
                           "with the same command")


# ---------------------------------------------------------------------------
# reslices (§3)
# ---------------------------------------------------------------------------


def repair_local_exfiltration() -> None:
    """Restore corrected exfiltration caz/gem into MODELS_ROOT for every
    reslice-roster slug, force-downloaded from HF main (which holds this
    session's verified uploads). Needed once after the 2026-07-17 clobber
    (see download_artifact_tree docstring) and harmless any other time.
    Refuses if a roster slug has no completion shard — that means HF main
    for it is NOT this session's corrected upload."""
    from huggingface_hub import hf_hub_download
    slugs = sorted({slugify(m) for m in BASE_28 + G3_SUBSET + alignment_roster()})
    for slug in slugs:
        if not shard_done(JOB, f"model_{slug}"):
            raise RuntimeError(f"[repair] {slug} has no completion shard — its "
                               "HF caz may be stale; finish the models phase first")
        d = MODELS_ROOT / slug
        d.mkdir(parents=True, exist_ok=True)
        for fname in (f"caz_{CONCEPT}.json", f"gem_{CONCEPT}.json"):
            p = hf_hub_download(HF_DATASET, f"{PAPER_TREE}/{slug}/{fname}",
                                repo_type="dataset", force_download=True)
            shutil.copy2(p, d / fname)
        baseline_p = BASELINES / slug / f"caz_{CONCEPT}.json"
        if (baseline_p.exists() and slug not in STORED_ALREADY_CORRECTED
                and baseline_p.read_bytes() == (d / f"caz_{CONCEPT}.json").read_bytes()):
            raise RuntimeError(f"[repair] {slug}: HF main caz is still the "
                               "defective baseline — upload didn't land?")
    log.info("[repair] corrected exfiltration caz/gem restored for %d slugs", len(slugs))


def reslice_g2() -> None:
    """Exfiltration-only G2 across BASE_28, merged into the round-3 aggregate."""
    verify_pool()
    import g2_split_pair_ablation as g2
    from forward_utils import load_model, release
    g2.JOB = "g2_exfil"   # shard namespace only; uploads happen in the merge

    for mid in sorted(BASE_28, key=lambda m: size_key(slugify(m))):
        if shard_done("g2_exfil", slugify(mid)):
            continue
        model, tok, device = None, None, None
        try:
            model, tok, device = load_model(mid)
            # offset calibration on causation (stable), not exfiltration —
            # see run_for_model's calib_concept docstring.
            g2.run_for_model(mid, model, tok, device, 32, [CONCEPT],
                             calib_concept="causation")
        finally:
            if model is not None:
                release(model)
        purge_weights(mid)

    # merge: fetch the round-3 aggregate, swap exfiltration rows, re-summarize
    from huggingface_hub import hf_hub_download
    agg_p = hf_hub_download(HF_DATASET, f"{PAPER_TREE}/_round3_gpu/g2/g2_split_pair_results.json",
                            repo_type="dataset")
    agg = json.loads(Path(agg_p).read_text())
    kept = [r for r in agg["rows"] if r["concept"] != CONCEPT]
    fresh: list[dict] = []
    for s in sorted((CKPT_ROOT / "g2_exfil").glob("*.json")):
        if not s.stem.endswith("_smoke"):
            fresh.extend(json.loads(s.read_text())["rows"])
    if not fresh:
        raise RuntimeError("[g2] no exfiltration rows produced — nothing to merge")
    rows = kept + fresh

    def agg_stat(role: str, field: str) -> dict:
        v = [r[field] for r in rows if r["role"] == role]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "median": float(np.median(v)) if v else None}

    agg["rows"] = rows
    agg["n_rows"] = len(rows)
    agg["exfiltration_correction"] = {
        "spec": "EXFILTRATION_RERUN_SPEC.md §3", "n_pairs": N_PAIRS,
        "pool_sha256": POOL_SHA256, "rerun_utc": time.strftime("%F %T UTC"),
        "note": "exfiltration rows regenerated from the corrected 249-pair set; "
                "halves are 124/125 (was 125/125 at N=250)",
    }
    agg["summary"] = {
        "peak_heldout_nd": agg_stat("peak", "reduction_heldout_nd"),
        "peak_insample_nd": agg_stat("peak", "reduction_insample_nd"),
        "control_heldout_nd": agg_stat("control", "reduction_heldout_nd"),
        "control_insample_nd": agg_stat("control", "reduction_insample_nd"),
    }
    p = agg["summary"]["peak_heldout_nd"]["mean"]
    c = agg["summary"]["control_heldout_nd"]["mean"]
    if p is not None and c:
        agg["summary"]["heldout_peak_over_control_ratio"] = p / c

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "g2_split_pair_results.json"
    out.write_text(json.dumps(agg, indent=1))
    hf_upload("g2", out)                      # replacement, same path as round-3
    hf_verify("g2", [out.name])
    hf_upload(JOB, out)                       # provenance copy
    log.info("[g2] merged aggregate uploaded (%d rows, %d fresh exfiltration)",
             len(rows), len(fresh))


def reslice_g3() -> None:
    """Full G3-subset rerun on corrected artifacts (33 of 289 cells change;
    rerunning all keeps every cell on one code path + one artifact state)."""
    verify_pool()
    import g3_cross_concept_matrix as g3
    from common import CONCEPTS_17
    from forward_utils import load_model, release
    for mid in G3_SUBSET:
        model = None
        try:
            model, tok, device = load_model(mid)
            g3.run_for_model(mid, model, tok, device, 32, CONCEPTS_17)
        finally:
            if model is not None:
                release(model)
        purge_weights(mid)
    g3.finalize()
    src = OUT_ROOT / "g3_cross_concept_matrix.json"
    if src.exists():
        hf_upload(JOB, src)                   # provenance copy


def reslice_g6(n_seeds: int = 3) -> None:
    verify_pool()
    r = run(f"cd {HERE} && {sys.executable} g6_c17_null_battery.py --n-seeds {n_seeds}",
            timeout=6 * 3600)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError("[g6] rerun failed — see log above")


# ---------------------------------------------------------------------------
# finalize — §1a.3 / §6-P3 provenance
# ---------------------------------------------------------------------------


def finalize(smoke: bool) -> None:
    if smoke:
        log.info("[finalize] smoke — skipping uploads")
        return
    for name in ("exfiltration_consensus_pairs.jsonl", "exfiltration_flip_list.json"):
        hf_upload(JOB, DATA_DIR / name)
    manifest = {
        "spec": "EXFILTRATION_RERUN_SPEC.md", "session_utc": time.strftime("%F %T UTC"),
        "n_pairs": N_PAIRS, "pool_sha256": POOL_SHA256,
        "roster": {slugify(m): m for m in union_roster()},
        "battery_roster": sorted(slugify(m) for m in PAPER_28),
        "shards": sorted(p.stem for p in (CKPT_ROOT / JOB).glob("*.json")),
    }
    mp = OUT_ROOT / "exfiltration_rerun_manifest.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1))
    hf_upload(JOB, mp)
    hf_verify(JOB, ["exfiltration_consensus_pairs.jsonl",
                    "exfiltration_flip_list.json",
                    "exfiltration_rerun_manifest.json"])
    log.info("[finalize] provenance uploads verified. P4 §4 recomputes "
             "(Procrustes refit + downstream) start from these artifacts — "
             "separate task, see the spec's §5.7-8.")


PHASES = ["sync", "stage", "download", "control", "models", "repair",
          "g2", "g3", "g6", "finalize"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="pythia-70m end-to-end: extract + all gates + gem, "
                         "no battery, no reslices, no uploads")
    ap.add_argument("--skip", nargs="*", default=[], choices=PHASES)
    ap.add_argument("--g6-seeds", type=int, default=3)
    args = ap.parse_args()

    skip = set(args.skip)
    if args.smoke:
        skip |= {"g2", "g3", "g6", "finalize"}

    if "sync" not in skip:
        sync_repos()
    else:
        os.environ.setdefault("ROSETTA_CONCEPTS_ROOT",
                              str(RCP_ROOT / "pairs" / "raw" / "v1"))
    if "stage" not in skip:
        stage_pool()
    if "download" not in skip:
        download_baselines(sorted({slugify(m) for m in union_roster()}))
        if not args.smoke:
            download_artifact_tree()
    if "control" not in skip:
        control_gate()
    if "models" not in skip:
        models_phase(args.smoke)
    if "repair" not in skip and not args.smoke:
        repair_local_exfiltration()
    if "g2" not in skip:
        reslice_g2()
    if "g3" not in skip:
        reslice_g3()
    if "g6" not in skip:
        reslice_g6(args.g6_seeds)
    if "finalize" not in skip:
        finalize(args.smoke)
    log.info("[session] EXFILTRATION RERUN COMPLETE — do not tear down until "
             "the P4-side recompute owner confirms artifact pickup.")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
