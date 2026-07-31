#!/usr/bin/env python3
"""rcp_v1 exfiltration re-extraction — corrected labels, full pool, EXTRACTION ONLY.

Regenerates the rcp_v1 (richest-line) exfiltration artifacts for every slug
that currently carries them on HF: calibration_alllayer_exfiltration.npy,
calibration_exfiltration.npy, calibration_exfiltration_meta.json,
caz_exfiltration.json. The stored versions were extracted from the
pre-correction pool (71.6%-inverted labels, t50c6362); this run extracts
from the CURRENT corrected RCP pool — the full 962 pairs via
``--split all``, NOT the paper snapshot's reconstructed 249-pair draw.
Derived analysis (gem/ablation/global_sweep) stays out of scope — that is
the separate, pending N≈2000 backfill task.

Roster: discovered from HF at runtime (rcp_v1/<slug>/ having
calibration_exfiltration_meta.json), model ids taken from each stored caz.
40 slugs as of 2026-07-17; largest are gemma-2-9b / Qwen2.5-14B — no
Cluster F here, so no bf16/8-bit budget ladder is needed.

Pool handling (IMPORTANT): the exfiltration rerun session (--exfil-rerun)
installs a REPLACEMENT 249-pair pool into the host RCP checkout. This mode
restores the real current pool (`git checkout -- <file>`) before anything
draws pairs, and verifies 962 pairs / 1924 records. A later --exfil-rerun
resume re-stages its own pool (stage_pool is idempotent), so ordering these
sessions is safe in both directions as long as they don't interleave.

Gates per model (hard): manifest set == the full corrected pool's composite
ids, count == 962, identical order across models. Soft (logged only):
peak-DOM anti-correlation vs the stored defective rcp_v1 caz, separation
ratio — same signals as the paper-snapshot rerun, informational here.

Uploads (size-verified): the 4 files replace rcp_v1/<slug>/ on main (old
versions stay retrievable via HF revision history — note: the `current` tag
re-cut is the data owner's call, this script does not move tags). caz+meta
additionally copy to paper_n250/_round3_gpu/exfiltration_rerun/rcp_v1_meta/
as cheap provenance; the npys are too big to double-store.

Via Prefect: run_session.py --rcp-v1 [...] on the round3-gpu-session
deployment (extra_args), same staging rules as the other session modes.

Written: 2026-07-17 12:05 UTC by claude:exfil-rerun (follow-on to
EXFILTRATION_RERUN_SPEC.md; Hopper tf08f424 thread)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from common import (
    HF_DATASET, ROSETTA_DATA, dom_matrix, hf_upload, log, peak_layer,
    shard_done, shard_write,
)
from exfiltration_rerun_session import (
    RCP_POOL, RCP_ROOT, ROSETTA_ANALYSIS, TimedOut, run, purge_weights,
    sync_repos,
)

JOB = "rcp_v1_exfiltration"
CONCEPT = "exfiltration"
OUT_ROOT_RCP = ROSETTA_DATA / "rcp_v1"
BASELINES = ROSETTA_DATA / "rcp_v1_exfil_baselines"
EXPECTED_PAIRS = 962           # current corrected pool: 1924 records
EXTRACT_BUDGET_S = 7200
CORE_FILES = [
    f"calibration_alllayer_{CONCEPT}.npy", f"calibration_{CONCEPT}.npy",
    f"calibration_{CONCEPT}_meta.json", f"caz_{CONCEPT}.json",
]
PROV_PREFIX = "paper_n250/_round3_gpu/exfiltration_rerun/rcp_v1_meta"


def restore_full_pool() -> list[str]:
    """Put the real (current, corrected, full) exfiltration pool back in the
    RCP checkout — the --exfil-rerun session leaves its 249-pair replacement
    there — and return the pool's composite pair ids in file order."""
    r = run(f"git -C {RCP_ROOT} checkout -- pairs/raw/v1/exfiltration_consensus_pairs.jsonl", 60)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError("failed to git-restore the full exfiltration pool")
    ids: list[str] = []
    seen = set()
    n_records = 0
    for line in RCP_POOL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n_records += 1
        cid = f"{rec['pair_id']}__{rec['model_name']}"
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    if len(ids) != EXPECTED_PAIRS or n_records != 2 * EXPECTED_PAIRS:
        raise RuntimeError(
            f"restored pool has {len(ids)} pairs / {n_records} records, expected "
            f"{EXPECTED_PAIRS}/{2 * EXPECTED_PAIRS} — RCP upstream changed; "
            "update EXPECTED_PAIRS only after confirming the change is intended")
    log.info("[pool] full corrected pool restored: %d pairs", len(ids))
    return ids


def discover_roster() -> list[tuple[str, str]]:
    """(slug, model_id) for every rcp_v1 slug carrying exfiltration files,
    ordered smallest-first by the stored caz's hidden_dim * n_layers."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset")
    slugs = sorted({
        f.split("/")[1] for f in files
        if f.startswith("rcp_v1/") and f.endswith(f"calibration_{CONCEPT}_meta.json")
    })
    if not slugs:
        raise RuntimeError("no rcp_v1 slugs with exfiltration metas found on HF")
    BASELINES.mkdir(parents=True, exist_ok=True)
    roster = []
    for slug in slugs:
        dest = BASELINES / slug / f"caz_{CONCEPT}.json"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            p = hf_hub_download(HF_DATASET, f"rcp_v1/{slug}/caz_{CONCEPT}.json",
                                repo_type="dataset")
            dest.write_bytes(Path(p).read_bytes())
        caz = json.loads(dest.read_text())
        size = caz["hidden_dim"] * caz["layer_data"]["n_layers"]
        roster.append((slug, caz["model_id"], size))
    roster.sort(key=lambda t: t[2])
    log.info("[roster] %d rcp_v1 slugs (smallest %s, largest %s)",
             len(roster), roster[0][0], roster[-1][0])
    return [(s, m) for s, m, _ in roster]


def extract_one(slug: str, model_id: str) -> None:
    d = OUT_ROOT_RCP / slug
    if d.exists():
        for pat in (f"*_{CONCEPT}.json", f"*_{CONCEPT}.npy", f"*_{CONCEPT}_meta.json"):
            for p in d.glob(pat):
                p.unlink()
    r = run(
        f"cd {ROSETTA_ANALYSIS} && uv run --no-sync python extraction/extract.py "
        f"--model '{model_id}' --concepts {CONCEPT} --n-pairs 2000 --split all "
        f"--dtype bfloat16 --no-clean-cache --out-root {OUT_ROOT_RCP}",
        timeout=EXTRACT_BUDGET_S)
    if isinstance(r, TimedOut) or r.returncode != 0:
        raise RuntimeError(f"extraction failed for {slug}: "
                           f"{(getattr(r, 'stderr', '') or '')[-600:]}")


def gate_one(slug: str, pool_ids: list[str], manifest_ref: list[str] | None) -> list[str]:
    meta = json.loads((OUT_ROOT_RCP / slug / f"calibration_{CONCEPT}_meta.json").read_text())
    ids = meta["corpus"]["pair_ids"]
    if set(ids) != set(pool_ids) or len(ids) != EXPECTED_PAIRS:
        raise RuntimeError(f"[gate] {slug}: extracted manifest != the full "
                           f"corrected pool ({len(ids)} vs {EXPECTED_PAIRS} pairs)")
    if manifest_ref is not None and ids != manifest_ref:
        raise RuntimeError(f"[gate] {slug}: pair order differs from the session's "
                           "reference model")
    fresh = json.loads((OUT_ROOT_RCP / slug / f"caz_{CONCEPT}.json").read_text())
    stored_p = BASELINES / slug / f"caz_{CONCEPT}.json"
    if stored_p.exists():
        stored = json.loads(stored_p.read_text())
        da = dom_matrix(fresh)[peak_layer(fresh)]
        db = dom_matrix(stored)[peak_layer(stored)]
        cos = float(np.dot(da, db))
        old = float(stored["layer_data"]["peak_separation"])
        new = float(fresh["layer_data"]["peak_separation"])
        ratio = new / old if old > 1e-9 else float("nan")
        log.info("[gate] %s: sep %.3f -> %.3f (%.2fx), cos vs defective stored "
                 "%.3f%s", slug, old, new, ratio, cos,
                 "" if cos < 0 else "  (POSITIVE — flag if widespread)")
    return ids


def upload_one(slug: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    d = OUT_ROOT_RCP / slug
    missing = [f for f in CORE_FILES if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"{slug}: outputs missing before upload: {missing}")
    dests = []
    for fname in CORE_FILES:
        dests.append((d / fname, f"rcp_v1/{slug}/{fname}"))
        if fname.endswith(".json"):
            dests.append((d / fname, f"{PROV_PREFIX}/{slug}/{fname}"))
    for f, dest in dests:
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(f), path_in_repo=dest,
                                repo_id=HF_DATASET, repo_type="dataset")
                break
            except Exception as e:  # noqa: BLE001
                wait = 2 ** attempt * 10
                log.warning("%s: upload %s failed (%s), retry in %ds", slug, dest, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"{slug}: upload failed after retries: {dest}")
    info = {p.path: p for p in api.get_paths_info(
        HF_DATASET, [dest for _, dest in dests], repo_type="dataset")}
    for f, dest in dests:
        got = info.get(dest)
        size = getattr(got, "size", None) or getattr(getattr(got, "lfs", None), "size", None)
        if got is None or (size is not None and size != f.stat().st_size):
            raise RuntimeError(f"{slug}: HF size-verification failed for {dest}")
    log.info("%s: %d uploads size-verified", slug, len(dests))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of slugs (default: every rcp_v1 slug with "
                         "exfiltration files on HF)")
    ap.add_argument("--skip-sync", action="store_true")
    args = ap.parse_args()

    if not args.skip_sync:
        sync_repos()
    pool_ids = restore_full_pool()
    roster = discover_roster()
    if args.models:
        roster = [(s, m) for s, m in roster if s in set(args.models)]

    ref_p = ROSETTA_DATA / "round3_ckpt" / JOB / "manifest_ref.json"
    manifest_ref = json.loads(ref_p.read_text())["ids"] if ref_p.exists() else None

    failures: list[tuple[str, str]] = []
    for i, (slug, model_id) in enumerate(roster, 1):
        if shard_done(JOB, f"model_{slug}"):
            log.info("[rcp_v1] %s already complete — skipping", slug)
            continue
        log.info("[rcp_v1] === %d/%d %s (%s) ===", i, len(roster), slug, model_id)
        t0 = time.time()
        try:
            extract_one(slug, model_id)
            ids = gate_one(slug, pool_ids, manifest_ref)
            if manifest_ref is None:
                manifest_ref = ids
                shard_write(JOB, "manifest_ref", {"ids": ids, "model": slug})
            upload_one(slug)
            shard_write(JOB, f"model_{slug}", {"elapsed_s": time.time() - t0})
            purge_weights(model_id)
        except Exception as e:  # noqa: BLE001 — collect and continue
            log.error("[rcp_v1] %s FAILED: %s: %s", slug, type(e).__name__, e)
            failures.append((slug, f"{type(e).__name__}: {e}"))
        log.info("[rcp_v1] %s done in %.0fmin", slug, (time.time() - t0) / 60)

    manifest = {
        "job": JOB, "utc": time.strftime("%F %T UTC"),
        "n_pairs": EXPECTED_PAIRS, "split": "all",
        "note": "corrected-label full-pool re-extraction; derived analysis "
                "(gem/ablation/global_sweep) deliberately NOT regenerated — "
                "pending N~2000 backfill task. `current` tag re-cut is the "
                "data owner's call.",
        "roster": [s for s, _ in roster],
        "failures": failures,
    }
    mp = ROSETTA_DATA / "results" / "round3_gpu" / "rcp_v1_exfiltration_manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1))
    hf_upload("exfiltration_rerun/rcp_v1_meta", mp)

    if failures:
        for slug, err in failures:
            log.error("[rcp_v1] FAILED: %s — %s", slug, err)
        raise RuntimeError(f"{len(failures)}/{len(roster)} slugs failed — "
                           "relaunch resumes from shards")
    log.info("[rcp_v1] COMPLETE — %d slugs re-extracted, uploaded, verified. "
             "Leave `current` tag re-cut to the data owner.", len(roster))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
