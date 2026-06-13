#!/usr/bin/env python3
"""Reconcile a GPU host's local rosetta_data/models against HF before teardown.

Goal: before destroying a rented VM, prove that nothing on its disk is data we
would LOSE — i.e. every local artifact is already on Hugging Face
(james-ra-henry/Rosetta-Activations, paper_n250/<slug>/) — and that the data is
correct (caz n_pairs=250, random-control baselines consistent with the sweep).

Design notes (why it works this way):
  * HF is the system of record. Hosts accumulate many models across the campaign
    in every state of completeness, and the SAME model appears on several hosts
    in different states (a complete copy on one, a failed `caz=1` stub on another).
  * So we do NOT check local against a fixed "everything=17" spec — random gaps
    (16/17) and patch/cka subsets are normal and would produce false alarms.
  * The gate is a per-file diff: a local file whose basename is ABSENT from HF is
    a data-loss risk. A local file already on HF is safe (destroying the VM loses
    nothing new), even if the local dir is an incomplete stub.
  * --fix uploads ONLY files missing from HF, one by one. It NEVER calls
    upload_folder and NEVER overwrites an existing HF file, so a local stub can
    never clobber a good HF copy. Size mismatches are REPORTED, never auto-fixed.

Exit 0 only if, after any --fix, every local file is on HF and no correctness
problem was found. Exit 3 otherwise (so a daemon marks the job blocked and we
inspect before destroying the host).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "james-ra-henry/Rosetta-Activations"
REPO_TYPE = "dataset"
HF_PREFIX = "paper_n250"

# Local files the per-model upload step intentionally skips (timestamped history
# copies). Anything else local is expected on HF. Mirrors the job's
# ignore_patterns=['*_20??????_??????.json'].
import re
TIMESTAMPED = re.compile(r"_20\d{6}_\d{6}\.json$")


def local_files(model_dir: Path) -> dict[str, int]:
    """Basename -> size for every uploadable file in the model dir (recursive)."""
    out: dict[str, int] = {}
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(model_dir).as_posix()
        if TIMESTAMPED.search(p.name):
            continue
        out[rel] = p.stat().st_size
    return out


def hf_files(api: HfApi, slug: str) -> dict[str, int] | None:
    """Basename(rel) -> size for the slug on HF, or None if the slug is absent."""
    base = f"{HF_PREFIX}/{slug}"
    try:
        tree = api.list_repo_tree(REPO, path_in_repo=base, repo_type=REPO_TYPE, recursive=True)
    except Exception:
        return None
    out: dict[str, int] = {}
    found = False
    for t in tree:
        found = True
        # RepoFile has .size; RepoFolder does not — skip folders.
        size = getattr(t, "size", None)
        if size is None:
            continue
        rel = t.path[len(base) + 1:]  # strip "paper_n250/<slug>/"
        out[rel] = size
    return out if found else None


def correctness(model_dir: Path) -> list[str]:
    """Return a list of human-readable correctness problems (empty = clean)."""
    problems: list[str] = []
    # caz n_pairs must be 250
    bad_np = []
    for f in sorted(glob.glob(str(model_dir / "caz_*.json"))):
        try:
            np_ = json.load(open(f)).get("n_pairs")
            if np_ != 250:
                bad_np.append(f"{os.path.basename(f)}={np_}")
        except Exception as e:
            bad_np.append(f"{os.path.basename(f)}=UNREADABLE({type(e).__name__})")
    if bad_np:
        problems.append(f"caz n_pairs!=250: {bad_np}")
    # random-control baseline must match the matching global-sweep baseline
    def baseline(p):
        try:
            return json.load(open(p)).get("baseline_final_sep")
        except Exception:
            return None
    mismatch = []
    for f in glob.glob(str(model_dir / "ablation_random_*.json")):
        concept = os.path.basename(f)[len("ablation_random_"):-len(".json")]
        sweep = model_dir / f"ablation_global_sweep_{concept}.json"
        if sweep.exists():
            br, bs = baseline(f), baseline(sweep)
            if br is None or bs is None or abs(br - bs) > 1e-4:
                mismatch.append(concept)
    if mismatch:
        problems.append(f"random/sweep baseline mismatch: {mismatch}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(Path.home() / "rosetta_data" / "models"))
    ap.add_argument("--slug", nargs="*", help="limit to these slugs (default: all local dirs)")
    ap.add_argument("--fix", action="store_true", help="upload files missing from HF (never overwrites)")
    args = ap.parse_args()

    models_root = Path(args.models_dir)
    if not models_root.is_dir():
        print(f"FATAL: {models_root} not found", file=sys.stderr)
        return 2

    api = HfApi()  # token auto-resolved from ~/.cache/huggingface/token
    slugs = args.slug or sorted(d.name for d in models_root.iterdir() if d.is_dir())

    host = os.uname().nodename
    print(f"=== verify_host_vs_hf on {host}: {len(slugs)} local model dir(s) vs {REPO}/{HF_PREFIX} ===\n")

    n_ok = n_upload = n_review = 0
    loss_risk: list[str] = []      # files only local, still missing after fix
    review: list[str] = []         # size mismatch / correctness — needs a human

    for slug in slugs:
        mdir = models_root / slug
        lf = local_files(mdir)
        hf = hf_files(api, slug)
        hf_keys = set(hf or {})

        missing = sorted(k for k in lf if k not in hf_keys)
        size_mm = sorted(k for k in lf if k in hf_keys and hf[k] != lf[k])
        probs = correctness(mdir)

        uploaded_now: list[str] = []
        if missing and args.fix:
            for rel in list(missing):
                try:
                    api.upload_file(
                        path_or_fileobj=str(mdir / rel),
                        path_in_repo=f"{HF_PREFIX}/{slug}/{rel}",
                        repo_id=REPO, repo_type=REPO_TYPE,
                    )
                    uploaded_now.append(rel)
                except Exception as e:
                    print(f"    !! upload failed {slug}/{rel}: {type(e).__name__}: {e}")
            missing = [k for k in missing if k not in uploaded_now]

        hf_state = "ABSENT" if hf is None else f"{len(hf_keys)} files"
        if missing:
            n_upload += 1
            loss_risk += [f"{slug}/{m}" for m in missing]
            verdict = f"UPLOAD-NEEDED ({len(missing)} local-only file(s) not on HF)"
        elif size_mm or probs:
            n_review += 1
            verdict = "REVIEW"
        else:
            n_ok += 1
            verdict = "OK" + (f" (uploaded {len(uploaded_now)})" if uploaded_now else "")

        print(f"[{verdict:<22}] {slug:<40} local={len(lf):>4}f  hf={hf_state}")
        if size_mm:
            review += [f"{slug}/{m} (local={lf[m]} hf={hf[m]})" for m in size_mm]
            print(f"      size-mismatch (NOT auto-fixed): {size_mm[:6]}{'…' if len(size_mm)>6 else ''}")
        for p in probs:
            review.append(f"{slug}: {p}")
            print(f"      correctness: {p}")
        if missing:
            print(f"      MISSING ON HF: {missing[:8]}{'…' if len(missing)>8 else ''}")

    print(f"\n=== {host}: {n_ok} OK, {n_upload} upload-needed, {n_review} review ===")
    if loss_risk:
        print(f"DATA-LOSS RISK — {len(loss_risk)} local-only file(s) NOT on HF (do NOT destroy):")
        for x in loss_risk[:40]:
            print(f"  - {x}")
        if len(loss_risk) > 40:
            print(f"  … and {len(loss_risk)-40} more")
    if review:
        print(f"REVIEW — {len(review)} item(s) need a human before teardown:")
        for x in review[:40]:
            print(f"  - {x}")
    if not loss_risk and not review:
        print(f"SAFE TO DESTROY {host}: every local artifact is on HF and data checks pass.")
        return 0
    return 3


if __name__ == "__main__":
    sys.exit(main())
