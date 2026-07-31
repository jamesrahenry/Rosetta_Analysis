#!/usr/bin/env python3
"""G6 — rerun P4's three 7-concept nulls at full C=17 scope (P4 §3.8).

The published five-null table runs random-vector, concept-label-shuffle, and
depth-label-permutation at 7-concept scope "for compute-cost reasons". The
implementation is papers/prh-validation/scripts/p5_validation_battery_gpu.py,
whose only concept-scope control is its module-level CONCEPTS list — so this
wrapper imports that module unmodified, patches CONCEPTS to the 17-concept
list, stages a data root containing exactly the alignment-roster model dirs,
and invokes the battery's own main() (which brings its own checkpointing via
write_partial/load_partial in --out-dir).

Population note: the battery pairs ALL ordered same-dimension model pairs in
its data root (no cross-family filter) — identical to the published 7-concept
battery, so the C=17 rerun is scope-for-scope comparable. Tests 1 and 4 rerun
too (cheap; test 4's C=17 figures already exist elsewhere and serve as an
internal consistency check on this run).

Usage:
    python g6_c17_null_battery.py [--n-seeds 3] [--engine gpu|cpu] [--smoke]

--engine cpu runs the scipy twin (p5_validation_battery.py) which carries the
gesdd->gesvd fallback — use it as the SVD-driver stability check on the
label-shuffle null if the GPU numbers look surprising.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from common import (
    CKPT_ROOT, CONCEPTS_17, MODELS_ROOT, OUT_ROOT, alignment_roster_from_hf,
    hf_upload, hf_verify, log,
)

JOB = "g6"
SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1] / ".." / "prh-validation" / "scripts"
).resolve()
STAGE_ROOT = CKPT_ROOT / JOB / "data_root"


def stage_data_root(roster: list[str]) -> Path:
    """Symlink exactly the roster model dirs into a private data root, so the
    battery's directory scan can't pick up strays (gpt_neo, dark-matter dirs)."""
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    STAGE_ROOT.mkdir(parents=True)
    staged = 0
    for slug in roster:
        src = (MODELS_ROOT / slug).resolve()
        if src.is_dir():
            (STAGE_ROOT / slug).symlink_to(src)
            staged += 1
        else:
            log.warning("[g6] roster slug missing locally: %s", slug)
    log.info("[g6] staged %d/%d roster dirs", staged, len(roster))
    if staged < 2:
        raise RuntimeError("g6: fewer than 2 model dirs staged — download artifacts first")
    return STAGE_ROOT


def run(engine: str, n_seeds: int, out_suffix: str, smoke: bool) -> Path:
    sys.path.insert(0, str(SCRIPTS_DIR))
    mod_name = "p5_validation_battery_gpu" if engine == "gpu" else "p5_validation_battery"
    battery = __import__(mod_name)

    # THE patch: C=7 -> C=17 (list mutated in place; battery reads it at call time)
    battery.CONCEPTS[:] = CONCEPTS_17[:3] if smoke else CONCEPTS_17
    log.info("[g6] CONCEPTS patched to %d concepts (engine=%s)",
             len(battery.CONCEPTS), engine)
    if smoke:
        battery.N_DEPTH_PERMS = 50
        battery.N_BOOTSTRAP = 200

    roster = (
        ["EleutherAI_pythia_160m", "openai_community_gpt2"] if smoke
        else alignment_roster_from_hf()
    )
    data_root = stage_data_root(roster)
    out_dir = OUT_ROOT / f"{JOB}{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        mod_name, "--data-root", str(data_root), "--out-dir", str(out_dir),
        "--n-seeds", str(n_seeds), "--out-suffix", out_suffix,
    ]
    if engine == "gpu":
        argv += ["--dtype", "float64"]
    old_argv, sys.argv = sys.argv, argv
    try:
        battery.main()
    finally:
        sys.argv = old_argv
    return out_dir


def upload_outputs(out_dir: Path, smoke: bool) -> None:
    files = sorted(p for p in out_dir.glob("*.json") if "_ckpt" not in p.stem)
    if not files:
        raise RuntimeError(f"g6: no outputs found in {out_dir}")
    manifest = {
        "job": JOB, "concepts": 17, "files": [p.name for p in files],
        "population": "all ordered same-dimension pairs, alignment roster "
                      "(matches the published 7-concept battery's pairing rule)",
    }
    man = out_dir / "g6_manifest.json"
    man.write_text(json.dumps(manifest, indent=1))
    if smoke:
        log.info("[g6] smoke run — skipping upload (%d files)", len(files))
        return
    for p in [*files, man]:
        hf_upload(JOB, p)
    hf_verify(JOB, [p.name for p in [*files, man]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="seeds for random-vector and label-shuffle nulls")
    ap.add_argument("--out-suffix", default="_C17")
    ap.add_argument("--smoke", action="store_true",
                    help="2 models, 3 concepts, reduced perms, no upload")
    args = ap.parse_args()

    suffix = args.out_suffix + ("_smoke" if args.smoke else "")
    out_dir = run(args.engine, 1 if args.smoke else args.n_seeds, suffix, args.smoke)
    upload_outputs(out_dir, args.smoke)
    log.info("[g6] done -> %s", out_dir)


if __name__ == "__main__":
    main()
