#!/usr/bin/env python3
"""Round-3 GPU session orchestrator — load each model once, run all its slices.

Order of operations (per ROUND3_COMPUTE_PLAN.md "GPU session implementation
plan"):

  0. download artifacts (caz/gem JSONs for every roster slug — small);
  1. G6 (no model forwards — runs first, needs only the artifacts);
  2. per model, smallest first: G2 slices (28-base roster), G5 extraction
     (alignment roster), G3 rows (5-model subset) — one load, all slices;
  3. G5 stage B (pairwise Procrustes — no GPU);
  4. finalize G2/G3 (aggregate + upload), verify, done.

Every job uploads its own outputs when it finishes; this orchestrator's final
step is verification only. Teardown (task gpu:down) stays manual and happens
AFTER this script exits 0.

Usage:
    python run_session.py            # the full required session
    python run_session.py --smoke    # pythia-70m end-to-end dry run
    python run_session.py --skip g6  # resume patterns; shards make reruns cheap

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import (
    BASE_28, CONCEPTS_17, G3_SUBSET, alignment_roster_from_hf,
    hf_download_artifacts, load_caz, log, slugify,
)

HERE = Path(__file__).resolve().parent


def download_phase(rosters: list[list[str]]) -> None:
    slugs = sorted({slugify(m) if "/" in m else m for r in rosters for m in r})
    patterns = [f"{s}/caz_*.json" for s in slugs] + [f"{s}/gem_*.json" for s in slugs]
    log.info("[session] downloading artifacts for %d slugs", len(slugs))
    hf_download_artifacts(patterns)


def size_key(model_id: str) -> float:
    """Sort key: hidden_dim * n_layers from the caz artifact (proxy for VRAM)."""
    caz = load_caz(slugify(model_id), "causation")
    return caz["hidden_dim"] * caz["layer_data"]["n_layers"]


def run_g6(smoke: bool) -> None:
    cmd = [sys.executable, str(HERE / "g6_c17_null_battery.py")]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)


def per_model_phase(batch_size: int, smoke: bool) -> None:
    import g2_split_pair_ablation as g2
    import g3_cross_concept_matrix as g3
    import g5_random_text_null as g5
    from forward_utils import load_model, release

    if smoke:
        # pythia-70m exercises G2/G3; the 768-dim cross-family pair
        # (pythia-160m x gpt2) exercises G5's extraction + pairwise stage.
        union = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
                 "openai-community/gpt2"]
        in_g2 = in_g3 = {"EleutherAI/pythia-70m"}
        in_g5 = {"EleutherAI/pythia-160m", "openai-community/gpt2"}
        concepts = CONCEPTS_17[:2]
    else:
        align_slugs = alignment_roster_from_hf()
        align_ids = [load_caz(s, "causation")["model_id"] for s in align_slugs]
        in_g2, in_g5, in_g3 = set(BASE_28), set(align_ids), set(G3_SUBSET)
        union = sorted(in_g2 | in_g5, key=size_key)
        concepts = CONCEPTS_17

    # One bad model must cost one model, not the session: each failure so
    # far (opt-350m dims, gemma-2-2b stale caz) killed the whole run and
    # cost a flow-run round-trip at $3.44/hr. Failures are collected and
    # re-raised AFTER the loop so good models still complete; per-model
    # shards mean the failed ones are cheap to re-run once fixed.
    failures: list[tuple[str, str]] = []
    for i, mid in enumerate(union, 1):
        t0 = time.time()
        log.info("[session] === model %d/%d: %s ===", i, len(union), mid)
        model = None
        try:
            model, tok, device = load_model(mid)
            if mid in in_g2:
                g2.run_for_model(mid, model, tok, device, batch_size,
                                 concepts, smoke=smoke)
            if mid in in_g5:
                g5.extract_for_model(mid, model, tok, device, batch_size,
                                     upload_acts=not smoke, smoke=smoke)
            if mid in in_g3:
                g3.run_for_model(mid, model, tok, device, batch_size,
                                 concepts, smoke=smoke)
        except Exception as e:  # noqa: BLE001 — reported + re-raised after loop
            log.error("[session] %s FAILED: %s: %s", mid, type(e).__name__, e)
            failures.append((mid, f"{type(e).__name__}: {e}"))
        finally:
            if model is not None:
                release(model)
        log.info("[session] %s done in %.0fs", mid, time.time() - t0)
    if failures:
        for mid, err in failures:
            log.error("[session] FAILED MODEL: %s — %s", mid, err)
        raise RuntimeError(
            f"{len(failures)}/{len(union)} models failed (listed above); "
            "completed models are checkpointed — fix and resume")


def main() -> None:
    # Dispatch: `run_session.py --exfil-rerun [...]` runs the exfiltration
    # full-rerun session (EXFILTRATION_RERUN_SPEC.md) instead — lets the
    # existing round3-gpu-session Prefect deployment drive it via extra_args
    # with no new flow/deployment. Remaining args are parsed by that script.
    if "--exfil-rerun" in sys.argv:
        sys.argv.remove("--exfil-rerun")
        import exfiltration_rerun_session
        exfiltration_rerun_session.main()
        return
    if "--gemma-27b" in sys.argv:   # spec §8 piggyback diagnostic
        sys.argv.remove("--gemma-27b")
        import gemma_27b_stability
        gemma_27b_stability.main()
        return
    if "--gemma-4" in sys.argv:     # P4 §3.6 follow-up: Gemma-4 generation check
        sys.argv.remove("--gemma-4")
        import gemma4_stability     # requires --target-model (repo naming ambiguous)
        gemma4_stability.main()
        return
    if "--rcp-v1" in sys.argv:      # full-pool exfiltration re-extraction
        sys.argv.remove("--rcp-v1")
        import rcp_v1_exfiltration_extraction
        rcp_v1_exfiltration_extraction.main()
        return
    if "--mm-cka" in sys.argv:      # multimodal+CKA backfill (P3 review catch)
        sys.argv.remove("--mm-cka")
        import exfiltration_mm_cka_backfill
        exfiltration_mm_cka_backfill.main()
        return
    if "--g5b-orig" in sys.argv:    # P4a: G5b random-text null, ORIGINAL corpus
        sys.argv.remove("--g5b-orig")
        import g5b_random_text_null_original_corpus
        g5b_random_text_null_original_corpus.main()
        return
    if "--c10-null" in sys.argv:    # C10 / P4 Test 5: permuted-label handoff null
        sys.argv.remove("--c10-null")
        import handoff_permuted_null
        handoff_permuted_null.main()
        return
    if "--g7-extract" in sys.argv:  # P4b: G7 human-written calibration extraction
        sys.argv.remove("--g7-extract")
        sys.path.insert(0, str(HERE.parent / "g7_human_written"))
        import extract_g7
        extract_g7.main()
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["download", "g6", "models", "g5b", "finalize"])
    args = ap.parse_args()

    import g2_split_pair_ablation as g2
    import g3_cross_concept_matrix as g3
    import g5_random_text_null as g5

    if "download" not in args.skip and not args.smoke:
        align = alignment_roster_from_hf()
        download_phase([BASE_28, G3_SUBSET, align])

    if "g6" not in args.skip:
        run_g6(args.smoke)

    if "models" not in args.skip:
        per_model_phase(args.batch_size, args.smoke)

    if "g5b" not in args.skip:
        if args.smoke:
            g5.pairwise(CONCEPTS_17[:2], smoke=True,
                        smoke_roster=[slugify("EleutherAI/pythia-160m"),
                                      slugify("openai-community/gpt2")])
        else:
            g5.pairwise(CONCEPTS_17)

    if "finalize" not in args.skip:
        g2.finalize(smoke=args.smoke)
        g3.finalize(smoke=args.smoke)

    log.info("[session] COMPLETE — verify HF listings above, then tear down.")


if __name__ == "__main__":
    main()
