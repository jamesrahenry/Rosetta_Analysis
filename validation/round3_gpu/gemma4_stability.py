#!/usr/bin/env python3
"""gemma4_stability.py — P4 §3.6 Test 2 follow-up (GPU_WORK_OUTSTANDING.md item 1):
does the Gemma-4-generation MoE model share Gemma-2's DOM split-half instability?

Background: the gemma_27b_stability.py discriminator established that Gemma-2's
instability is a Gemma-2 *family/substrate* property (gemma-2-27b, trained from
scratch on the same architecture, reproduced it — so it is not the distillation
recipe alone). Gemma 4 is a DIFFERENT, newer generation (and a mixture-of-experts
variant); P3 has NO Gemma-4 data, so whether the instability crosses generations
is untested. P4 retained "Gemma 4" rows in the 46-model peak-depth variance
decomposition (§3.6 Test 2) with a "not recomputable on corrected labels,
retained as reported" caveat. This check either validates or sharpens that caveat:

    Gemma-4 split-half >= 0.85 (control-like) -> Gemma-4 is STABLE; the instability
        is Gemma-2-specific, does not cross generations. P4's Gemma-4 rows are
        reliable as reported — validates their inclusion.
    Gemma-4 split-half <= 0.75 (like 2b/9b)   -> Gemma-4 SHARES the instability;
        it spans generations. P4's retained-with-caveat rows need that caveat
        kept/strengthened.
    intermediate 0.75-0.85                     -> partial.

Method: identical to gemma_27b_stability.py — the same split-half harness
(n=250 pairs/concept via the deterministic sampler, 5 random half/half splits,
per-layer DOM cosine between halves), the SAME gpt2 + gemma-2-2b controls run
first and hard-gated before the Gemma-4 number counts. Everything model-agnostic
is imported from gemma_27b_stability so this is provably "the same harness".

MODEL ID IS NOT HARDCODED — the repo is internally inconsistent about the
upstream name (PROVENANCE_RESOLUTIONS.md (9) calls it Gemma-4-26B-A4B, matching
the stored HF slug `google_gemma_4_26B_A4B`; CROSS_PAPER_CONSOLIDATION_LEDGER.md
L204 says gemma-4-26B-A4B "does not exist, real name Gemma-3-26B-A4B"). Pass the
confirmed HF id with --target-model; extracting the wrong checkpoint would make
this check meaningless. --target-slug controls the HF results directory name so
it can be matched to the stored `google_gemma_4_26B_A4B` artifact tree if wanted.

This is a DIAGNOSTIC: the target model must not be added to any alignment roster.

Standalone:  python gemma4_stability.py --target-model google/gemma-4-26b-a4b [--smoke]
Via Prefect: run_session.py --gemma-4 --target-model <id> [...] (round3-gpu-session
deployment extra_args) — the worker serializes flow runs, no VRAM contention.

Written: 2026-07-18 UTC by claude:gpu-runner-h200 (GPU_WORK_OUTSTANDING.md item 1)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import OUT_ROOT, hf_upload, hf_verify, log
# The harness proper — imported verbatim so this is literally the same code path
# gemma_27b_stability.py used for its controls and target.
from gemma_27b_stability import (
    CONCEPTS, N_PAIRS, N_SPLITS,
    GPT2_MIN, GEMMA2B_MAX_PEAK, MIN_CONTROL_GAP, GEMMA2B_BAND,
    peak_stat, run_model, stored_peak,
)

JOB = "gemma4_stability"
RESULTS_FILE = "gemma4_stability_results.json"


def verdict_of(mean_g4: float) -> str:
    """Framed for the cross-generation question (see module docstring)."""
    if mean_g4 >= 0.85:
        v = ("STABLE — instability does not cross to the Gemma-4 generation; "
             "it is a Gemma-2-specific property. P4 §3.6 Gemma-4 rows validated")
        if mean_g4 < 0.90:
            v += " [borderline: 0.85-0.90]"
    elif mean_g4 > 0.75:
        v = "PARTIAL — Gemma-4 partially shares the instability"
    else:
        v = ("UNSTABLE — Gemma-4 shares Gemma-2's instability; it spans "
             "generations. Keep/strengthen the P4 §3.6 retained-with-caveat rows")
        if mean_g4 > 0.70:
            v += " [borderline: 0.70-0.75]"
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-model", required=True,
                    help="confirmed HF id of the Gemma-4 target (repo naming is "
                         "inconsistent — do NOT guess; see module docstring)")
    ap.add_argument("--smoke", action="store_true",
                    help="controls + target on causation only, 2 splits, no upload")
    args = ap.parse_args()

    target = args.target_model
    concepts = CONCEPTS[:1] if args.smoke else CONCEPTS
    n_splits = 2 if args.smoke else N_SPLITS

    results: dict = {
        "spec": "GPU_WORK_OUTSTANDING.md item 1 (P4 §3.6 Test 2 follow-up)",
        "target_model": target, "n_pairs": N_PAIRS, "n_splits": n_splits,
        "utc": time.strftime("%F %T UTC"),
        "note": "diagnostic only — target is NOT part of any alignment roster",
        "models": {},
    }

    # --- controls, hard-gated before the target number counts (same as §8.4) --
    gpt2 = run_model("openai-community/gpt2", ["causation"], n_splits, args.smoke)
    g2b = run_model("google/gemma-2-2b", ["causation"], n_splits, args.smoke)
    results["models"]["openai-community/gpt2"] = gpt2
    results["models"]["google/gemma-2-2b"] = g2b

    gpt2_peak = peak_stat(gpt2["causation"], stored_peak("openai-community/gpt2", "causation"))
    g2b_peak = peak_stat(g2b["causation"], stored_peak("google/gemma-2-2b", "causation"))
    if gpt2_peak < GPT2_MIN:
        raise RuntimeError(f"[control] gpt2 x causation {gpt2_peak:.4f} at stored "
                           f"peak < {GPT2_MIN} — harness broken, target number would "
                           "be uninterpretable")
    if g2b_peak > GEMMA2B_MAX_PEAK or (gpt2_peak - g2b_peak) < MIN_CONTROL_GAP:
        raise RuntimeError(f"[control] gemma-2-2b x causation {g2b_peak:.4f} at "
                           f"stored peak is not clearly unstable (need <= "
                           f"{GEMMA2B_MAX_PEAK} and a >= {MIN_CONTROL_GAP} gap to "
                           f"gpt2's {gpt2_peak:.4f}) — baseline does not reproduce")
    if not GEMMA2B_BAND[0] <= g2b_peak <= GEMMA2B_BAND[1]:
        log.warning("[control] gemma-2-2b %.4f outside the spec's literal "
                    "~%.2f-%.2f band (instability contract holds — proceeding)",
                    g2b_peak, *GEMMA2B_BAND)
    log.info("[control] PASSED: gpt2=%.4f at peak (>=%.2f), gemma-2-2b=%.4f "
             "(unstable, gap %.3f)", gpt2_peak, GPT2_MIN, g2b_peak,
             gpt2_peak - g2b_peak)

    # --- the Gemma-4 target -------------------------------------------------
    # No stored caz (not in the corpus) — best-layer mean is the fair
    # single-layer analogue of the controls' stored-peak readout, same as 27b.
    m_g4 = run_model(target, concepts, n_splits, args.smoke)
    results["models"][target] = m_g4
    mean_g4 = float(np.mean([m_g4[c]["best_layer_mean"] for c in concepts]))

    results["summary"] = {
        "stat_definition": "controls read at stored caz peak layer; target (no "
                           "stored caz) at best layer — identical to "
                           "gemma_27b_stability.py",
        "gpt2_causation_at_peak": gpt2_peak,
        "gemma_2_2b_causation_at_peak": g2b_peak,
        "target_mean_over_concepts_best_layer": mean_g4,
        "target_per_concept_best_layer":
            {c: m_g4[c]["best_layer_mean"] for c in concepts},
        "verdict": verdict_of(mean_g4) if not args.smoke else "smoke — no verdict",
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / RESULTS_FILE
    out_path.write_text(json.dumps(results, indent=1))
    log.info("wrote %s", out_path)
    log.info("=== %s mean over %d concepts: %.4f -> %s ===",
             target, len(concepts), mean_g4, results["summary"]["verdict"])

    if not args.smoke:
        hf_upload(JOB, out_path)
        hf_verify(JOB, [RESULTS_FILE])
        log.info("[%s] uploaded + verified — append the verdict to the P4 §3.6 "
                 "notes on the dev box (papers tree isn't on this host)", JOB)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
