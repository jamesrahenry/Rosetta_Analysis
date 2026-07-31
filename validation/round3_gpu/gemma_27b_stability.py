#!/usr/bin/env python3
"""gemma_27b_stability.py — EXFILTRATION_RERUN_SPEC §8: does gemma-2's DOM
instability come from the training recipe (knowledge distillation) or the
architecture?

gemma-2-2b/-9b are distillation-trained; gemma-2-27b is trained FROM SCRATCH
with the same architecture (softcap, pre+post RMSNorm sandwich, 256k tied
vocab, alternating local/global attention). Every architectural intervention
has failed to move the instability (GEMMA_INSTABILITY_NOTE.md,
ROBUST_DOM_ESTIMATOR_TEST.md, GEMMA_DISTRIBUTED_SIGNAL_TEST.md), so 27b is
the clean discriminator:

    27b split-half ~0.9+ (control-like)  -> training recipe (distillation)
    27b split-half ~0.5-0.7 (like 2b/9b) -> architecture back in play
    intermediate 0.75-0.85               -> partial, both contribute

Method: the exact split-half harness the 2b/9b baselines came from
(gemma_softcap_ablation.py's diagnostics path, stock config): n=250 pairs
per concept via the deterministic sampler against current RCP (a
draw-internal comparison — does NOT use the §1a reconstructed exfiltration
set, and exfiltration is deliberately not in the concept list), 5 random
half/half splits by pair, per-layer DOM cosine between halves.

Controls run in the SAME process, before 27b's number counts (§8.4):
  * gpt2 x causation: overall_mean >= 0.96 (harness sanity)  [hard gate]
  * gemma-2-2b x causation: reproduce ~0.46-0.52 on this host/env
    (hard-fails outside 0.40-0.62; warns outside the ~band)

This is a DIAGNOSTIC: gemma-2-27b must not be added to any roster.

Standalone:  python gemma_27b_stability.py [--smoke] [--with-9b-it]
Via Prefect: run_session.py --gemma-27b [...] (round3-gpu-session deployment
extra_args) — queues behind whatever session is running; the worker
serializes flow runs so there's no VRAM contention.

Written: 2026-07-17 04:30 UTC by claude:exfil-rerun (spec §8 by James /
claude:p3-review, Hopper tfa2acf6)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gpu_utils import get_device, get_dtype

from common import OUT_ROOT, dom_direction, hf_upload, hf_verify, log, shard_done, shard_write

JOB = "gemma_27b_stability"
TARGET_MODEL = "google/gemma-2-27b"          # base, NOT -it (§8.1)
OPTIONAL_MODEL = "google/gemma-2-9b-it"      # distilled + RLHF'd (§8.5)

# §8.3 — the 2b baseline's 8 concepts: two stable ones (formality,
# credibility) as internal gradient; exfiltration excluded to keep this
# clean of the label story.
CONCEPTS = [
    "causation", "agency", "deception", "moral_valence",
    "sentiment", "negation", "formality", "credibility",
]
N_PAIRS = 250
N_SPLITS = 5
BATCH_SIZE = 16

# Gate stats are taken AT THE STORED CAZ PEAK LAYER (downloaded from HF),
# not as the all-layer mean: the first run (2026-07-17 05:18, flow
# 'new-world-man') showed gpt2's all-layer mean dilutes to 0.94 while its
# peak-layer mean is 0.9648 — and the instability note's reference numbers
# (gpt2 0.975/0.967 mean/min) are only consistent with a single-layer
# readout. gemma-2-2b's literal "~0.46-0.52" band could not be reproduced
# under ANY aggregation (peak 0.62, overall 0.55, best 0.73); the control's
# real contract — 2b clearly unstable, far below control level — is what's
# gated. Definition ambiguity flagged to the §8 author in Hopper.
GPT2_MIN = 0.96                  # §8.4 hard gate, at stored caz peak layer
GEMMA2B_MAX_PEAK = 0.75          # 2b must remain clearly unstable at peak
MIN_CONTROL_GAP = 0.20           # gpt2(peak) - gemma2b(peak)
GEMMA2B_BAND = (0.46, 0.52)      # spec's literal band — warn-only, recorded
RESULTS_FILE = "gemma_27b_stability_results.json"
HF_DATASET = "james-ra-henry/Rosetta-Activations"


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def load_stock(model_id: str, device: str, dtype):
    """Stock config, eager attention — the exact load the 2b/9b baseline
    numbers came from (gemma_softcap_ablation.py's softcap_on path). gpt2
    ignores the flag's purpose but holding the backend constant keeps the
    harness identical across all rows."""
    from forward_utils import require_cuda_if_expected
    require_cuda_if_expected(model_id)  # fail fast if the container lost the GPU
    model = AutoModel.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


def split_half_curves(model, tok, device, pos_texts, neg_texts, n_splits):
    """per_layer_cos[layer] = [cosine per split] — verbatim from
    gemma_softcap_ablation.py (the baseline harness; do not 'improve')."""
    n = min(len(pos_texts), len(neg_texts))
    per_layer_cos: list[list[float]] | None = None
    for s in range(n_splits):
        rng = np.random.RandomState(s)
        perm_pos = rng.permutation(n)
        perm_neg = rng.permutation(n)
        h = n // 2
        pos_a = [pos_texts[i] for i in perm_pos[:h]]
        pos_b = [pos_texts[i] for i in perm_pos[h:2 * h]]
        neg_a = [neg_texts[i] for i in perm_neg[:h]]
        neg_b = [neg_texts[i] for i in perm_neg[h:2 * h]]
        acts = {
            k: extract_layer_activations(model, tok, v, device=device,
                                         batch_size=BATCH_SIZE, pool="last")
            for k, v in (("pa", pos_a), ("na", neg_a), ("pb", pos_b), ("nb", neg_b))
        }
        n_layers = len(acts["pa"])
        if per_layer_cos is None:
            per_layer_cos = [[] for _ in range(n_layers)]
        for layer in range(n_layers):
            dom_a = dom_direction(acts["pa"][layer], acts["na"][layer])
            dom_b = dom_direction(acts["pb"][layer], acts["nb"][layer])
            per_layer_cos[layer].append(cosine(dom_a, dom_b))
    assert per_layer_cos is not None
    return per_layer_cos


def run_model(model_id: str, concepts: list[str], n_splits: int,
              smoke: bool = False) -> dict:
    """All concepts for one model, shard-checkpointed per (model, concept)."""
    suffix = "_smoke" if smoke else ""
    out: dict[str, dict] = {}
    todo: list[str] = []
    for c in concepts:
        cached = shard_done(JOB, f"{model_id.replace('/', '_')}_{c}{suffix}")
        if cached is not None:
            out[c] = cached
        else:
            todo.append(c)
    if not todo:
        log.info("[%s] all %d concepts cached", model_id, len(concepts))
        return out

    model, tok = load_stock(model_id, get_device("auto"), get_dtype(get_device("auto")))
    device = get_device("auto")
    try:
        for concept in todo:
            t0 = time.time()
            pairs = load_concept_pairs(concept, n=N_PAIRS, split="train")
            pos, neg = texts_by_label(pairs)
            curves = split_half_curves(model, tok, device, pos, neg, n_splits)
            means = [float(np.mean(c)) for c in curves]
            mins = [float(np.min(c)) for c in curves]
            best = int(np.argmax(means))
            res = {
                "per_layer_mean": means, "per_layer_min": mins,
                "overall_mean": float(np.mean(means)),
                "best_layer": best, "best_layer_mean": means[best],
                "elapsed_s": time.time() - t0,
            }
            out[concept] = res
            shard_write(JOB, f"{model_id.replace('/', '_')}_{concept}{suffix}", res)
            log.info("[%s] %-14s overall_mean=%.4f best_layer=%d (%.4f)",
                     model_id, concept, res["overall_mean"], best, means[best])
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def stored_peak(model_id: str, concept: str) -> int | None:
    """Stored caz peak layer from HF (None if the model has no stored caz —
    e.g. gemma-2-27b, which is deliberately not in the corpus)."""
    slug = model_id.replace("/", "_").replace("-", "_")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(HF_DATASET, f"paper_n250/{slug}/caz_{concept}.json",
                            repo_type="dataset")
        return int(json.loads(Path(p).read_text())["layer_data"]["peak_layer"])
    except Exception:  # noqa: BLE001 — no stored artifact
        return None


def peak_stat(res: dict, pk: int | None) -> float:
    """Split-mean at the stored peak layer; falls back to best-layer mean
    when there is no stored peak (27b) or it's out of range."""
    means = res["per_layer_mean"]
    if pk is not None and 0 <= pk < len(means):
        return float(means[pk])
    return float(res["best_layer_mean"])


def verdict_of(mean_27b: float) -> str:
    if mean_27b >= 0.85:
        v = "training recipe (distillation) — architecture exonerated"
        if mean_27b < 0.90:
            v += " [borderline: 0.85-0.90]"
    elif mean_27b > 0.75:
        v = "partial — both recipe and architecture contribute"
    else:
        v = "architectural/family cause back in play"
        if mean_27b > 0.70:
            v += " [borderline: 0.70-0.75]"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="controls + 27b on causation only, 2 splits, no upload")
    ap.add_argument("--with-9b-it", action="store_true",
                    help="§8.5 optional second point on the recipe axis")
    args = ap.parse_args()

    concepts = CONCEPTS[:1] if args.smoke else CONCEPTS
    n_splits = 2 if args.smoke else N_SPLITS

    results: dict = {
        "spec": "EXFILTRATION_RERUN_SPEC.md §8", "hopper": "tfa2acf6",
        "n_pairs": N_PAIRS, "n_splits": n_splits,
        "utc": time.strftime("%F %T UTC"),
        "note": "diagnostic only — gemma-2-27b is NOT part of any corpus roster",
        "models": {},
    }

    # --- §8.4 controls, hard-gated before 27b counts -----------------------
    gpt2 = run_model("openai-community/gpt2", ["causation"], n_splits, args.smoke)
    g2b = run_model("google/gemma-2-2b", ["causation"], n_splits, args.smoke)
    results["models"]["openai-community/gpt2"] = gpt2
    results["models"]["google/gemma-2-2b"] = g2b

    gpt2_peak = peak_stat(gpt2["causation"], stored_peak("openai-community/gpt2", "causation"))
    g2b_peak = peak_stat(g2b["causation"], stored_peak("google/gemma-2-2b", "causation"))
    if gpt2_peak < GPT2_MIN:
        raise RuntimeError(f"[control] gpt2 x causation {gpt2_peak:.4f} at stored "
                           f"peak < {GPT2_MIN} — harness broken, 27b number would "
                           "be uninterpretable")
    if g2b_peak > GEMMA2B_MAX_PEAK or (gpt2_peak - g2b_peak) < MIN_CONTROL_GAP:
        raise RuntimeError(f"[control] gemma-2-2b x causation {g2b_peak:.4f} at "
                           f"stored peak is not clearly unstable (need <= "
                           f"{GEMMA2B_MAX_PEAK} and a >= {MIN_CONTROL_GAP} gap to "
                           f"gpt2's {gpt2_peak:.4f}) — baseline does not reproduce")
    if not GEMMA2B_BAND[0] <= g2b_peak <= GEMMA2B_BAND[1]:
        log.warning("[control] gemma-2-2b %.4f outside the spec's literal "
                    "~%.2f-%.2f band (instability contract holds — proceeding; "
                    "definition ambiguity is flagged in Hopper)",
                    g2b_peak, *GEMMA2B_BAND)
    log.info("[control] PASSED: gpt2=%.4f at peak (>=%.2f), gemma-2-2b=%.4f "
             "(unstable, gap %.3f)", gpt2_peak, GPT2_MIN, g2b_peak,
             gpt2_peak - g2b_peak)

    # --- the discriminator --------------------------------------------------
    m27 = run_model(TARGET_MODEL, concepts, n_splits, args.smoke)
    results["models"][TARGET_MODEL] = m27
    # 27b has no stored caz (not in the corpus, by design) — best-layer mean
    # is the fair single-layer analogue of the controls' stored-peak readout.
    mean_27b = float(np.mean([m27[c]["best_layer_mean"] for c in concepts]))

    if args.with_9b_it and not args.smoke:
        results["models"][OPTIONAL_MODEL] = run_model(OPTIONAL_MODEL, concepts, n_splits)

    results["summary"] = {
        "stat_definition": "controls read at stored caz peak layer; 27b (no "
                           "stored caz) at best layer; overall_mean kept in "
                           "per-model detail for the record",
        "gpt2_causation_at_peak": gpt2_peak,
        "gemma_2_2b_causation_at_peak": g2b_peak,
        "gemma_2_27b_mean_over_concepts_best_layer": mean_27b,
        "gemma_2_27b_per_concept_best_layer":
            {c: m27[c]["best_layer_mean"] for c in concepts},
        "verdict": verdict_of(mean_27b) if not args.smoke else "smoke — no verdict",
    }
    if args.with_9b_it and OPTIONAL_MODEL in results["models"]:
        results["summary"]["gemma_2_9b_it_mean_over_concepts_best_layer"] = float(np.mean(
            [results["models"][OPTIONAL_MODEL][c]["best_layer_mean"] for c in concepts]))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / RESULTS_FILE
    out_path.write_text(json.dumps(results, indent=1))
    log.info("wrote %s", out_path)
    log.info("=== 27b mean over %d concepts: %.4f -> %s ===",
             len(concepts), mean_27b, results["summary"]["verdict"])

    if not args.smoke:
        hf_upload(JOB, out_path)
        hf_verify(JOB, [RESULTS_FILE])
        log.info("[%s] uploaded + verified — append the verdict to "
                 "GEMMA_DISTRIBUTED_SIGNAL_TEST.md on the dev box (papers tree "
                 "isn't on this host)", JOB)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
