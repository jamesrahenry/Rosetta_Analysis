#!/usr/bin/env python3
"""G2 — split-pair held-out ablation (P3, te256b3c objection 2 companion).

Breaks the triple-use of the same 250 pairs (detection, intervention
direction, and outcome all from one estimate): directions are estimated on
pair half A (RCP train indices 0-124), suppression is measured on half B
(125-249) — and, for the inflation contrast, on half A as well.

Per (model, concept):
  * targets = the dominant CAZ peak + up to 3 depth-matched non-CAZ controls
    (>3 layers from every detected region peak: closest-to-peak-depth,
    median-depth, deepest);
  * u_A^(L) = DOM direction at target layer L estimated from half A;
  * baseline and ablated final-layer Fisher separation (trace-normalized,
    matching the pipeline) measured separately on half A (in-sample) and
    half B (held-out);
  * secondary metric: 1-D Fisher separation of projections onto the
    final-layer direction estimated from half A.

Outputs: one shard per model (checkpointed), plus g2_split_pair_results.json
uploaded to HF paper_n250/_round3_gpu/g2/ at finalize.

Standalone:  python g2_split_pair_ablation.py --all [--smoke]
In-session:  run_session.py calls run_for_model() / finalize().

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import (
    BASE_28, CONCEPTS_17, OUT_ROOT, fisher_separation_1d, fisher_separation_nd,
    dom_direction, hf_upload, hf_verify, load_caz, log, noncaz_layers,
    peak_layer, shard_done, shard_write, slugify,
)
from forward_utils import (
    ablated_contrastive_acts, calibrate_offset, contrastive_acts, load_model,
    release,
)

JOB = "g2"
N_PAIRS = 250
# exfiltration's corrected pool is 249 pairs (EXFILTRATION_RERUN_SPEC §1a):
# pool == draw size, the sampler takes everything. Every other concept stays
# at the original 250.
N_PAIRS_OVERRIDE = {"exfiltration": 249}
N_CONTROLS = 3


def pick_controls(slug: str, concept: str, n_layers: int, pk: int) -> list[int]:
    """Deterministic depth-matched non-CAZ controls: nearest to the peak,
    nearest to mid-depth, and deepest available (deduplicated)."""
    pool = noncaz_layers(slug, concept, n_layers)
    if not pool:
        return []
    picks = [
        min(pool, key=lambda l: (abs(l - pk), l)),
        min(pool, key=lambda l: (abs(l - n_layers // 2), l)),
        max(pool),
    ]
    out: list[int] = []
    for p in picks:
        if p not in out:
            out.append(p)
    return out[:N_CONTROLS]


def halves(pos: list[str], neg: list[str]) -> tuple:
    h = len(pos) // 2
    return (pos[:h], neg[:h]), (pos[h:], neg[h:])


def measure(acts_half, final_dir_A: np.ndarray) -> dict:
    """Final-layer separations for one half's activation list."""
    pos_f, neg_f = acts_half[-1]
    return {
        "final_sep_nd": fisher_separation_nd(pos_f, neg_f),
        "final_sep_1d": fisher_separation_1d(pos_f @ final_dir_A, neg_f @ final_dir_A),
    }


def run_for_model(model_id: str, model, tok, device, batch_size: int,
                  concepts: list[str], smoke: bool = False,
                  calib_concept: str | None = None) -> dict | None:
    """calib_concept: concept used ONLY for the layer-indexing offset
    calibration (a model-level, concept-independent property). Defaults to
    concepts[0]; pass a stable concept when running a noisy-concept-only
    slice — an exfiltration-only run failed the 0.9 structural check on
    pythia-70m (cos 0.889) purely because exfiltration is the corpus's
    noisiest concept, not because the offset was wrong (2026-07-17)."""
    slug = slugify(model_id)
    key = slug + ("_smoke" if smoke else "")
    if (done := shard_done(JOB, key)) is not None:
        log.info("[g2] %s already done — skipping", slug)
        return done

    offset = calibrate_offset(model, tok, device, slug,
                              calib_concept or concepts[0], batch_size)

    from rosetta_tools.dataset import load_concept_pairs, texts_by_label

    rows = []
    t0 = time.time()
    for concept in concepts:
        caz = load_caz(slug, concept)
        n_layers = caz["layer_data"]["n_layers"]
        pk = peak_layer(caz)

        n_pairs = 16 if smoke else N_PAIRS_OVERRIDE.get(concept, N_PAIRS)
        pairs = load_concept_pairs(concept, n=n_pairs, split="train")
        pos, neg = texts_by_label(pairs)
        (posA, negA), (posB, negB) = halves(pos, neg)

        acts_A = contrastive_acts(model, tok, posA, negA, device, batch_size)
        acts_B = contrastive_acts(model, tok, posB, negB, device, batch_size)

        # everything estimated is estimated from half A only
        final_dir_A = dom_direction(*acts_A[-1])
        base_A = measure(acts_A, final_dir_A)
        base_B = measure(acts_B, final_dir_A)

        targets = [("peak", pk)] + [
            ("control", c) for c in pick_controls(slug, concept, n_layers, pk)
        ]
        if smoke:
            targets = targets[:2]

        for role, layer in targets:
            u_A = dom_direction(*acts_A[layer + offset])
            abl_A = ablated_contrastive_acts(
                model, tok, posA, negA, device, batch_size, u_A, layer, offset)
            abl_B = ablated_contrastive_acts(
                model, tok, posB, negB, device, batch_size, u_A, layer, offset)
            m_A, m_B = measure(abl_A, final_dir_A), measure(abl_B, final_dir_A)
            rows.append({
                "model_id": model_id, "concept": concept, "role": role,
                "layer": layer, "peak_layer": pk, "n_layers": n_layers,
                "depth_pct": 100.0 * layer / n_layers, "offset": offset,
                "baseline_insample_nd": base_A["final_sep_nd"],
                "baseline_heldout_nd": base_B["final_sep_nd"],
                "reduction_insample_nd":
                    1.0 - m_A["final_sep_nd"] / max(base_A["final_sep_nd"], 1e-12),
                "reduction_heldout_nd":
                    1.0 - m_B["final_sep_nd"] / max(base_B["final_sep_nd"], 1e-12),
                "reduction_insample_1d":
                    1.0 - m_A["final_sep_1d"] / max(base_A["final_sep_1d"], 1e-12),
                "reduction_heldout_1d":
                    1.0 - m_B["final_sep_1d"] / max(base_B["final_sep_1d"], 1e-12),
            })
        log.info("[g2] %s/%s done (%d targets)", slug, concept, len(targets))

    payload = {"model_id": model_id, "elapsed_s": time.time() - t0, "rows": rows}
    shard_write(JOB, key, payload)
    return payload


def finalize(smoke: bool = False) -> None:
    """Aggregate shards, write + upload the job JSON."""
    from common import CKPT_ROOT
    suffix = "_smoke" if smoke else ""
    shards = sorted((CKPT_ROOT / JOB).glob(f"*{suffix}.json"))
    if not smoke:
        shards = [s for s in shards if not s.stem.endswith("_smoke")]
    rows = []
    for s in shards:
        rows.extend(json.loads(s.read_text())["rows"])

    def agg(sel_role: str, field: str) -> dict:
        v = [r[field] for r in rows if r["role"] == sel_role]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "median": float(np.median(v)) if v else None}

    out = {
        "job": JOB, "n_models": len(shards), "n_rows": len(rows),
        "design": "directions and final-layer reference estimated on RCP train "
                  "pair indices 0-124 (half A); held-out measurement on 125-249 "
                  "(half B); in-sample measurement on half A for the inflation "
                  "contrast",
        "summary": {
            "peak_heldout_nd": agg("peak", "reduction_heldout_nd"),
            "peak_insample_nd": agg("peak", "reduction_insample_nd"),
            "control_heldout_nd": agg("control", "reduction_heldout_nd"),
            "control_insample_nd": agg("control", "reduction_insample_nd"),
        },
        "rows": rows,
    }
    p = agg("peak", "reduction_heldout_nd")["mean"]
    c = agg("control", "reduction_heldout_nd")["mean"]
    if p is not None and c not in (None, 0):
        out["summary"]["heldout_peak_over_control_ratio"] = p / c

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fname = f"g2_split_pair_results{suffix}.json"
    fpath = OUT_ROOT / fname
    fpath.write_text(json.dumps(out, indent=1))
    hf_upload(JOB, fpath)
    hf_verify(JOB, [fname])
    log.info("[g2] finalized: %d rows, peak held-out mean=%s control=%s",
             len(rows), p, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="single HF model id")
    ap.add_argument("--all", action="store_true", help="run the full 28-model roster")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="pythia-70m, 2 concepts, 16 pairs, 2 targets")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.smoke:
        models, concepts = ["EleutherAI/pythia-70m"], CONCEPTS_17[:2]
    elif args.model:
        models, concepts = [args.model], CONCEPTS_17
    elif args.all:
        models, concepts = BASE_28, CONCEPTS_17
    else:
        models, concepts = [], CONCEPTS_17

    for mid in models:
        model, tok, device = load_model(mid)
        try:
            run_for_model(mid, model, tok, device, args.batch_size,
                          concepts, smoke=args.smoke)
        finally:
            release(model)

    if args.finalize or args.smoke or args.all:
        finalize(smoke=args.smoke)


if __name__ == "__main__":
    main()
