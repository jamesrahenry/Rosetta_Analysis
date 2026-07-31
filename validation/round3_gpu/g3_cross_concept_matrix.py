#!/usr/bin/env python3
"""G3 — 17x17 cross-concept ablation matrix (P3, te256b3c objection 5 companion).

The structured-direction null the random-vector control cannot provide: at
each target concept A's dominant CAZ peak, ablate every concept B's stored
DOM direction *at that same layer* (from caz_<B>.json metrics — no direction
estimation needed) and measure A's final-layer separation reduction.

Diagonal (B == A) reproduces the standard concept-direction ablation;
off-diagonal cells quantify how much suppression a *different* structured,
concept-bearing direction causes at the same site. Compare against the
existing random-direction control (§6.1: mean random reduction 0.002).

First pass: the 5-model permutation-null subset (caz-validation §4.1).
Full 28-model matrix only if the subset is interesting (plan gate).

Outputs: per-model shards; g3_cross_concept_matrix.json to HF
paper_n250/_round3_gpu/g3/ at finalize.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import (
    BASE_28, CONCEPTS_17, G3_SUBSET, OUT_ROOT, fisher_separation_nd,
    dom_matrix, hf_upload, hf_verify, load_caz, log, peak_layer, shard_done,
    shard_write, slugify,
)
from forward_utils import (
    ablated_contrastive_acts, calibrate_offset, contrastive_acts, load_model,
    release,
)

JOB = "g3"
N_PAIRS = 250
# exfiltration's corrected pool is 249 pairs (EXFILTRATION_RERUN_SPEC §1a) —
# see g2_split_pair_ablation.py's identical override.
N_PAIRS_OVERRIDE = {"exfiltration": 249}


def run_for_model(model_id: str, model, tok, device, batch_size: int,
                  concepts: list[str], smoke: bool = False) -> dict | None:
    slug = slugify(model_id)
    key = slug + ("_smoke" if smoke else "")
    if (done := shard_done(JOB, key)) is not None:
        log.info("[g3] %s already done — skipping", slug)
        return done

    offset = calibrate_offset(model, tok, device, slug, concepts[0], batch_size)

    from rosetta_tools.dataset import load_concept_pairs, texts_by_label

    # Preload every concept's per-layer DOM matrix once (source directions).
    doms = {c: dom_matrix(load_caz(slug, c)) for c in concepts}

    rows = []
    t0 = time.time()
    for target in concepts:
        caz = load_caz(slug, target)
        pk = peak_layer(caz)
        n_layers = caz["layer_data"]["n_layers"]

        n_pairs = 16 if smoke else N_PAIRS_OVERRIDE.get(target, N_PAIRS)
        pairs = load_concept_pairs(target, n=n_pairs, split="train")
        pos, neg = texts_by_label(pairs)
        acts = contrastive_acts(model, tok, pos, neg, device, batch_size)
        base_nd = fisher_separation_nd(*acts[-1])

        for source in concepts:
            u = doms[source][pk]  # source concept's stored direction AT TARGET'S PEAK
            abl = ablated_contrastive_acts(
                model, tok, pos, neg, device, batch_size, u, pk, offset)
            red = 1.0 - fisher_separation_nd(*abl[-1]) / max(base_nd, 1e-12)
            rows.append({
                "model_id": model_id, "target": target, "source": source,
                "peak_layer": pk, "n_layers": n_layers, "offset": offset,
                "baseline_final_sep_nd": base_nd,
                "reduction_nd": red,
                "cos_source_target_at_peak":
                    float(abs(np.dot(doms[source][pk], doms[target][pk]))),
            })
        log.info("[g3] %s target=%s done (%d sources, base=%.3f)",
                 slug, target, len(concepts), base_nd)

    payload = {"model_id": model_id, "elapsed_s": time.time() - t0, "rows": rows}
    shard_write(JOB, key, payload)
    return payload


def finalize(smoke: bool = False) -> None:
    from common import CKPT_ROOT
    suffix = "_smoke" if smoke else ""
    shards = sorted((CKPT_ROOT / JOB).glob(f"*{suffix}.json"))
    if not smoke:
        shards = [s for s in shards if not s.stem.endswith("_smoke")]
    rows = []
    for s in shards:
        rows.extend(json.loads(s.read_text())["rows"])

    diag = [r["reduction_nd"] for r in rows if r["source"] == r["target"]]
    offd = [r["reduction_nd"] for r in rows if r["source"] != r["target"]]
    out = {
        "job": JOB, "n_models": len(shards), "n_rows": len(rows),
        "design": "ablate concept B's stored DOM direction at concept A's "
                  "dominant CAZ peak; measure A's final-layer trace-Fisher "
                  "reduction over A's 250 train pairs",
        "summary": {
            "diagonal_mean_reduction": float(np.mean(diag)) if diag else None,
            "offdiag_mean_reduction": float(np.mean(offd)) if offd else None,
            "offdiag_median_reduction": float(np.median(offd)) if offd else None,
            "offdiag_p90_reduction":
                float(np.percentile(offd, 90)) if offd else None,
            "reference_random_direction_mean": 0.002,
        },
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fname = f"g3_cross_concept_matrix{suffix}.json"
    fpath = OUT_ROOT / fname
    fpath.write_text(json.dumps(out, indent=1))
    hf_upload(JOB, fpath)
    hf_verify(JOB, [fname])
    log.info("[g3] finalized: diag=%.3f offdiag=%.3f (n=%d)",
             out["summary"]["diagonal_mean_reduction"] or -1,
             out["summary"]["offdiag_mean_reduction"] or -1, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--subset", action="store_true", help="the 5-model §4.1 subset")
    ap.add_argument("--full", action="store_true", help="all 28 base models")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.smoke:
        models, concepts = ["EleutherAI/pythia-70m"], CONCEPTS_17[:3]
    elif args.model:
        models, concepts = [args.model], CONCEPTS_17
    elif args.full:
        models, concepts = BASE_28, CONCEPTS_17
    elif args.subset:
        models, concepts = G3_SUBSET, CONCEPTS_17
    else:
        models, concepts = [], CONCEPTS_17

    for mid in models:
        model, tok, device = load_model(mid)
        try:
            run_for_model(mid, model, tok, device, args.batch_size,
                          concepts, smoke=args.smoke)
        finally:
            release(model)

    if args.finalize or args.smoke or args.subset or args.full:
        finalize(smoke=args.smoke)


if __name__ == "__main__":
    main()
