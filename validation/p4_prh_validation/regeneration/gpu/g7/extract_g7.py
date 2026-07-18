#!/usr/bin/env python3
"""G7 — human-written calibration subset: extraction + zero-PCA Procrustes
spot-check (P4, t3ef5f65 / ROUND3_COMPUTE_PLAN.md G7).

Reruns the primary alignment methodology (§2.3-2.4) with human-written
calibration data in place of RCP's LLM-generated pairs, for 3 concepts
(sentiment: SST-5; negation: Conan Doyle/Gutenberg; temporal_order:
Wikipedia) — see README.md in this directory for sourcing, licensing, and
important caveats (unpaired-by-topic design, lexical-marker-classified
labels, matched-LLM-domain-control not yet built).

Uses each model's ALREADY-KNOWN peak layer for the same concept (from the
stored caz_<concept>.json, same as the primary corpus) rather than
re-detecting peaks on this new data — this isolates "does the alignment
survive a different calibration source" from "does peak-layer selection
itself change," which is a separate question (§3.6) already addressed.

Stage A (GPU, per model): extract last-token activations for all G7 texts
(3 concepts' pos+neg pools) at each concept's stored peak layer; save one
.npz shard per model, upload to HF.
Stage B (no GPU): for every ordered cross-family same-dimension pair in
the alignment roster (clusters A-E) x 3 concepts: compute NEW DOM vectors
from the G7 activations, fit Procrustes R on the G7 calibration matrices
(scipy orthogonal_procrustes, float64, zero-PCA/same-dim, matching §2.4
exactly except for the data source), report aligned cosine — the direct
spot-check against the primary corpus's per-concept means (§3.1: sentiment
0.9951, negation 0.9727, temporal_order 0.9834).

Written: 2026-07-17 UTC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
# canonical location: gpu/ (one level up) holds common.py + forward_utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    CKPT_ROOT, OUT_ROOT, alignment_roster_from_hf, family_of, hf_upload,
    hf_verify, load_caz, log, peak_layer, shard_done, shard_write,
)
from forward_utils import calibrate_offset, load_model, plain_acts, release

JOB = "g7"
CONCEPTS = ["sentiment", "negation", "temporal_order"]
PAIRS_DIR = Path(__file__).resolve().parent
ACTS_DIR = CKPT_ROOT / JOB / "acts"

PRIMARY_REFERENCE = {  # §3.1 per-concept aligned means, for the spot-check comparison
    "sentiment": 0.9951, "negation": 0.9727, "temporal_order": 0.9834,
}


# ---------------------------------------------------------------------------
# Load G7 pair pools
# ---------------------------------------------------------------------------


def load_g7_pairs(concept: str) -> tuple[list[str], list[str]]:
    """Returns (positive_texts, negative_texts) for one concept."""
    path = PAIRS_DIR / f"g7_{concept}_pairs.jsonl"
    pos, neg = [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        (pos if row["label"] == 1 else neg).append(row["text"])
    return pos, neg


# ---------------------------------------------------------------------------
# Stage A — per-model extraction
# ---------------------------------------------------------------------------


def extract_for_model(model_id: str, model, tok, device, batch_size: int,
                      upload_acts: bool = True, smoke: bool = False) -> None:
    from common import slugify
    slug = slugify(model_id)
    key = slug + ("_smoke" if smoke else "")
    if shard_done(JOB, key) is not None:
        log.info("[g7] %s acts already extracted — skipping", slug)
        return

    offset = calibrate_offset(model, tok, device, slug, "causation", batch_size)
    t0 = time.time()
    result = {}
    concepts = CONCEPTS[:1] if smoke else CONCEPTS
    for concept in concepts:
        pos, neg = load_g7_pairs(concept)
        if smoke:
            pos, neg = pos[:10], neg[:10]
        caz = load_caz(slug, concept)
        l_peak = peak_layer(caz) + offset
        pos_acts = plain_acts(model, tok, pos, device, batch_size)[l_peak]
        neg_acts = plain_acts(model, tok, neg, device, batch_size)[l_peak]
        result[concept] = {"pos": pos_acts, "neg": neg_acts, "peak_layer_no_offset": peak_layer(caz)}

    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    npz = ACTS_DIR / f"{key}.npz"
    save_kwargs = {"offset": np.int64(offset)}
    for concept, d in result.items():
        save_kwargs[f"{concept}_pos"] = d["pos"]
        save_kwargs[f"{concept}_neg"] = d["neg"]
        save_kwargs[f"{concept}_peak_no_offset"] = np.int64(d["peak_layer_no_offset"])
    np.savez_compressed(npz, **save_kwargs)
    if upload_acts and not smoke:
        hf_upload(JOB, npz)
    shard_write(JOB, key, {
        "model_id": model_id, "offset": offset, "elapsed_s": time.time() - t0,
        "npz": npz.name, "concepts": list(result.keys()),
    })
    log.info("[g7] %s extracted %d concepts in %.0fs", slug, len(result), time.time() - t0)


# ---------------------------------------------------------------------------
# Stage B — DOM vectors + pairwise Procrustes
# ---------------------------------------------------------------------------


def _dom(pos_acts: np.ndarray, neg_acts: np.ndarray) -> np.ndarray:
    d = pos_acts.mean(0) - neg_acts.mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def _aligned_cos(src_cal: np.ndarray, tgt_cal: np.ndarray,
                 dom_src: np.ndarray, dom_tgt: np.ndarray) -> float:
    from scipy.linalg import orthogonal_procrustes, svd as _svd
    src_c = src_cal.astype(np.float64) - src_cal.mean(0, dtype=np.float64)
    tgt_c = tgt_cal.astype(np.float64) - tgt_cal.mean(0, dtype=np.float64)
    if not (np.isfinite(src_c).all() and np.isfinite(tgt_c).all()):
        raise ValueError("non-finite values in calibration acts")
    try:
        R, _ = orthogonal_procrustes(tgt_c, src_c)
    except np.linalg.LinAlgError:
        u, _, vt = _svd(tgt_c.T @ src_c, lapack_driver="gesvd")
        R = u @ vt
    v = dom_tgt @ R
    den = np.linalg.norm(dom_src) * np.linalg.norm(v)
    return float(np.dot(dom_src, v) / den) if den > 1e-12 else 0.0


def pairwise(smoke: bool = False, smoke_roster: list[str] | None = None) -> None:
    suffix = "_smoke" if smoke else ""
    concepts = CONCEPTS[:1] if smoke else CONCEPTS
    if smoke:
        slugs = smoke_roster or []
    else:
        slugs = alignment_roster_from_hf()
        have = {p.stem for p in ACTS_DIR.glob("*.npz")}
        missing = [s for s in slugs if s not in have]
        if missing:
            log.warning("[g7] %d roster models missing acts (reported, not fatal): %s",
                        len(missing), missing)
        slugs = [s for s in slugs if s in have]

    meta = {}
    for s in slugs:
        caz = load_caz(s, "causation")
        meta[s] = {"hidden_dim": caz["hidden_dim"], "family": family_of(s)}

    shards = {s: np.load(ACTS_DIR / f"{s}{suffix}.npz") for s in slugs}

    rows = []
    for concept in concepts:
        for a in slugs:
            for b in slugs:
                if a == b or meta[a]["hidden_dim"] != meta[b]["hidden_dim"]:
                    continue
                if meta[a]["family"] == meta[b]["family"]:
                    continue
                za, zb = shards[a], shards[b]
                pos_a, neg_a = za[f"{concept}_pos"], za[f"{concept}_neg"]
                pos_b, neg_b = zb[f"{concept}_pos"], zb[f"{concept}_neg"]
                dom_a, dom_b = _dom(pos_a, neg_a), _dom(pos_b, neg_b)
                cal_a = np.concatenate([pos_a, neg_a], axis=0)
                cal_b = np.concatenate([pos_b, neg_b], axis=0)
                n_full = min(len(cal_a), len(cal_b))
                rows.append({
                    "src": a, "tgt": b, "concept": concept,
                    "hidden_dim": meta[a]["hidden_dim"],
                    "aligned_cos": _aligned_cos(cal_a[:n_full], cal_b[:n_full], dom_a, dom_b),
                    "raw_cos": float(np.dot(dom_a, dom_b)
                                     / (np.linalg.norm(dom_a) * np.linalg.norm(dom_b))),
                })
        log.info("[g7] pairwise: %s done", concept)

    per_concept = {}
    for concept in concepts:
        c_rows = [r["aligned_cos"] for r in rows if r["concept"] == concept]
        per_concept[concept] = {
            "mean_aligned_cos": float(np.mean(c_rows)) if c_rows else None,
            "n": len(c_rows),
            "primary_reference_mean": PRIMARY_REFERENCE.get(concept),
        }

    out = {
        "job": JOB, "n_rows": len(rows), "n_models": len(slugs),
        "population": "ordered cross-family same-dimension pairs, alignment "
                      "roster (clusters A-E)",
        "corpus_note": "Human-written calibration data (SST-5 sentiment, "
                       "Conan Doyle/Gutenberg negation, Wikipedia "
                       "temporal_order) in place of RCP's LLM-generated "
                       "pairs -- see README.md for sourcing, licensing, "
                       "and caveats (unpaired-by-topic design, "
                       "lexical-marker-classified negation/temporal_order "
                       "labels, matched-LLM-domain-control not yet built).",
        "per_concept": per_concept,
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fname = f"g7_human_written_alignment{suffix}.json"
    fpath = OUT_ROOT / fname
    fpath.write_text(json.dumps(out, indent=1))
    if not smoke:
        hf_upload(JOB, fpath)
        hf_verify(JOB, [fname])
    log.info("[g7] finalized: %s", json.dumps(per_concept, indent=1))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="stage A for one HF model id")
    ap.add_argument("--extract-all", action="store_true",
                    help="stage A for the full alignment roster")
    ap.add_argument("--pairwise", action="store_true", help="stage B")
    ap.add_argument("--smoke", action="store_true",
                    help="pythia-160m + gpt2 (768-dim cross-family pair), 1 concept, 20 texts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-upload-acts", action="store_true")
    args = ap.parse_args()

    smoke_models = ["EleutherAI/pythia-160m", "openai-community/gpt2"]

    if args.smoke:
        from common import slugify
        for mid in smoke_models:
            model, tok, device = load_model(mid)
            try:
                extract_for_model(mid, model, tok, device, args.batch_size,
                                  upload_acts=False, smoke=True)
            finally:
                release(model)
        pairwise(smoke=True, smoke_roster=[slugify(m) for m in smoke_models])
        return

    if args.model:
        model, tok, device = load_model(args.model)
        try:
            extract_for_model(args.model, model, tok, device, args.batch_size,
                              upload_acts=not args.no_upload_acts)
        finally:
            release(model)

    if args.extract_all:
        roster = alignment_roster_from_hf()
        for slug in roster:
            mid = load_caz(slug, "causation")["model_id"]
            model, tok, device = load_model(mid)
            try:
                extract_for_model(mid, model, tok, device, args.batch_size,
                                  upload_acts=not args.no_upload_acts)
            finally:
                release(model)

    if args.pairwise:
        pairwise()


if __name__ == "__main__":
    main()
