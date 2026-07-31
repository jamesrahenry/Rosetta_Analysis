#!/usr/bin/env python3
"""G5 — random-text calibration null at n=500 (P4, tbc29f76 item 3).

P4 §3.2's random-text null fit the Procrustes rotation on 200 neutral texts
while the primary analysis uses 500 concept-contrastive calibration rows —
flagged in §4.5 as un-size-matched. This job reruns the null at n=500 (and,
from the same corpus, an n=200 subsample so the size effect is measured
within-run).

NOTE ON TEXT PROVENANCE: the original 200-text corpus lives only in the
external analysis repo and is not in this checkout, so this run draws a NEW
neutral corpus — wikitext-103-raw-v1 validation split, seed 42, first 500
passages with 150-300 whitespace tokens. It tests the same hypothesis
(generic rotation sufficiency) at matched calibration size; its n=200
subsample is the bridge to the published 0.1484 figure. The text list is
uploaded alongside the results for exact reproducibility.

Stage A (GPU, per model): extract last-token activations for the 500 texts at
every layer; save one .npz shard per model, upload to HF.
Stage B (no GPU): for every ordered cross-family same-dimension pair in the
alignment roster (clusters A-E) x 17 concepts: mean-center both models'
random-text matrices at their own concept-peak layers, fit R
(scipy orthogonal_procrustes, float64, zero-PCA/same-dim), report
cos(dom_src, dom_tgt @ R) at n=500 and n=200.

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (
    CKPT_ROOT, CONCEPTS_17, OUT_ROOT, alignment_roster_from_hf, dom_matrix,
    family_of, hf_upload, hf_verify, load_caz, log, peak_layer, shard_done,
    shard_write,
)
from forward_utils import calibrate_offset, load_model, plain_acts, release

JOB = "g5"
N_TEXTS = 500
N_SUB = 200
SEED = 42
MIN_TOK, MAX_TOK = 150, 300
ACTS_DIR = CKPT_ROOT / JOB / "acts"
TEXTS_FILE = CKPT_ROOT / JOB / "neutral_texts.json"


# ---------------------------------------------------------------------------
# Neutral corpus (built once, shared by every model)
# ---------------------------------------------------------------------------


def build_neutral_texts(n: int = N_TEXTS) -> list[str]:
    if TEXTS_FILE.exists():
        return json.loads(TEXTS_FILE.read_text())["texts"]
    from datasets import load_dataset
    # "Salesforce/wikitext" = the canonical wikitext repo's full id; the
    # bare "wikitext" shorthand was removed in datasets 4.x and fails with
    # HfUriError on the GPU runner's stack. Same dataset, same revision.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ds))
    texts = []
    for i in order:
        t = ds[int(i)]["text"].strip()
        if MIN_TOK <= len(t.split()) <= MAX_TOK:
            texts.append(t)
            if len(texts) == n:
                break
    if len(texts) < n:
        raise RuntimeError(f"only {len(texts)}/{n} neutral texts found")
    TEXTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEXTS_FILE.write_text(json.dumps({
        "source": "wikitext-103-raw-v1/validation", "seed": SEED,
        "length_filter_ws_tokens": [MIN_TOK, MAX_TOK], "n": n, "texts": texts,
    }))
    return texts


# ---------------------------------------------------------------------------
# Stage A — per-model extraction
# ---------------------------------------------------------------------------


def extract_for_model(model_id: str, model, tok, device, batch_size: int,
                      upload_acts: bool = True, smoke: bool = False) -> None:
    from common import slugify
    slug = slugify(model_id)
    key = slug + ("_smoke" if smoke else "")
    if shard_done(JOB, key) is not None:
        log.info("[g5] %s acts already extracted — skipping", slug)
        return

    offset = calibrate_offset(model, tok, device, slug, "causation", batch_size)
    texts = build_neutral_texts(40 if smoke else N_TEXTS)
    t0 = time.time()
    acts = plain_acts(model, tok, texts, device, batch_size)  # [rows][n,d] f32
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    npz = ACTS_DIR / f"{key}.npz"
    np.savez_compressed(npz, acts=np.stack(acts), offset=np.int64(offset))
    if upload_acts and not smoke:
        hf_upload(JOB, npz)
    shard_write(JOB, key, {
        "model_id": model_id, "n_texts": len(texts), "n_rows": len(acts),
        "offset": offset, "elapsed_s": time.time() - t0, "npz": npz.name,
    })
    log.info("[g5] %s extracted %d rows in %.0fs", slug, len(acts), time.time() - t0)


# ---------------------------------------------------------------------------
# Stage B — pairwise Procrustes
# ---------------------------------------------------------------------------


def _load_acts(key: str) -> tuple[np.ndarray, int]:
    z = np.load(ACTS_DIR / f"{key}.npz")
    return z["acts"], int(z["offset"])


def _aligned_cos(src_cal: np.ndarray, tgt_cal: np.ndarray,
                 dom_src: np.ndarray, dom_tgt: np.ndarray) -> float:
    """Mirror rosetta_tools.alignment: R s.t. tgt_cal @ R ~= src_cal (both
    mean-centered, float64); aligned cosine = cos(dom_src, dom_tgt @ R)."""
    from scipy.linalg import orthogonal_procrustes, svd as _svd
    src_c = src_cal.astype(np.float64) - src_cal.mean(0, dtype=np.float64)
    tgt_c = tgt_cal.astype(np.float64) - tgt_cal.mean(0, dtype=np.float64)
    if not (np.isfinite(src_c).all() and np.isfinite(tgt_c).all()):
        raise ValueError("non-finite values in calibration acts — data problem, "
                         "not LAPACK flakiness; do not fall back")
    try:
        R, _ = orthogonal_procrustes(tgt_c, src_c)
    except np.linalg.LinAlgError:
        # gesdd (scipy's default driver) sporadically fails to converge on
        # valid matrices; gesvd is slower but robust — same fallback the
        # G6 battery uses. Same math as orthogonal_procrustes(tgt_c, src_c).
        u, _, vt = _svd(tgt_c.T @ src_c, lapack_driver="gesvd")
        R = u @ vt
    v = dom_tgt @ R
    den = np.linalg.norm(dom_src) * np.linalg.norm(v)
    return float(np.dot(dom_src, v) / den) if den > 1e-12 else 0.0


def pairwise(concepts: list[str], smoke: bool = False,
             smoke_roster: list[str] | None = None) -> None:
    suffix = "_smoke" if smoke else ""
    if smoke:
        slugs = smoke_roster or []
    else:
        slugs = alignment_roster_from_hf()
        have = {p.stem for p in ACTS_DIR.glob("*.npz")}
        missing = [s for s in slugs if s not in have]
        if missing:
            log.warning("[g5] %d roster models missing acts (reported, not fatal): %s",
                        len(missing), missing)
        slugs = [s for s in slugs if s in have]

    meta = {}
    for s in slugs:
        caz = load_caz(s, concepts[0])
        meta[s] = {"hidden_dim": caz["hidden_dim"], "family": family_of(s)}

    rows = []
    for a in slugs:
        for b in slugs:
            if a == b or meta[a]["hidden_dim"] != meta[b]["hidden_dim"]:
                continue
            if meta[a]["family"] == meta[b]["family"]:
                continue  # cross-family only, matching the published population
            acts_a, off_a = _load_acts(a + suffix)
            acts_b, off_b = _load_acts(b + suffix)
            for concept in concepts:
                caz_a, caz_b = load_caz(a, concept), load_caz(b, concept)
                la = peak_layer(caz_a) + off_a
                lb = peak_layer(caz_b) + off_b
                dom_a = dom_matrix(caz_a)[peak_layer(caz_a)]
                dom_b = dom_matrix(caz_b)[peak_layer(caz_b)]
                cal_a, cal_b = acts_a[la], acts_b[lb]
                n_full = min(len(cal_a), len(cal_b))
                row = {
                    "src": a, "tgt": b, "concept": concept,
                    "hidden_dim": meta[a]["hidden_dim"],
                    "aligned_cos_n500":
                        _aligned_cos(cal_a[:n_full], cal_b[:n_full], dom_a, dom_b),
                    "raw_cos": float(np.dot(dom_a, dom_b)
                                     / (np.linalg.norm(dom_a) * np.linalg.norm(dom_b))),
                }
                if n_full > N_SUB:
                    row["aligned_cos_n200"] = _aligned_cos(
                        cal_a[:N_SUB], cal_b[:N_SUB], dom_a, dom_b)
                rows.append(row)
        log.info("[g5] pairwise: %s done", a)

    c500 = [r["aligned_cos_n500"] for r in rows]
    c200 = [r["aligned_cos_n200"] for r in rows if "aligned_cos_n200" in r]
    out = {
        "job": JOB, "n_rows": len(rows), "n_models": len(slugs),
        "population": "ordered cross-family same-dimension pairs, alignment "
                      "roster (clusters A-E)",
        "corpus_note": "NEW neutral corpus (wikitext-103 validation, seed 42) — "
                       "not the original external n=200 texts; see module docstring",
        "summary": {
            "grand_mean_n500": float(np.mean(c500)) if c500 else None,
            "grand_sd_n500": float(np.std(c500)) if c500 else None,
            "grand_mean_n200_same_corpus": float(np.mean(c200)) if c200 else None,
            "published_n200_reference": 0.1484,
            "primary_reference": 0.9709,
        },
        "rows": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fname = f"g5_random_text_null_n500{suffix}.json"
    fpath = OUT_ROOT / fname
    fpath.write_text(json.dumps(out, indent=1))
    if not smoke:
        hf_upload(JOB, fpath)
        hf_upload(JOB, TEXTS_FILE)
        hf_verify(JOB, [fname, TEXTS_FILE.name])
    log.info("[g5] finalized: n500 mean=%s  n200(same corpus)=%s",
             out["summary"]["grand_mean_n500"],
             out["summary"]["grand_mean_n200_same_corpus"])


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="stage A for one HF model id")
    ap.add_argument("--extract-all", action="store_true",
                    help="stage A for the full alignment roster")
    ap.add_argument("--pairwise", action="store_true", help="stage B")
    ap.add_argument("--smoke", action="store_true",
                    help="pythia-160m + gpt2 (768-dim cross-family pair), 40 texts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-upload-acts", action="store_true")
    args = ap.parse_args()

    concepts = CONCEPTS_17[:2] if args.smoke else CONCEPTS_17
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
        pairwise(concepts, smoke=True,
                 smoke_roster=[slugify(m) for m in smoke_models])
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
        pairwise(concepts)


if __name__ == "__main__":
    main()
