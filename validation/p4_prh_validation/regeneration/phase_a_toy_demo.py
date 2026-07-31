#!/usr/bin/env python3
"""Phase-A toy demonstrator — the whole argument on synthetic data you can poke.

*Written: 2026-07-29 14:35 UTC by claude:p4-phaseA-dive, for James.*

Self-contained (numpy + scipy only, no downloads, ~60-90 s). The alignment operator
below is copied line-for-line from the paper's own `common.aligned_cosine`
(Rosetta_Analysis/validation/p4_prh_validation/regeneration/common.py), so
everything here goes through the identical math as P4 and Phase A.

Three demonstrations:

  DEMO 1  "The free superposition"      — pure noise reproduces the paper's old
          evidence signature (pre ~0, aligned ~0.9+, permuted-label null ~0).
          This is Phase-A §0 / review blocker P4-B1.

  DEMO 2  "The pairing was never used"  — scrambling which text is which
          (within class) does not hurt the headline metric. Phase-A §3 / P4-B3.

  DEMO 3  "The knob the old metric can't see" — two synthetic models whose
          concept ARRANGEMENTS are shared to a tunable degree (--shared 0..1).
          The old same-concept metric stays high (0.83-0.99) at EVERY setting
          of the knob — and its scrambled variant is dead flat ~0.90 — while
          cross-concept transfer tracks the truth, and a scrambled fit kills
          exactly the part that needed true correspondence. Phase-A §4.

Play suggestions:
    python phase_a_toy_demo.py                     # defaults
    python phase_a_toy_demo.py --noise 1.5         # noisier models
    python phase_a_toy_demo.py --d 2048            # wider activation space
    python phase_a_toy_demo.py --idio 0.2          # weaker text idiosyncrasy:
                                                   # watch cross-transfer degrade —
                                                   # correspondence is the fuel
"""
import argparse

import numpy as np
from scipy.linalg import orthogonal_procrustes


# ── the paper's operator, verbatim ──────────────────────────────────────────
def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d < 1e-12 else float(a @ b / d)


def aligned_cosine(dom_s, dom_t, cal_s, cal_t):
    """Rank-reduced orthogonal Procrustes — identical to common.aligned_cosine."""
    sc = np.asarray(cal_s, np.float64); sc = sc - sc.mean(0)
    tc = np.asarray(cal_t, np.float64); tc = tc - tc.mean(0)
    Q, _ = np.linalg.qr(np.hstack([tc.T, sc.T]))
    Rq, _ = orthogonal_procrustes(tc @ Q, sc @ Q)
    return cosine(dom_s @ Q, (np.asarray(dom_t, np.float64).ravel() @ Q) @ Rq)


def dom_of(cal):
    """The concept arrow: mean(present rows) − mean(absent rows)."""
    h = cal.shape[0] // 2
    return cal[:h].mean(0) - cal[h:].mean(0)


def scramble_within_class(cal, rng):
    """Shuffle which text is which, separately inside each class block.
    Leaves both class means — and therefore the arrow — exactly unchanged."""
    n = cal.shape[0]; h = n // 2
    out = cal.copy()
    out[:h] = cal[rng.permutation(h)]
    out[h:] = cal[rng.permutation(n - h) + h]
    return out


# ── DEMO 1+2: noise through the pipeline ────────────────────────────────────
def demo_noise(d, n, rng):
    A = rng.standard_normal((n, d))
    B = rng.standard_normal((n, d))          # totally unrelated to A
    da, db = dom_of(A), dom_of(B)

    pre = cosine(da, db)
    aligned = aligned_cosine(da, db, A, B)

    # paper-style permuted-label null: relabel rows at random, same clouds
    pa = dom_of(A[rng.permutation(n)])
    pb = dom_of(B[rng.permutation(n)])
    permuted = aligned_cosine(pa, pb, A, B)

    # within-class scramble: destroy the row pairing, keep the class blocks
    scr = aligned_cosine(da, db, A, scramble_within_class(B, rng))
    return pre, aligned, permuted, scr


# ── DEMO 3: two models with a tunable shared concept arrangement ────────────
def make_world(d, m, K, n, shared, noise, idio, rng):
    """Two synthetic 'models'.

    Semantic space is R^m. A common concept arrangement U (K unit directions)
    is blended per model with an independent arrangement E_M:

        U_M = unit_columns( shared·U + (1−shared)·E_M )

    shared=1: both models place their K concepts in the SAME relative
    arrangement (pairwise angles identical) — one rigid rotation can map all of
    model A's concept directions onto model B's.
    shared=0: arrangements unrelated — no single rotation exists, even though
    each individual concept is perfectly real in both models.

    Each model embeds semantic space into its own activation space through a
    private orthonormal frame Q_M (this is what Procrustes has to undo).

    Text i of concept k has label y=±1 and an idiosyncrasy vector w_i SHARED
    between the models — that shared w_i is what "the same text" means here.

        activation row = Q_M( y·u_kM + idio·w_i ) + noise·(private junk)
    """
    def unit_cols(X):
        return X / np.linalg.norm(X, axis=0, keepdims=True)

    U = np.linalg.qr(rng.standard_normal((m, K)))[0]
    frames, concepts = [], []
    for _ in range(2):
        E = np.linalg.qr(rng.standard_normal((m, K)))[0]
        concepts.append(unit_cols(shared * U + (1.0 - shared) * E))
        frames.append(np.linalg.qr(rng.standard_normal((d, m)))[0])

    y = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    cals = [[None] * K, [None] * K]
    for k in range(K):
        W = rng.standard_normal((n, m)) * idio          # shared per-text content
        for M in range(2):
            sem = np.outer(y, concepts[M][:, k]) + W
            cals[M][k] = sem @ frames[M].T + noise * rng.standard_normal((n, d))
    return cals


def demo_world(d, m, K, n, shared, noise, idio, seed):
    rng = np.random.default_rng(seed)
    cals = make_world(d, m, K, n, shared, noise, idio, rng)
    doms = [[dom_of(c) for c in cals[M]] for M in range(2)]

    same, same_scr, cross, cross_scr = [], [], [], []
    for X in range(K):
        Y = (X + 1) % K
        cA, cB = cals[0][X], cals[1][X]
        cB_scr = scramble_within_class(cB, rng)
        same.append(aligned_cosine(doms[0][X], doms[1][X], cA, cB))
        same_scr.append(aligned_cosine(doms[0][X], doms[1][X], cA, cB_scr))
        cross.append(aligned_cosine(doms[0][Y], doms[1][Y], cA, cB))
        cross_scr.append(aligned_cosine(doms[0][Y], doms[1][Y], cA, cB_scr))
    return tuple(float(np.mean(v)) for v in (same, same_scr, cross, cross_scr))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=768, help="activation dimension")
    ap.add_argument("--m", type=int, default=48, help="semantic (latent) dimension")
    ap.add_argument("--K", type=int, default=8, help="number of concepts")
    ap.add_argument("--n", type=int, default=300, help="calibration rows per concept")
    ap.add_argument("--noise", type=float, default=0.3, help="model-private noise")
    ap.add_argument("--idio", type=float, default=1.0,
                    help="per-text idiosyncrasy strength (the correspondence signal)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=" * 78)
    print("DEMO 1 — pure noise through the paper's pipeline (Phase-A §0, P4-B1)")
    print("=" * 78)
    print("Two clouds of pure random noise. No shared anything.\n")
    print(f"{'d':>6} {'pre-rotation':>13} {'ALIGNED':>9} {'permuted null':>14} {'scrambled':>10}")
    rng = np.random.default_rng(a.seed)
    for d in (768, 2048):
        pre, al, pm, sc = demo_noise(d, 500, rng)
        print(f"{d:>6} {pre:>13.3f} {al:>9.3f} {pm:>14.3f} {sc:>10.3f}")
    print("""
  The old evidence signature — pre ~0, aligned ~0.9+, permuted null ~0 — in
  full, from data containing nothing. The permuted-label null is ~0 by
  construction; it cannot tell this from real data. And the scrambled column
  previews DEMO 2: breaking the text pairing doesn't dent the number.
""")

    print("=" * 78)
    print(f"DEMO 3 — the shared-arrangement knob (Phase-A §4)   "
          f"[d={a.d}, m={a.m}, K={a.K}, n={a.n}, noise={a.noise}, idio={a.idio}]")
    print("=" * 78)
    print("""Two synthetic models. Each concept is REAL in both models at every knob
setting. The knob only controls whether the models place their concepts in the
same relative ARRANGEMENT (--shared 1) or in unrelated arrangements (--shared 0).
Fit the rotation on concept X; 'cross' = does it also carry concept X+1?
""")
    print(f"{'shared':>7} {'same-concept':>13} {'same(scrambled)':>16} "
          f"{'CROSS-CONCEPT':>14} {'cross(scrambled)':>17}")
    for shared in (0.0, 0.25, 0.5, 0.75, 1.0):
        s, ss, c, cs = demo_world(a.d, a.m, a.K, a.n, shared, a.noise, a.idio, a.seed)
        print(f"{shared:>7.2f} {s:>13.3f} {ss:>16.3f} {c:>14.3f} {cs:>17.3f}")
    print("""
  Read down the columns:

  * same-concept — the paper's old headline. It NEVER goes low: ~0.83 even at
    shared=0, a world where the two models' concept arrangements have nothing
    to do with each other. Handed this column alone, you cannot tell the
    worlds apart — its magnitude is mostly the in-basis superposition from
    DEMO 1, not shared structure. (It does drift upward with the knob — in
    this toy the true-pairing fit picks up some genuine sensitivity through
    the shared per-text content. The next column removes that.)

  * same(scrambled) — destroy the text pairing and refit: DEAD FLAT ~0.90 at
    every knob setting. This is the pure in-basis reading — the metric with
    the correspondence information gone — and it is completely blind to the
    truth. Note it BEATS the true-pairing number at low shared: honoring real
    correspondence can only constrain the fit, so the scramble scores higher.
    P4's real corpus shows exactly this signature (scrambled >= true, 8/8
    clusters) — which is what told us the headline wasn't measuring pairing.

  * CROSS-CONCEPT — rises with the knob, ~0.15 -> ~0.71. This is the quantity
    that actually responds to shared structure: a rotation learned from ONE
    concept's texts carries a DIFFERENT concept's direction only insofar as
    the models really share an arrangement.

  * cross(scrambled) — refit on correspondence-destroyed texts and transfer
    collapses to the within-model overlap floor (~0 here, because this toy's
    concepts are near-orthogonal inside each model). The GAP between cross and
    cross-scrambled is the correspondence-dependent signal.

  P4's real data through the identical machinery (Phase-A §1/§3/§4, cluster B):
      same-concept 0.986 | same scrambled 0.992 | cross_L 0.674 | cross scr 0.139
  Same shape as the upper-middle rows of this table: scrambled >= true on the
  headline, strong cross-transfer, and a scrambled-cross residue of 0.139 that
  Phase-A §4.6 shows is exactly the real concepts' within-model overlap
  free-ride (real concepts overlap ~0.28 inside a model; the toy's don't).
  The ~0.5 gap between cross and cross-scrambled, replicated across clusters,
  is the surviving PRH evidence. The 0.97 was never measuring it.
""")


if __name__ == "__main__":
    main()
