"""
eigengap_probe.py — Davis-Kahan/Wedin spectral probe of the Phase-A floors.

Plan of record: Rosetta_Program/papers/prh-validation/EIGENGAP_PROBE_PLAN.md
(hypotheses H1-H4, statistics §3.1-3.6, tests §4, guardrails §7). CPU-only,
reads the local paper_n250 mirror at ~/rosetta_data/models/ directly (A-E/G/H;
cluster F has no local peak-layer calibration and is excluded from spectra —
H1 runs on 7 of 8 floor points).

Spectrum convention LOCKED to nullfloor_analysis.py: singular values of the
mean-centred [n=500, d] peak-layer calibration, float64 (real_spectrum()).
DOM = peak-layer difference-of-means from caz_<concept>.json, as in
common.load_dom_and_peak, read locally.

Outputs: eigengap_probe_out/{spectral_stats.json,summary.json,
figures/flatness_vs_floor.png,figures/wedin_overlay.png}
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common import ROSTER, CONCEPTS_17, cross_family_same_dim_pairs

LOCAL_MODELS = Path.home() / "rosetta_data" / "models"
OUT = Path(__file__).parent / "eigengap_probe_out"
ALIGN_CSV = Path.home() / "rosetta_data" / "paper_n250" / "_alignment" / "prh_primary_xfam_samedim_C17.csv"

DIMLET = {768: "A", 1024: "G", 2048: "B", 2560: "H", 3584: "D", 4096: "C", 5120: "E", 8192: "F"}

# Ground truth — PHASE_A_NULLFLOOR_RESULTS.md §1 (full-coverage floors) and
# §4.5 (cross-concept table). F excluded from spectra (no local calibration).
FLOOR = {"A": 0.3223, "G": 0.3511, "B": 0.4173, "H": 0.4542, "D": 0.5225, "C": 0.4588, "E": 0.4985, "F": 0.4614}
CLSAME = {"A": 0.71, "G": 0.68, "B": 0.68, "H": 0.66, "D": 0.39, "C": 0.70, "E": 0.45, "F": 0.67}
# n-sweep v2 (PHASE_A §2), fixed-spectrum floor column, cluster B d=2048
NSWEEP = [(100, 20.5, 0.6364), (250, 8.2, 0.5219), (500, 4.1, 0.4179)]

TOPK_ENERGY = (1, 5, 17, 50)
OVERLAP_KS = (17, 50)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_local(slug, concept):
    """(dom float64 [d] | None, cal float64 [n,d]) from the local paper_n250 mirror."""
    caz = json.load(open(LOCAL_MODELS / slug / f"caz_{concept}.json"))
    ld = caz["layer_data"]
    pm = next(m for m in ld["metrics"] if m["layer"] == ld["peak_layer"])
    dom = np.asarray(pm["dom_vector"], np.float64) if "dom_vector" in pm else None
    cal = np.load(LOCAL_MODELS / slug / f"calibration_{concept}.npy").astype(np.float64)
    return dom, cal


def spectral_stats(cc, dom):
    """All per-(model,concept) statistics from one centred calibration matrix."""
    U, S, Vt = np.linalg.svd(cc, full_matrices=False)
    lam = S ** 2
    tot = lam.sum()
    p = lam / tot
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())      # (Σλ)²/Σλ²
    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    effrank = float(np.exp(ent))
    topk = {f"top{k}_energy": float(lam[:k].sum() / tot) for k in TOPK_ENERGY}
    g = (lam[:-1] - lam[1:]) / lam[0]
    st = dict(participation_ratio=pr, effective_rank=effrank, **topk,
              max_gap_top50=float(g[:50].max()), max_gap_idx=int(g[:50].argmax()),
              min_gap_top16=float(g[:16].min()))
    if dom is not None:
        dh = dom / np.linalg.norm(dom)
        m = (Vt @ dh) ** 2                      # mass on each covariance eigvec
        st["dom_rowspace_frac"] = float(m.sum())
        order = np.argsort(m)[::-1]
        cum = np.cumsum(m[order]) / m.sum()
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        widx = float((np.arange(len(m)) * m).sum() / m.sum())
        st["dom_n90"] = n90                     # eigvecs holding 90% of DOM mass
        st["dom_mean_index"] = widx             # mass-weighted spectral depth
        j = min(int(round(widx)), len(g) - 1)
        st["dom_local_gap"] = float(g[j])       # DK denominator proxy at DOM depth
    return st, Vt


def subspace_overlap(Va, Vb, k):
    """Mean cos^2 principal angle between top-k right-singular subspaces."""
    M = Va[:k] @ Vb[:k].T
    return float((M ** 2).sum() / k)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    slugs = [s for s, (fam, d) in ROSTER.items() if DIMLET[d] != "F"]
    per_mc, per_model = {}, {}
    pair_overlap = {}

    clusters = {}
    for s in slugs:
        clusters.setdefault(DIMLET[ROSTER[s][1]], []).append(s)

    for L, members in sorted(clusters.items()):
        d = ROSTER[members[0]][1]
        upairs = sorted({tuple(sorted(p)) for p in cross_family_same_dim_pairs(members)})
        log(f"cluster {L} (d={d}): {len(members)} models, {len(upairs)} unordered cross-fam pairs")
        for con in CONCEPTS_17:
            Vts = {}
            for slug in members:
                dom, cal = load_local(slug, con)
                cc = cal - cal.mean(0)
                st, Vt = spectral_stats(cc, dom)
                per_mc[f"{slug}|{con}"] = st
                Vts[slug] = Vt
            for a, b in upairs:
                rec = pair_overlap.setdefault((a, b), {k: [] for k in OVERLAP_KS})
                for k in OVERLAP_KS:
                    rec[k].append(subspace_overlap(Vts[a], Vts[b], k))
            del Vts

    scalar_keys = [k for k in next(iter(per_mc.values())) if k != "max_gap_idx"]
    for slug in slugs:
        rows = [per_mc[f"{slug}|{c}"] for c in CONCEPTS_17]
        per_model[slug] = {k: float(np.mean([r[k] for r in rows if k in r])) for k in scalar_keys}
        per_model[slug]["cluster"] = DIMLET[ROSTER[slug][1]]

    cluster_stats = {}
    for L, members in clusters.items():
        cluster_stats[L] = {k: float(np.mean([per_model[s][k] for s in members])) for k in scalar_keys}
        ovs = [(a, b) for (a, b) in pair_overlap if DIMLET[ROSTER[a][1]] == L]
        for k in OVERLAP_KS:
            cluster_stats[L][f"pair_overlap_k{k}"] = float(np.mean([np.mean(pair_overlap[p][k]) for p in ovs]))

    # ---- T1: cluster stat vs floor (7 points), baseline = d (n fixed at 500,
    # so d/n ranks identically to d) ------------------------------------------
    Ls = sorted(cluster_stats)
    floors = [FLOOR[L] for L in Ls]
    dims = [ROSTER[clusters[L][0]][1] for L in Ls]
    t1 = {"baseline_d": dict(zip(("rho", "p"), map(float, spearmanr(dims, floors))))}
    for k in scalar_keys + [f"pair_overlap_k{k}" for k in OVERLAP_KS]:
        vals = [cluster_stats[L][k] for L in Ls]
        t1[k] = dict(zip(("rho", "p"), map(float, spearmanr(vals, floors))))

    # ---- T2: D exception + per-model ranking + E vintage --------------------
    clsame = [CLSAME[L] for L in Ls]
    t2 = {"clsame_vs": {}}
    for k in scalar_keys + [f"pair_overlap_k{k}" for k in OVERLAP_KS]:
        vals = [cluster_stats[L][k] for L in Ls]
        rho_all = spearmanr(vals, clsame)
        noD = [i for i, L in enumerate(Ls) if L != "D"]
        rho_noD = spearmanr([vals[i] for i in noD], [clsame[i] for i in noD])
        t2["clsame_vs"][k] = dict(rho=float(rho_all[0]), rho_without_D=float(rho_noD[0]))
    rank_pr = sorted(per_model, key=lambda s: per_model[s]["participation_ratio"], reverse=True)
    t2["pr_ranking_top8_flattest"] = [(s, round(per_model[s]["participation_ratio"], 1)) for s in rank_pr[:8]]
    t2["gemma_pr_rank"] = {s: rank_pr.index(s) + 1 for s in per_model if "gemma" in s}
    t2["E_pair_overlaps_k17"] = {f"{a}~{b}": round(float(np.mean(pair_overlap[(a, b)][17])), 4)
                                 for (a, b) in pair_overlap if DIMLET[ROSTER[a][1]] == "E"}
    t2["D_pair_overlaps_k17"] = {f"{a}~{b}": round(float(np.mean(pair_overlap[(a, b)][17])), 4)
                                 for (a, b) in pair_overlap if DIMLET[ROSTER[a][1]] == "D"}

    # ---- T3: per-concept margin vs DOM-local gap, within cluster ------------
    t3 = {}
    if ALIGN_CSV.exists():
        import csv
        rows = list(csv.DictReader(open(ALIGN_CSV)))
        con_cl = {}
        for r in rows:
            L = DIMLET[int(r["dim"])]
            if L == "F":
                continue
            con_cl.setdefault((L, r["concept"]), []).append(float(r["aligned"]))
        for L in Ls:
            xs, ys = [], []
            for con in CONCEPTS_17:
                vals = con_cl.get((L, con))
                if not vals:
                    continue
                margin = float(np.mean(vals)) - FLOOR[L]
                gaps = [per_mc[f"{s}|{con}"].get("dom_local_gap") for s in clusters[L]]
                gaps = [g for g in gaps if g is not None]
                if gaps:
                    xs.append(float(np.mean(gaps))); ys.append(margin)
            if len(xs) > 3:
                t3[L] = dict(n=len(xs), rho=float(spearmanr(xs, ys)[0]))
        pooled = [(t3[L]["rho"], t3[L]["n"]) for L in t3]
        t3["mean_rho"] = float(np.mean([r for r, _ in pooled]))
    else:
        t3["skipped"] = "alignment CSV not found"

    # ---- T4: Wedin/YWS functional form on n-sweep v2 (3 points) -------------
    ln_f = np.log([f for _, _, f in NSWEEP]); ln_dn = np.log([dn for _, dn, _ in NSWEEP])
    slope = float(np.polyfit(ln_dn, ln_f, 1)[0])
    c_at = [f / np.sqrt(dn) for _, dn, f in NSWEEP]
    t4 = dict(observed_exponent=slope, wedin_sqrt_exponent=0.5,
              sqrt_form_constant_per_point=[round(float(c), 4) for c in c_at],
              verdict="sqrt-form " + ("consistent" if abs(slope - 0.5) < 0.1 else
                                      f"NOT consistent (exponent {slope:.2f} vs 0.5)"))

    summary = dict(
        generated_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        n_models=len(slugs), clusters={L: len(m) for L, m in clusters.items()},
        note="F excluded (no local peak-layer calibration); n=500 fixed so d/n == d as a ranker",
        cluster_stats=cluster_stats, floors=FLOOR, clsame=CLSAME,
        T1_floor_prediction=t1, T2_D_exception=t2, T3_margin_vs_domgap=t3, T4_wedin_form=t4)
    (OUT / "spectral_stats.json").write_text(json.dumps(
        dict(per_model=per_model, per_model_concept=per_mc), indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        best = max((k for k in t1 if k != "baseline_d"), key=lambda k: abs(t1[k]["rho"]))
        xs = [cluster_stats[L][best] for L in Ls]
        ax[0].scatter(xs, floors)
        for L, x, y in zip(Ls, xs, floors):
            ax[0].annotate(L, (x, y), textcoords="offset points", xytext=(4, 4))
        ax[0].set_xlabel(best); ax[0].set_ylabel("Phase-A floor")
        ax[0].set_title(f"T1 best: {best} (rho={t1[best]['rho']:+.2f}; baseline d rho={t1['baseline_d']['rho']:+.2f})")
        dns = np.linspace(3, 22, 50)
        c = float(np.mean(c_at))
        ax[1].plot(dns, c * np.sqrt(dns), "--", label=f"Wedin sqrt-form c={c:.3f}")
        ax[1].plot(dns, np.exp(np.polyval(np.polyfit(ln_dn, ln_f, 1), np.log(dns))), "-",
                   label=f"observed power-law exp={slope:.2f}")
        ax[1].scatter([dn for _, dn, _ in NSWEEP], [f for _, _, f in NSWEEP], zorder=3)
        ax[1].set_xlabel("d/n"); ax[1].set_ylabel("fixed-spectrum floor"); ax[1].legend()
        ax[1].set_title("T4: n-sweep v2 vs Wedin form")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "flatness_vs_floor.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log(f"figures skipped: {e}")

    log(f"T1 baseline d: rho={t1['baseline_d']['rho']:+.3f}")
    for k in sorted(t1, key=lambda k: -abs(t1[k]["rho"]))[:6]:
        log(f"T1 {k}: rho={t1[k]['rho']:+.3f}")
    log(f"T2 gemma PR ranks (of {len(slugs)}): {t2['gemma_pr_rank']}")
    log(f"T2 E pair overlaps k17: {t2['E_pair_overlaps_k17']}")
    log(f"T3: {t3}")
    log(f"T4: {t4['verdict']}")
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
