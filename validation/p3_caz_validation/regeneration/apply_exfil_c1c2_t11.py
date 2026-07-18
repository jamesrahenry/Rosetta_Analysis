#!/usr/bin/env python3
"""Apply-pass recompute: C1/C2 regressions + Table 11 enrichment + C4 family
enrichment, in two states — 'defective' (frozen paper-n250 exfiltration) and
'corrected' (rerun exfiltration) — over the local paper_n250 mirror.

Validation contract: the 'defective' state must reproduce the published
values (C1 is_caz_peak +0.305; C2 +0.313; Table 11 0.517/0.140, ratio
3.60–3.69, p=9.47e-141-scale) before the 'corrected' numbers are trusted
as the ledger entries. Replicates p3_c1_c2_c4_results.json's regression
spec: OLS with is_caz_peak = (layer == sweep's caz_peak); C2 filters
layers with non-positive local Fisher separation before log.
OLS p-values computed manually (no statsmodels in this venv).
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

for _p in (str(Path.home()/"rosetta_tools"), str(Path.home()/"Source"/"Rosetta_Program"/"rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
MODELS = ["EleutherAI_pythia_70m","EleutherAI_pythia_160m","EleutherAI_pythia_410m","EleutherAI_pythia_1b",
"EleutherAI_pythia_1.4b","EleutherAI_pythia_2.8b","EleutherAI_pythia_6.9b","EleutherAI_pythia_12b",
"openai_community_gpt2","openai_community_gpt2_medium","openai_community_gpt2_large","openai_community_gpt2_xl",
"facebook_opt_125m","facebook_opt_350m","facebook_opt_1.3b","facebook_opt_2.7b","facebook_opt_6.7b",
"Qwen_Qwen2.5_0.5B","Qwen_Qwen2.5_1.5B","Qwen_Qwen2.5_3B","Qwen_Qwen2.5_7B","Qwen_Qwen2.5_14B",
"google_gemma_2_2b","google_gemma_2_9b","meta_llama_Llama_3.2_1B","meta_llama_Llama_3.2_3B",
"mistralai_Mistral_7B_v0.3","microsoft_phi_2"]
CONCEPTS = ["credibility","negation","causation","temporal_order","sentiment","certainty",
"moral_valence","specificity","plurality","agency","formality","threat_severity",
"authorization","urgency","sarcasm","deception","exfiltration"]
FAMILY = {"EleutherAI_pythia":"pythia","openai_community_gpt2":"gpt2","facebook_opt":"opt",
          "Qwen_Qwen2.5":"qwen2","google_gemma":"gemma2","meta_llama":"llama3",
          "mistralai":"mistral","microsoft":"phi"}
ROOT = Path.home()/"rosetta_data/paper_n250"
DEFECTIVE = Path.home()/"rosetta_data/_defective_exfiltration"

def fam(m):
    for pre, f in FAMILY.items():
        if m.startswith(pre): return f
    raise ValueError(m)

def load_rows(state):
    rows = []
    for m in MODELS:
        for c in CONCEPTS:
            base = DEFECTIVE/m if (state=="defective" and c=="exfiltration") else ROOT/m
            gs = json.load(open(base/f"ablation_global_sweep_{c}.json"))
            caz = json.load(open(base/f"caz_{c}.json"))
            mets = caz["layer_data"]["metrics"]
            sep_by_layer = {mm["layer"]: mm["separation_fisher"] for mm in mets}
            peak = gs["caz_peak"]
            for L in gs["layers"]:
                rows.append(dict(model=m, family=fam(m), concept=c, layer=L["layer"],
                                 depth_pct=L["depth_pct"], red=L["global_sep_reduction"],
                                 is_peak=1.0 if L["layer"]==peak else 0.0,
                                 local_sep=sep_by_layer.get(L["layer"], np.nan)))
    return rows

def ols(y, X):
    """OLS with intercept; returns coefs, p-values (t-test)."""
    X1 = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1@beta
    dof = len(y) - X1.shape[1]
    s2 = resid@resid/dof
    cov = s2*np.linalg.inv(X1.T@X1)
    se = np.sqrt(np.diag(cov))
    t = beta/se
    p = 2*stats.t.sf(np.abs(t), dof)
    return beta, p

def analyze(state):
    rows = load_rows(state)
    y = np.array([r["red"] for r in rows])
    depth = np.array([r["depth_pct"] for r in rows])
    peak = np.array([r["is_peak"] for r in rows])
    print(f"\n===== {state}: n rows = {len(rows)}")
    # C1
    b, p = ols(y, [depth, peak])
    print(f"C1: coef_depth={b[1]:.6f} (p={p[1]:.2e})  coef_is_caz_peak={b[2]:.4f} (p={p[2]:.2e})")
    # decile diffs
    dec = np.percentile(depth, np.arange(0,101,10))
    diffs = []
    for i in range(10):
        m_ = (depth>=dec[i]) & (depth<=dec[i+1] if i==9 else depth<dec[i+1])
        if peak[m_].sum()>0 and (1-peak[m_]).sum()>0:
            diffs.append(y[m_][peak[m_]==1].mean() - y[m_][peak[m_]==0].mean())
    print(f"C1 depth-decile mean diff = {np.mean(diffs):.4f} ({len(diffs)}/10 deciles, all>0: {all(d>0 for d in diffs)})")
    # C2
    ls = np.array([r["local_sep"] for r in rows])
    ok = np.isfinite(ls) & (ls>0)
    b2, p2 = ols(y[ok], [np.log(ls[ok]), peak[ok]])
    print(f"C2: n={ok.sum()}  coef_log_sep={b2[1]:.4f} (p={p2[1]:.2e})  coef_is_caz_peak={b2[2]:.4f} (p={p2[2]:.2e})")
    # Table 11: peak vs non-CAZ layers. non-CAZ = outside [caz_start-ish]? Published uses
    # scored regions; approximate published def: peak rows vs rows not inside ANY scored region.
    # Use rosetta_tools scored regions for exact membership.
    from rosetta_tools.reporting import load_results_dir, load_scored_region_df
    import pandas as pd
    dirs = [ROOT/m for m in MODELS]
    layer_df = load_results_dir(dirs)
    layer_df = layer_df[layer_df["concept"].isin(CONCEPTS)]
    if state=="defective":
        # swap exfiltration layer rows for defective ones
        ldf_def = load_results_dir([DEFECTIVE/m for m in MODELS])
        ldf_def = ldf_def[ldf_def["concept"]=="exfiltration"]
        layer_df = pd.concat([layer_df[layer_df["concept"]!="exfiltration"], ldf_def])
    region_df = load_scored_region_df(layer_df, min_prominence_frac=0.005)
    def norm(x): return x.replace("/","_").replace("-","_").replace(".","_")
    spans2 = {}
    for _, rr in region_df.iterrows():
        spans2.setdefault((norm(rr["model_id"]), rr["concept"]), []).append((rr["start"], rr["end"]))
    in_caz = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        for (s_, e_) in spans2.get((norm(r["model"]), r["concept"]), []):
            if s_ <= r["layer"] <= e_:
                in_caz[i] = True
                break
    peak_mask = peak==1
    noncaz_mask = (~in_caz) & (peak==0)
    mp, mn = y[peak_mask].mean(), y[noncaz_mask].mean()
    t, pv = stats.ttest_ind(y[peak_mask], y[noncaz_mask], equal_var=False)
    print(f"T11: mean@peak={mp:.3f} (n={peak_mask.sum()})  mean@nonCAZ={mn:.3f} (n={noncaz_mask.sum()})  ratio={mp/mn:.2f}  Welch p={pv:.3e}")
    # C4 family enrichment: per-family ratio + bootstrap over 8 families
    fams = sorted(set(r["family"] for r in rows))
    farr = np.array([r["family"] for r in rows])
    ratios = {}
    for f in fams:
        fm = farr==f
        ratios[f] = y[fm & peak_mask].mean()/y[fm & noncaz_mask].mean()
    rng = np.random.default_rng(0)
    boots = [np.mean([ratios[f] for f in rng.choice(fams, len(fams), replace=True)]) for _ in range(10000)]
    print(f"C4 family enrichment: per-family {dict((k, round(v,2)) for k,v in ratios.items())}")
    print(f"  bootstrap mean {np.mean(boots):.2f} CI95 [{np.percentile(boots,2.5):.2f}, {np.percentile(boots,97.5):.2f}]")
    return dict(state=state, n=len(rows), c1_coef=float(b[2]), c1_p=float(p[2]),
                c1_decile_mean=float(np.mean(diffs)), c2_coef=float(b2[2]), c2_p=float(p2[2]),
                t11_peak=float(mp), t11_noncaz=float(mn), t11_ratio=float(mp/mn), t11_p=float(pv),
                c4_family_ratios={k: float(v) for k,v in ratios.items()},
                c4_boot_mean=float(np.mean(boots)),
                c4_boot_ci=[float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))])

out = {"job":"apply-pass C1/C2/T11/C4-enrichment, defective-vs-corrected exfiltration"}
out["defective"] = analyze("defective")
out["corrected"] = analyze("corrected")
import datetime
out["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
dst = Path(__file__).parent/"results/apply_exfil_c1c2_t11.json"
dst.write_text(json.dumps(out, indent=1))
print("\nsaved", dst)
