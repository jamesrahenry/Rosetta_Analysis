#!/usr/bin/env python3
"""Retrospective QA sweep over the P4 corpus (33 models x 17 concepts).
Two parts:
  1. Cross-model consistency (FREE, from the primary alignment artifact) — the
     check that catches a clean systematic label inversion (the split-half blind
     spot): for each concept x model, mean aligned cosine to its cross-family
     same-dim partners. A cleanly inverted model anti-aligns (mean < 0).
  2. Intrinsic split-half DOM reproducibility + peak separation, per model x
     concept, from the peak-layer calibration_<concept>.npy. Calibrates the
     QUALITY_THRESHOLDS and flags any other exfiltration-class defect before
     submission.
Checkpointed; streams + deletes calibration files. Output: qa_sweep_out/.
"""
import os, json, csv, time
from pathlib import Path
import numpy as np
from collections import defaultdict
from huggingface_hub import hf_hub_download

HF="james-ra-henry/Rosetta-Activations"; REV="paper-n250"; TREE="paper_n250"
STAGE=Path("/home/jhenry/Games2/rosetta_qa_stage"); OUT=Path("/home/jhenry/Games2/Eigan/Rosetta_Program/papers/prh-validation/scripts/qa_sweep_out")
SPLIT_HALF_THR=0.90

def slugify(mid): return mid.replace("/","_").replace("-","_")

def split_half(pos, neg, seed=0, n=5):
    pos=pos[np.isfinite(pos).all(1)]; neg=neg[np.isfinite(neg).all(1)]
    if len(pos)<4 or len(neg)<4: return None
    rng=np.random.default_rng(seed); cs=[]
    for _ in range(n):
        pi=rng.permutation(len(pos)); ni=rng.permutation(len(neg))
        ph,nh=len(pos)//2,len(neg)//2
        da=pos[pi[:ph]].mean(0)-neg[ni[:nh]].mean(0); db=pos[pi[ph:]].mean(0)-neg[ni[nh:]].mean(0)
        d=np.linalg.norm(da)*np.linalg.norm(db); cs.append(float(np.dot(da,db)/d) if d>1e-12 else 0.0)
    return float(np.mean(cs))

def separation(pos,neg,eps=1e-8):
    pos=pos[np.isfinite(pos).all(1)].astype(np.float64); neg=neg[np.isfinite(neg).all(1)].astype(np.float64)
    if len(pos)<2 or len(neg)<2: return 0.0
    cd=np.linalg.norm(pos.mean(0)-neg.mean(0)); ws=np.sqrt(0.5*(pos.var(0,ddof=1).sum()+neg.var(0,ddof=1).sum()))+eps
    return float(cd/ws)

def main():
    STAGE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    # ---- Part 1: cross-model consistency (from primary artifact) ----
    p=hf_hub_download(HF,f"{TREE}/_alignment/prh_primary_xfam_samedim_C17.csv",repo_type="dataset",revision=REV)
    rows=list(csv.DictReader(open(p)))
    bysc=defaultdict(list)
    for r in rows: bysc[(r['concept'],r['source'])].append(float(r['aligned']))
    xmodel=[]
    for (c,s),vs in bysc.items():
        m=float(np.mean(vs)); xmodel.append({"concept":c,"model":s,"mean_aligned_to_roster":m,"n":len(vs),
            "flag":("INVERSION" if m<0 else "low_outlier" if m<0.5 else "")})
    xmodel.sort(key=lambda r:r['mean_aligned_to_roster'])
    (OUT/"cross_model_consistency.json").write_text(json.dumps(xmodel,indent=1))
    flagged=[r for r in xmodel if r['flag']]
    print(f"[part1] cross-model consistency: {len(xmodel)} model-concepts, {len(flagged)} flagged", flush=True)
    for r in flagged[:20]: print(f"  {r['flag']:11s} {r['concept']:15s} {r['model']:32s} mean={r['mean_aligned_to_roster']:+.3f}", flush=True)
    if not flagged: print("  none below 0.5 / none inverted — corpus clean on cross-model consistency", flush=True)
    models=sorted({r['source'] for r in rows} | {r['target'] for r in rows})
    concepts=sorted({r['concept'] for r in rows})
    print(f"[part1] done ({time.time()-t0:.0f}s). Part 2: split-half over {len(models)}x{len(concepts)}...", flush=True)

    # ---- Part 2: intrinsic split-half + separation per model x concept ----
    intr=[]
    for mi,mid in enumerate(models):
        slug=slugify(mid)
        cpath=OUT/f"intr_{slug}.json"
        if cpath.exists(): intr+=json.loads(cpath.read_text()); continue
        mrows=[]
        for c in concepts:
            npy=None
            try:
                npy=hf_hub_download(HF,f"{TREE}/{slug}/calibration_{c}.npy",repo_type="dataset",revision=REV,local_dir=str(STAGE))
                a=np.load(npy).astype(np.float64); h=a.shape[0]//2; pos,neg=a[:h],a[h:]
                sh=split_half(pos,neg); sep=separation(pos,neg)
                mrows.append({"model":mid,"concept":c,"split_half":sh,"separation":sep,"n_pairs":h,
                    "flag":("split_half_unstable" if (sh is not None and sh<SPLIT_HALF_THR) else "")})
            except Exception as e:
                mrows.append({"model":mid,"concept":c,"error":str(e)[:60]})
            finally:
                if npy and os.path.exists(npy):
                    try: os.remove(npy)
                    except OSError: pass
        cpath.write_text(json.dumps(mrows,indent=1)); intr+=mrows
        nf=sum(1 for r in mrows if r.get('flag'))
        print(f"  [{mi+1}/{len(models)}] {slug:34s} {nf} flagged ({time.time()-t0:.0f}s)", flush=True)

    # ---- Merge + summary ----
    (OUT/"intrinsic_all.json").write_text(json.dumps(intr,indent=1))
    sh_vals=[r['split_half'] for r in intr if r.get('split_half') is not None]
    flagged_i=[r for r in intr if r.get('flag')]
    summ={"n_model_concepts":len(intr),"split_half_flagged":len(flagged_i),
        "split_half_pctiles":{p:float(np.percentile(sh_vals,p)) for p in [1,5,25,50]} if sh_vals else {},
        "cross_model_flagged":len(flagged),"elapsed_s":time.time()-t0}
    (OUT/"summary.json").write_text(json.dumps(summ,indent=2))
    print(f"\n=== SWEEP DONE ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"cross-model flagged: {len(flagged)} | split-half flagged (<{SPLIT_HALF_THR}): {len(flagged_i)}", flush=True)
    if sh_vals: print(f"split-half pctiles 1/5/25/50: {[round(np.percentile(sh_vals,q),3) for q in [1,5,25,50]]}", flush=True)
    for r in flagged_i[:25]: print(f"  {r['concept']:15s} {slugify(r['model']):32s} split_half={r['split_half']:.3f} sep={r['separation']:.3f}", flush=True)

if __name__=="__main__": main()
