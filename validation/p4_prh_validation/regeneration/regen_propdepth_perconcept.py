#!/usr/bin/env python3
"""M4: recompute the §3.8 proportional-depth per-concept Delta on the CORRECTED
exfiltration data, reproducing p5_propdepth.py exactly (reads per-layer DOM
vectors from caz JSONs). Gives corrected exfiltration Delta (was pre-correction
+0.092) and confirms the pooled Delta stays ~+0.195. CPU, caz JSONs only.
"""
import json, time
from pathlib import Path
from itertools import permutations
from collections import defaultdict
import numpy as np
from scipy.linalg import orthogonal_procrustes
from huggingface_hub import hf_hub_download

HF="james-ra-henry/Rosetta-Activations"; REV="paper-n250"; TREE="paper_n250"
DEPTHS=[0.3,0.5,0.7]
CONCEPTS=["agency","authorization","causation","certainty","credibility","deception",
    "exfiltration","formality","moral_valence","negation","plurality","sarcasm",
    "sentiment","specificity","temporal_order","threat_severity","urgency"]
# handoff-primary A-E roster (30 models) + family map
MODELS=["openai_community_gpt2","EleutherAI_gpt_neo_125m","EleutherAI_pythia_160m","facebook_opt_125m",
    "openai_community_gpt2_medium","facebook_opt_350m","EleutherAI_pythia_410m","EleutherAI_pythia_1b",
    "EleutherAI_pythia_1.4b","facebook_opt_1.3b","meta_llama_Llama_3.2_1B","Qwen_Qwen2.5_3B",
    "EleutherAI_pythia_2.8b","facebook_opt_2.7b","microsoft_phi_2","EleutherAI_pythia_6.9b","facebook_opt_6.7b",
    "meta_llama_Llama_3.1_8B","mistralai_Mistral_7B_v0.3","Qwen_Qwen2.5_7B","google_gemma_2_9b",
    "EleutherAI_pythia_12b","Qwen_Qwen2.5_14B","Qwen_Qwen2.5_32B","Qwen_Qwen2.5_3B_Instruct",
    "Qwen_Qwen2.5_7B_Instruct","meta_llama_Llama_3.2_1B_Instruct","meta_llama_Llama_3.1_8B_Instruct",
    "mistralai_Mistral_7B_Instruct_v0.3","google_gemma_2_9b_it"]
FAMILY={"openai_community_gpt2":"GPT-2","openai_community_gpt2_medium":"GPT-2","EleutherAI_gpt_neo_125m":"GPT-Neo",
    "EleutherAI_pythia_160m":"Pythia","EleutherAI_pythia_410m":"Pythia","EleutherAI_pythia_1b":"Pythia",
    "EleutherAI_pythia_1.4b":"Pythia","EleutherAI_pythia_2.8b":"Pythia","EleutherAI_pythia_6.9b":"Pythia",
    "EleutherAI_pythia_12b":"Pythia","facebook_opt_125m":"OPT","facebook_opt_350m":"OPT","facebook_opt_1.3b":"OPT",
    "facebook_opt_2.7b":"OPT","facebook_opt_6.7b":"OPT","meta_llama_Llama_3.2_1B":"Llama 3",
    "meta_llama_Llama_3.2_1B_Instruct":"Llama 3","meta_llama_Llama_3.1_8B":"Llama 3","meta_llama_Llama_3.1_8B_Instruct":"Llama 3",
    "Qwen_Qwen2.5_3B":"Qwen 2.5","Qwen_Qwen2.5_3B_Instruct":"Qwen 2.5","Qwen_Qwen2.5_7B":"Qwen 2.5",
    "Qwen_Qwen2.5_7B_Instruct":"Qwen 2.5","Qwen_Qwen2.5_14B":"Qwen 2.5","Qwen_Qwen2.5_32B":"Qwen 2.5",
    "microsoft_phi_2":"Phi-2","mistralai_Mistral_7B_v0.3":"Mistral","mistralai_Mistral_7B_Instruct_v0.3":"Mistral",
    "google_gemma_2_9b":"Gemma 2","google_gemma_2_9b_it":"Gemma 2"}

def cosine(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>1e-10 and nb>1e-10 else float("nan")
def interp_rows(M,n):
    if len(M)==n: return M
    return M[np.round(np.linspace(0,len(M)-1,n)).astype(int)]
def depth_layer(n,f): return int(np.clip(round(f*(n-1)),0,n-1))

def load_dom(slug,c,cache):
    k=(slug,c)
    if k in cache: return cache[k]
    try:
        p=hf_hub_download(HF,f"{TREE}/{slug}/caz_{c}.json",repo_type="dataset",revision=REV)
        d=json.loads(Path(p).read_text())
        dv=np.array([m["dom_vector"] for m in d["layer_data"]["metrics"]],dtype=np.float64)
        nr=np.linalg.norm(dv,axis=1,keepdims=True); dv=dv/np.where(nr>1e-10,nr,1.0)
        cache[k]={"dim":d["hidden_dim"],"n_layers":d["layer_data"]["n_layers"],"dom":dv}
    except Exception as e:
        cache[k]=None
    return cache[k]

def main():
    import argparse
    ap=argparse.ArgumentParser(description="§3.8 proportional-depth per-concept Delta (corrected). "
        "--concepts subsets the HELD-OUT concept (basis always uses the other 16); "
        "e.g. --concepts exfiltration reproduces the corrected exfiltration Delta, "
        "--concepts moral_valence is the known-answer validation (paper +0.305).")
    ap.add_argument("--concepts", nargs="*", default=None)
    HELD=ap.parse_args().concepts or CONCEPTS
    t0=time.time(); cache={}
    store={m:{} for m in MODELS}
    for i,m in enumerate(MODELS):
        for c in CONCEPTS:
            d=load_dom(m,c,cache)
            if d is not None: store[m][c]=d
        print(f"[load {i+1}/30] {m} ({len(store[m])} concepts, {time.time()-t0:.0f}s)",flush=True)
    by_dim=defaultdict(list)
    for m,cm in store.items():
        if cm: by_dim[next(iter(cm.values()))["dim"]].append(m)
    rows=[]
    for dim,names in sorted(by_dim.items()):
        if len(names)<2: continue
        for a,b in permutations(names,2):
            if FAMILY[a]==FAMILY[b]: continue
            ca,cb=store[a],store[b]
            for held in HELD:
                if held not in ca or held not in cb: continue
                fit=[c for c in CONCEPTS if c!=held and c in ca and c in cb]
                if len(fit)<2: continue
                na,nb=ca[held]["n_layers"],cb[held]["n_layers"]; nc=min(na,nb)
                A=np.vstack([interp_rows(ca[c]["dom"],nc) for c in fit])
                B=np.vstack([interp_rows(cb[c]["dom"],nc) for c in fit])
                try: R,_=orthogonal_procrustes(B,A)
                except Exception: continue
                dvA,dvB=ca[held]["dom"],cb[held]["dom"]
                M=np.zeros((3,3))
                for ii,di in enumerate(DEPTHS):
                    vA=dvA[depth_layer(na,di)]
                    for jj,dj in enumerate(DEPTHS):
                        M[ii,jj]=cosine(vA,dvB[depth_layer(nb,dj)]@R)
                matched=np.array([M[k,k] for k in range(3)])
                mism=np.array([M[i,j] for i in range(3) for j in range(3) if i!=j])
                if np.any(np.isnan(matched)) or np.any(np.isnan(mism)): continue
                rows.append({"concept":held,"delta":float(matched.mean()-mism.mean())})
    # aggregate
    byc=defaultdict(list)
    for r in rows: byc[r["concept"]].append(r["delta"])
    alld=np.array([r["delta"] for r in rows])
    print(f"\n=== §3.8 proportional-depth, CORRECTED (n_trials={len(rows)}) ===",flush=True)
    print(f"POOLED Delta = {alld.mean():+.4f}  ({(alld>0).sum()}/{len(alld)} positive)  [paper: +0.195, 1,666/1,666]",flush=True)
    print("per-concept Delta:",flush=True)
    for c in sorted(byc,key=lambda c:np.mean(byc[c])):
        mark=" <-- EXFIL (was pre-correction +0.092)" if c=="exfiltration" else ""
        print(f"  {c:16s} {np.mean(byc[c]):+.4f} (n={len(byc[c])}){mark}",flush=True)
    out={"pooled_delta":float(alld.mean()),"n_trials":len(rows),"n_positive":int((alld>0).sum()),
         "per_concept":{c:float(np.mean(v)) for c,v in byc.items()},
         "exfiltration_corrected":float(np.mean(byc["exfiltration"])),"exfiltration_stale":0.092,
         "elapsed_s":time.time()-t0}
    o=Path("/home/jhenry/Games2/Eigan/Rosetta_Program/papers/prh-validation/scripts/m4_out"); o.mkdir(parents=True,exist_ok=True)
    (o/"propdepth_corrected.json").write_text(json.dumps(out,indent=2))
    print(f"wrote {o} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__": main()
