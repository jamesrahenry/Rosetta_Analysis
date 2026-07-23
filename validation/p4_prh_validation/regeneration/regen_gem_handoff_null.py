#!/usr/bin/env python3
"""Correct-method permuted-CORRESPONDENCE handoff null, ALL 17 concepts, CPU,
reduced-rank Procrustes (validated == full d-dim SVD). This is what
handoff_permuted_null.py was written to produce but never ran (GPU destroyed).
Replaces the stale old-method prh_gem_handoff_null.csv (which sits at sd~0.023,
a random-direction baseline, instead of the intended permuted-correspondence
null at sd~0.16 that matches the peak-layer permuted null's 0.199).

Per-concept checkpoint: writes _handoff_null_out/<concept>.csv as each concept
finishes, so a crash/resume keeps completed concepts. Re-run to resume.
"""
import os, json, csv, time, sys
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

HF = "james-ra-henry/Rosetta-Activations"; REV = "paper-n250"; PAPER_TREE = "paper_n250"
STAGE = Path("/home/jhenry/Games2/rosetta_handoff_stage")
OUTDIR = Path("/home/jhenry/Games2/Eigan/Rosetta_Program/papers/prh-validation/scripts/_handoff_null_out")
N_TRIALS = 25; GLOBAL_SEED = 42

CONCEPTS = ["agency","authorization","causation","certainty","credibility","deception",
    "exfiltration","formality","moral_valence","negation","plurality","sarcasm",
    "sentiment","specificity","temporal_order","threat_severity","urgency"]

PRIMARY_MODELS = ["openai_community_gpt2","EleutherAI_gpt_neo_125m","EleutherAI_pythia_160m",
    "facebook_opt_125m","openai_community_gpt2_medium","facebook_opt_350m","EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b","EleutherAI_pythia_1.4b","facebook_opt_1.3b","meta_llama_Llama_3.2_1B",
    "Qwen_Qwen2.5_3B","EleutherAI_pythia_2.8b","facebook_opt_2.7b","microsoft_phi_2",
    "EleutherAI_pythia_6.9b","facebook_opt_6.7b","meta_llama_Llama_3.1_8B","mistralai_Mistral_7B_v0.3",
    "Qwen_Qwen2.5_7B","google_gemma_2_9b","EleutherAI_pythia_12b","Qwen_Qwen2.5_14B","Qwen_Qwen2.5_32B",
    "Qwen_Qwen2.5_3B_Instruct","Qwen_Qwen2.5_7B_Instruct","meta_llama_Llama_3.2_1B_Instruct",
    "meta_llama_Llama_3.1_8B_Instruct","mistralai_Mistral_7B_Instruct_v0.3","google_gemma_2_9b_it"]
FAMILY = {"openai_community_gpt2":"GPT-2","openai_community_gpt2_medium":"GPT-2","EleutherAI_gpt_neo_125m":"GPT-Neo",
    "EleutherAI_pythia_160m":"Pythia","EleutherAI_pythia_410m":"Pythia","EleutherAI_pythia_1b":"Pythia",
    "EleutherAI_pythia_1.4b":"Pythia","EleutherAI_pythia_2.8b":"Pythia","EleutherAI_pythia_6.9b":"Pythia",
    "EleutherAI_pythia_12b":"Pythia","facebook_opt_125m":"OPT","facebook_opt_350m":"OPT","facebook_opt_1.3b":"OPT",
    "facebook_opt_2.7b":"OPT","facebook_opt_6.7b":"OPT","meta_llama_Llama_3.2_1B":"Llama 3",
    "meta_llama_Llama_3.2_1B_Instruct":"Llama 3","meta_llama_Llama_3.1_8B":"Llama 3","meta_llama_Llama_3.1_8B_Instruct":"Llama 3",
    "Qwen_Qwen2.5_3B":"Qwen 2.5","Qwen_Qwen2.5_3B_Instruct":"Qwen 2.5","Qwen_Qwen2.5_7B":"Qwen 2.5",
    "Qwen_Qwen2.5_7B_Instruct":"Qwen 2.5","Qwen_Qwen2.5_14B":"Qwen 2.5","Qwen_Qwen2.5_32B":"Qwen 2.5",
    "microsoft_phi_2":"Phi-2","mistralai_Mistral_7B_v0.3":"Mistral","mistralai_Mistral_7B_Instruct_v0.3":"Mistral",
    "google_gemma_2_9b":"Gemma 2","google_gemma_2_9b_it":"Gemma 2"}
HF_ID = {"openai_community_gpt2":"openai-community/gpt2","openai_community_gpt2_medium":"openai-community/gpt2-medium",
    "EleutherAI_gpt_neo_125m":"EleutherAI/gpt-neo-125m","EleutherAI_pythia_160m":"EleutherAI/pythia-160m",
    "EleutherAI_pythia_410m":"EleutherAI/pythia-410m","EleutherAI_pythia_1b":"EleutherAI/pythia-1b",
    "EleutherAI_pythia_1.4b":"EleutherAI/pythia-1.4b","EleutherAI_pythia_2.8b":"EleutherAI/pythia-2.8b",
    "EleutherAI_pythia_6.9b":"EleutherAI/pythia-6.9b","EleutherAI_pythia_12b":"EleutherAI/pythia-12b",
    "facebook_opt_125m":"facebook/opt-125m","facebook_opt_350m":"facebook/opt-350m","facebook_opt_1.3b":"facebook/opt-1.3b",
    "facebook_opt_2.7b":"facebook/opt-2.7b","facebook_opt_6.7b":"facebook/opt-6.7b",
    "meta_llama_Llama_3.2_1B":"meta-llama/Llama-3.2-1B","meta_llama_Llama_3.2_1B_Instruct":"meta-llama/Llama-3.2-1B-Instruct",
    "meta_llama_Llama_3.1_8B":"meta-llama/Llama-3.1-8B","meta_llama_Llama_3.1_8B_Instruct":"meta-llama/Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5_3B":"Qwen/Qwen2.5-3B","Qwen_Qwen2.5_3B_Instruct":"Qwen/Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5_7B":"Qwen/Qwen2.5-7B","Qwen_Qwen2.5_7B_Instruct":"Qwen/Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5_14B":"Qwen/Qwen2.5-14B","Qwen_Qwen2.5_32B":"Qwen/Qwen2.5-32B","microsoft_phi_2":"microsoft/phi-2",
    "mistralai_Mistral_7B_v0.3":"mistralai/Mistral-7B-v0.3","mistralai_Mistral_7B_Instruct_v0.3":"mistralai/Mistral-7B-Instruct-v0.3",
    "google_gemma_2_9b":"google/gemma-2-9b","google_gemma_2_9b_it":"google/gemma-2-9b-it"}


def dom_at(sl):
    h = sl.shape[0]//2; d = sl[:h].mean(0)-sl[h:].mean(0); n = np.linalg.norm(d)
    return d/n if n>0 else d

def R_of(M):
    U,_,Vh = np.linalg.svd(M, full_matrices=False); return U@Vh

def cos(a,b):
    den = np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/den) if den>1e-12 else 0.0


def load_concept(concept):
    per = {}
    for m in PRIMARY_MODELS:
        npy = None
        try:
            gp = hf_hub_download(HF, f"{PAPER_TREE}/{m}/gem_{concept}.json", repo_type="dataset", revision=REV, local_dir=str(STAGE))
            gem = json.loads(Path(gp).read_text())
            npy = hf_hub_download(HF, f"{PAPER_TREE}/{m}/calibration_alllayer_{concept}.npy", repo_type="dataset", revision=REV, local_dir=str(STAGE))
            acts = np.load(npy, mmap_mode="r"); nl = acts.shape[0]
            h = min(max(gem["nodes"], key=lambda nd: nd["caz_end"])["handoff_layer"], nl-1)
            sl = np.asarray(acts[h], dtype=np.float64)
            per[m] = {"acts": sl, "dom": dom_at(sl), "dim": sl.shape[-1]}
            del acts
        except Exception as e:
            print(f"  [SKIP] {m} {concept}: {type(e).__name__} {str(e)[:50]}", flush=True)
        finally:
            if npy and os.path.exists(npy):
                try: os.remove(npy)
                except OSError: pass
    return per


def run_concept(concept, gpair0, gate):
    per = load_concept(concept)
    pairs = [(s,t) for s in PRIMARY_MODELS if s in per for t in PRIMARY_MODELS
             if t in per and s!=t and FAMILY[s]!=FAMILY[t] and per[s]["dim"]==per[t]["dim"]]
    rows = []; prims = []; gpair = gpair0; gate_err = 0.0
    for (s,t) in pairs:
        A,B = per[s],per[t]; n = min(A["acts"].shape[0], B["acts"].shape[0])
        sc = A["acts"][:n]-A["acts"][:n].mean(0); tc = B["acts"][:n]-B["acts"][:n].mean(0)
        ds,dt = A["dom"],B["dom"]
        Q,_ = np.linalg.qr(np.hstack([sc.T, tc.T]))
        sr = sc@Q; tr = tc@Q; dsr = ds@Q; dtr = dt@Q
        prims.append(cos(dsr, dtr@R_of(tr.T@sr)))
        rng = np.random.default_rng(GLOBAL_SEED + gpair)
        for tri in range(N_TRIALS):
            perm = rng.permutation(n)
            c = cos(dsr, dtr@R_of(tr[perm].T@sr))
            if gate and tri==0 and len(prims)==1:
                fc = cos(ds, dt@R_of(tc[np.random.default_rng(GLOBAL_SEED+gpair).permutation(n)].T@sc))
                gate_err = abs(fc-c)
            rows.append({"concept":concept,"source":HF_ID[s],"target":HF_ID[t],"trial":tri,"null_aligned_cosine":float(c)})
        gpair += 1
    return rows, prims, gpair, gate_err, len(pairs)


def main():
    STAGE.mkdir(parents=True, exist_ok=True); OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); gpair = 0
    for i,concept in enumerate(CONCEPTS):
        cpath = OUTDIR/f"{concept}.csv"
        if cpath.exists():
            done = list(csv.DictReader(open(cpath))); gpair += len(done)//N_TRIALS
            print(f"[{i+1}/17] {concept} cached ({len(done)//N_TRIALS} pairs) — skip", flush=True); continue
        tc0 = time.time()
        rows, prims, gpair, gerr, npairs = run_concept(concept, gpair, gate=(i==0))
        if i==0 and gerr>1e-8:
            raise SystemExit(f"GATE FAILED: reduced vs full {gerr:.2e}")
        with open(cpath,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["concept","source","target","trial","null_aligned_cosine"]); w.writeheader(); w.writerows(rows)
        v=np.array([r["null_aligned_cosine"] for r in rows])
        print(f"[{i+1}/17] {concept}: {npairs}p prim_mean={np.mean(prims):.4f} null_mean={v.mean():+.5f} sd={v.std():.4f} "
              f"{'gate='+format(gerr,'.1e')+' ' if i==0 else ''}({time.time()-tc0:.0f}s, tot {time.time()-t0:.0f}s)", flush=True)
    # merge
    allrows=[]
    for concept in CONCEPTS:
        allrows += list(csv.DictReader(open(OUTDIR/f"{concept}.csv")))
    merged = OUTDIR/"prh_gem_handoff_null_CORRECTED.csv"
    with open(merged,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["concept","source","target","trial","null_aligned_cosine"]); w.writeheader(); w.writerows(allrows)
    v=np.array([float(r["null_aligned_cosine"]) for r in allrows])
    summ={"n_rows":len(allrows),"grand_mean":float(v.mean()),"grand_sd":float(v.std()),"abs_max":float(np.abs(v).max()),
          "method":"permuted-correspondence orthogonal Procrustes, reduced-rank (validated==full SVD)","n_trials":N_TRIALS,"elapsed_s":time.time()-t0}
    (OUTDIR/"summary.json").write_text(json.dumps(summ,indent=2))
    print(f"\n=== MERGED: {len(allrows)} rows, grand mean {v.mean():+.5f} sd {v.std():.4f} abs_max {np.abs(v).max():.3f} ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"wrote {merged}", flush=True)


if __name__ == "__main__":
    main()
