#!/usr/bin/env python3
"""Exfiltration-only recompute of the GEM-handoff primary + permuted-label null
on the CORRECTED calibration, CPU, reduced-rank Procrustes (validated == full
d-dim SVD to <1e-8 per run). Replaces the stale exfiltration rows in
prh_gem_handoff_null.csv and gives the corrected exfiltration handoff mean.

Reduced-rank identity: DOM vectors and (mean-centred) calibration rows both live
in the <=2n-dim row space B of hstack([src_c.T, tgt_c.T]); the full d-dim
orthogonal Procrustes R acts identically to the r-dim reduced R on any vector in
B, so cos(dom_src, dom_tgt @ R) is unchanged. Permuting tgt rows (the null) does
not change B, so one projection serves all 25 trials. Gate below proves it.
"""
import os, json, csv, time
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

HF = "james-ra-henry/Rosetta-Activations"
REV = "paper-n250"           # both revs carry corrected exfil (498=249x2 samples)
PAPER_TREE = "paper_n250"
STAGE = Path("/home/jhenry/Games2/rosetta_handoff_stage")
CONCEPT = "exfiltration"
N_TRIALS = 25
SEED = 42
STALE_PRIMARY = 0.9022119226593048   # results_corrected.json exfil handoff_mean (pre-correction)

PRIMARY_MODELS = [
    "openai_community_gpt2", "EleutherAI_gpt_neo_125m", "EleutherAI_pythia_160m",
    "facebook_opt_125m", "openai_community_gpt2_medium", "facebook_opt_350m",
    "EleutherAI_pythia_410m", "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b",
    "facebook_opt_1.3b", "meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B",
    "EleutherAI_pythia_2.8b", "facebook_opt_2.7b", "microsoft_phi_2",
    "EleutherAI_pythia_6.9b", "facebook_opt_6.7b", "meta_llama_Llama_3.1_8B",
    "mistralai_Mistral_7B_v0.3", "Qwen_Qwen2.5_7B", "google_gemma_2_9b",
    "EleutherAI_pythia_12b", "Qwen_Qwen2.5_14B", "Qwen_Qwen2.5_32B",
    "Qwen_Qwen2.5_3B_Instruct", "Qwen_Qwen2.5_7B_Instruct",
    "meta_llama_Llama_3.2_1B_Instruct", "meta_llama_Llama_3.1_8B_Instruct",
    "mistralai_Mistral_7B_Instruct_v0.3", "google_gemma_2_9b_it",
]
FAMILY = {
    "openai_community_gpt2": "GPT-2", "openai_community_gpt2_medium": "GPT-2",
    "EleutherAI_gpt_neo_125m": "GPT-Neo", "EleutherAI_pythia_160m": "Pythia",
    "EleutherAI_pythia_410m": "Pythia", "EleutherAI_pythia_1b": "Pythia",
    "EleutherAI_pythia_1.4b": "Pythia", "EleutherAI_pythia_2.8b": "Pythia",
    "EleutherAI_pythia_6.9b": "Pythia", "EleutherAI_pythia_12b": "Pythia",
    "facebook_opt_125m": "OPT", "facebook_opt_350m": "OPT", "facebook_opt_1.3b": "OPT",
    "facebook_opt_2.7b": "OPT", "facebook_opt_6.7b": "OPT",
    "meta_llama_Llama_3.2_1B": "Llama 3", "meta_llama_Llama_3.2_1B_Instruct": "Llama 3",
    "meta_llama_Llama_3.1_8B": "Llama 3", "meta_llama_Llama_3.1_8B_Instruct": "Llama 3",
    "Qwen_Qwen2.5_3B": "Qwen 2.5", "Qwen_Qwen2.5_3B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_7B": "Qwen 2.5", "Qwen_Qwen2.5_7B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_14B": "Qwen 2.5", "Qwen_Qwen2.5_32B": "Qwen 2.5",
    "microsoft_phi_2": "Phi-2", "mistralai_Mistral_7B_v0.3": "Mistral",
    "mistralai_Mistral_7B_Instruct_v0.3": "Mistral",
    "google_gemma_2_9b": "Gemma 2", "google_gemma_2_9b_it": "Gemma 2",
}
HF_ID = {
    "openai_community_gpt2": "openai-community/gpt2",
    "openai_community_gpt2_medium": "openai-community/gpt2-medium",
    "EleutherAI_gpt_neo_125m": "EleutherAI/gpt-neo-125m",
    "EleutherAI_pythia_160m": "EleutherAI/pythia-160m",
    "EleutherAI_pythia_410m": "EleutherAI/pythia-410m",
    "EleutherAI_pythia_1b": "EleutherAI/pythia-1b",
    "EleutherAI_pythia_1.4b": "EleutherAI/pythia-1.4b",
    "EleutherAI_pythia_2.8b": "EleutherAI/pythia-2.8b",
    "EleutherAI_pythia_6.9b": "EleutherAI/pythia-6.9b",
    "EleutherAI_pythia_12b": "EleutherAI/pythia-12b",
    "facebook_opt_125m": "facebook/opt-125m", "facebook_opt_350m": "facebook/opt-350m",
    "facebook_opt_1.3b": "facebook/opt-1.3b", "facebook_opt_2.7b": "facebook/opt-2.7b",
    "facebook_opt_6.7b": "facebook/opt-6.7b",
    "meta_llama_Llama_3.2_1B": "meta-llama/Llama-3.2-1B",
    "meta_llama_Llama_3.2_1B_Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta_llama_Llama_3.1_8B": "meta-llama/Llama-3.1-8B",
    "meta_llama_Llama_3.1_8B_Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen_Qwen2.5_3B": "Qwen/Qwen2.5-3B", "Qwen_Qwen2.5_3B_Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5_7B": "Qwen/Qwen2.5-7B", "Qwen_Qwen2.5_7B_Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5_14B": "Qwen/Qwen2.5-14B", "Qwen_Qwen2.5_32B": "Qwen/Qwen2.5-32B",
    "microsoft_phi_2": "microsoft/phi-2",
    "mistralai_Mistral_7B_v0.3": "mistralai/Mistral-7B-v0.3",
    "mistralai_Mistral_7B_Instruct_v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "google_gemma_2_9b": "google/gemma-2-9b", "google_gemma_2_9b_it": "google/gemma-2-9b-it",
}


def dom_at(sl):
    half = sl.shape[0] // 2
    d = sl[:half].mean(0) - sl[half:].mean(0)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else d


def procrustes_R(M):
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh


def cos(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def full_aligned(src_c, tgt_c, dom_s, dom_t, perm=None):
    """Reference: full d-dim orthogonal Procrustes."""
    t = tgt_c if perm is None else tgt_c[perm]
    R = procrustes_R(t.T @ src_c)
    return cos(dom_s, dom_t @ R)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    # ---- Stage load: corrected handoff-layer acts + DOM per model ----
    per_model = {}
    for m in PRIMARY_MODELS:
        try:
            gp = hf_hub_download(HF, f"{PAPER_TREE}/{m}/gem_{CONCEPT}.json",
                                 repo_type="dataset", revision=REV, local_dir=str(STAGE))
            gem = json.loads(Path(gp).read_text())
            npy = hf_hub_download(HF, f"{PAPER_TREE}/{m}/calibration_alllayer_{CONCEPT}.npy",
                                  repo_type="dataset", revision=REV, local_dir=str(STAGE))
            acts = np.load(npy, mmap_mode="r")
            n_layers = acts.shape[0]
            deepest = max(gem["nodes"], key=lambda nd: nd["caz_end"])
            h = min(deepest["handoff_layer"], n_layers - 1)
            sl = np.asarray(acts[h], dtype=np.float64)   # [n, d]
            per_model[m] = {"acts_h": sl, "dom_h": dom_at(sl),
                            "handoff": h, "dim": sl.shape[-1], "n": sl.shape[0]}
            del acts
            os.remove(npy)
            print(f"[load] {m:38s} h={h:3d} dim={sl.shape[-1]:5d} n={sl.shape[0]}", flush=True)
        except Exception as e:
            print(f"[SKIP] {m}: {type(e).__name__}: {str(e)[:70]}", flush=True)
    print(f"[load] {len(per_model)} models loaded in {time.time()-t_start:.0f}s", flush=True)

    # ---- Pairs: cross-family, same-dim ----
    pairs = []
    for s in PRIMARY_MODELS:
        if s not in per_model:
            continue
        for t in PRIMARY_MODELS:
            if s == t or t not in per_model or FAMILY[s] == FAMILY[t]:
                continue
            if per_model[s]["dim"] != per_model[t]["dim"]:
                continue
            pairs.append((s, t))
    print(f"[pairs] {len(pairs)} cross-family same-dim pairs", flush=True)

    # ---- Compute (reduced) + validation gate (full) on first 3 pairs ----
    rows, primaries = [], []
    max_gate_err = 0.0
    for idx, (s, t) in enumerate(pairs):
        A, B = per_model[s], per_model[t]
        src = A["acts_h"]; tgt = B["acts_h"]
        n = min(src.shape[0], tgt.shape[0])
        src_c = src[:n] - src[:n].mean(0)
        tgt_c = tgt[:n] - tgt[:n].mean(0)
        dom_s = A["dom_h"]; dom_t = B["dom_h"]
        # reduced basis Q [d, r]
        Q, _ = np.linalg.qr(np.hstack([src_c.T, tgt_c.T]))    # [d, 2n]
        src_r = src_c @ Q; tgt_r = tgt_c @ Q
        dom_s_r = dom_s @ Q; dom_t_r = dom_t @ Q
        # primary (reduced)
        Rr = procrustes_R(tgt_r.T @ src_r)
        prim = cos(dom_s_r, dom_t_r @ Rr)
        primaries.append(prim)
        # null trials (reduced)
        rng = np.random.default_rng(SEED + idx)
        trial_cos = []
        for tr in range(N_TRIALS):
            perm = rng.permutation(n)
            Rr = procrustes_R(tgt_r[perm].T @ src_r)
            trial_cos.append(cos(dom_s_r, dom_t_r @ Rr))
        # validation gate: full d-dim SVD on first 3 pairs (primary + trial 0)
        if idx < 3:
            f_prim = full_aligned(src_c, tgt_c, dom_s, dom_t)
            rng2 = np.random.default_rng(SEED + idx)
            perm0 = rng2.permutation(n)
            f_null0 = full_aligned(src_c, tgt_c, dom_s, dom_t, perm0)
            e = max(abs(f_prim - prim), abs(f_null0 - trial_cos[0]))
            max_gate_err = max(max_gate_err, e)
            print(f"[GATE] pair {idx} {s[:20]}/{t[:20]} d={A['dim']} "
                  f"reduced_prim={prim:.10f} full_prim={f_prim:.10f} "
                  f"null0 red={trial_cos[0]:.2e} full={f_null0:.2e} err={e:.2e}", flush=True)
        for tr, c in enumerate(trial_cos):
            rows.append({"concept": CONCEPT, "source": HF_ID[s], "target": HF_ID[t],
                         "trial": tr, "null_aligned_cosine": float(c)})
        if idx % 20 == 0:
            print(f"[compute] {idx+1}/{len(pairs)} prim_running_mean={np.mean(primaries):.4f}", flush=True)

    if max_gate_err > 1e-8:
        raise SystemExit(f"VALIDATION FAILED: reduced vs full err {max_gate_err:.2e} > 1e-8")

    prim_mean = float(np.mean(primaries))
    null = np.array([r["null_aligned_cosine"] for r in rows])
    out = {
        "concept": CONCEPT, "n_pairs": len(pairs), "n_null_rows": len(rows),
        "handoff_primary_mean_CORRECTED": prim_mean,
        "handoff_primary_mean_STALE": STALE_PRIMARY,
        "delta_vs_stale": prim_mean - STALE_PRIMARY,
        "per_pair_primary": primaries,
        "null_grand_mean": float(null.mean()), "null_grand_sd": float(null.std()),
        "null_abs_max": float(np.abs(null).max()),
        "validation_reduced_vs_full_err": max_gate_err,
        "method": "reduced-rank orthogonal Procrustes (validated==full d-dim SVD)",
        "revision": REV, "seed": SEED, "n_trials": N_TRIALS,
        "elapsed_s": time.time() - t_start,
    }
    outdir = Path("/home/jhenry/Games2/Eigan/Rosetta_Program/papers/prh-validation/scripts/_handoff_exfil_out")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exfil_handoff_recompute_summary.json").write_text(json.dumps(out, indent=2))
    with open(outdir / "exfil_handoff_null_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "source", "target", "trial", "null_aligned_cosine"])
        w.writeheader(); w.writerows(rows)
    print("\n=== RESULT ===", flush=True)
    print(f"  gate err (reduced vs full): {max_gate_err:.2e}  (< 1e-8 OK)", flush=True)
    print(f"  handoff primary CORRECTED : {prim_mean:.5f}   (stale was {STALE_PRIMARY:.5f}, "
          f"delta {prim_mean-STALE_PRIMARY:+.5f})", flush=True)
    print(f"  null grand mean/sd        : {null.mean():+.6f} / {null.std():.4f}  abs_max {np.abs(null).max():.4f}", flush=True)
    print(f"  wrote {outdir}", flush=True)
    print(f"  total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
