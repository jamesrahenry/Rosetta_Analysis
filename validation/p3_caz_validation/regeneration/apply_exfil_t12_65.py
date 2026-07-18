#!/usr/bin/env python3
"""Apply-pass: Table 12 cohort pooling + §6.5 handoff-vs-peak recompute,
defective vs corrected exfiltration.

Stages ablation_gem_*.json + patch_*.json for all 17 concepts x 28 models
into the local mirror (corrected exfiltration already in place from the
rerun; defective exfiltration staged from the frozen paper-n250 revision),
then pools:

  T12  - GEM handoff ablation sep-reduction by cohort (MHA n=306 -> 0.658
         published; GQA n=136 -> 0.626; gemma fisher-peak 0.386 in §8.3)
  T12  - patching recovery raw + trimmed by cohort (0.699/0.530 MHA n=323;
         0.770/0.691 GQA) — n=323 provenance is a known open item
         (tc4fd04e); we pool what the artifacts contain and report n.
  §6.5 - Phase 1: handoff-better count for pythia-1.4b + gpt2-xl x 17
         (published 33/34); full-corpus 26-model extension (297/442,
         MHA 207/289, GQA 74/119, gemma 16/34).

Validation contract: defective state must reproduce published values
before corrected values enter the ledger.
"""
import json
import sys
import numpy as np
from pathlib import Path

from huggingface_hub import hf_hub_download
import shutil

ROOT = Path.home() / "rosetta_data/paper_n250"
DEF = Path.home() / "rosetta_data/_defective_exfiltration"
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
MHA = [m for m in MODELS if m.startswith(("EleutherAI","openai","facebook","microsoft"))]
GQA = [m for m in MODELS if m.startswith(("Qwen","meta_llama","mistralai"))]
GEMMA = [m for m in MODELS if m.startswith("google")]
ADDITIONS_2026_07_03 = {"EleutherAI_pythia_12b", "Qwen_Qwen2.5_14B"}


def stage():
    """Fetch missing ablation_gem/patch jsons into mirror + defective staging."""
    fetched = 0
    for m in MODELS:
        for c in CONCEPTS:
            for kind in (f"ablation_gem_{c}.json", f"patch_{c}.json"):
                dst = ROOT / m / kind
                if not dst.exists():
                    p = hf_hub_download("james-ra-henry/Rosetta-Activations",
                                        f"paper_n250/{m}/{kind}", repo_type="dataset")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dst)
                    fetched += 1
        for kind in ("ablation_gem_exfiltration.json", "patch_exfiltration.json"):
            dst = DEF / m / kind
            if not dst.exists():
                p = hf_hub_download("james-ra-henry/Rosetta-Activations",
                                    f"paper_n250/{m}/{kind}", repo_type="dataset",
                                    revision="paper-n250")
                shutil.copyfile(p, dst)
                fetched += 1
    print(f"staged {fetched} new files", flush=True)


def base_for(state, m, c):
    return DEF / m if (state == "defective" and c == "exfiltration") else ROOT / m


def pool(state):
    print(f"\n===== {state}")
    # --- Table 12 ablation column: handoff-mode sep reduction (1 - retained/100)
    red = {"MHA": [], "GQA": [], "GEMMA": []}
    hb_pilot = []          # §6.5 Phase 1 (pythia-1.4b, gpt2-xl)
    hb_ext = {"MHA": [], "GQA": [], "GEMMA": []}   # §6.5 26-model extension
    for m in MODELS:
        cohort = "MHA" if m in MHA else ("GQA" if m in GQA else "GEMMA")
        for c in CONCEPTS:
            try:
                ag = json.load(open(base_for(state, m, c) / f"ablation_gem_{c}.json"))
            except FileNotFoundError:
                continue
            comp = ag.get("comparison") or {}
            hr = comp.get("handoff_retained_pct")
            if hr is not None:
                red[cohort].append(1 - hr / 100.0)
            hb = comp.get("handoff_better")
            if hb is not None:
                if m in ("EleutherAI_pythia_1.4b", "openai_community_gpt2_xl"):
                    hb_pilot.append(bool(hb))
                if m not in ADDITIONS_2026_07_03:
                    hb_ext[cohort].append(bool(hb))
    for k, v in red.items():
        print(f"T12 ablation {k}: mean sep-reduction {np.mean(v):.3f} (n={len(v)})")
    print(f"§6.5 Phase1 handoff-better: {sum(hb_pilot)}/{len(hb_pilot)}")
    tot_n = sum(len(v) for v in hb_ext.values()); tot_y = sum(sum(v) for v in hb_ext.values())
    print(f"§6.5 ext: total {tot_y}/{tot_n} ({100*tot_y/max(tot_n,1):.1f}%) | "
          + " | ".join(f"{k} {sum(v)}/{len(v)} ({100*sum(v)/max(len(v),1):.1f}%)" for k, v in hb_ext.items()))
    # --- Table 12 patching columns
    rec = {"MHA": [], "GQA": [], "GEMMA": []}
    for m in MODELS:
        cohort = "MHA" if m in MHA else ("GQA" if m in GQA else "GEMMA")
        for c in CONCEPTS:
            try:
                pj = json.load(open(base_for(state, m, c) / f"patch_{c}.json"))
            except FileNotFoundError:
                continue
            r = pj.get("recovery", pj.get("patch_recovery", pj.get("mean_recovery")))
            if r is None and "layers" in pj:
                continue  # schema fallback handled after first inspection
            if r is not None:
                rec[cohort].append(r)
    for k, v in rec.items():
        if v:
            arr = np.array(v)
            trim = arr[arr <= 1.0]
            print(f"T12 patching {k}: raw {arr.mean():.3f} (n={len(arr)}) | trimmed {trim.mean():.3f} "
                  f"({100*(arr>1.0).mean():.0f}% overshoot)")
        else:
            print(f"T12 patching {k}: no recovery field found — inspect patch schema")
    return red, rec


if __name__ == "__main__":
    stage()
    # schema peek on one patch json so the recovery field is confirmed
    pj = json.load(open(ROOT / "openai_community_gpt2" / "patch_causation.json"))
    print("patch json keys:", list(pj.keys()))
    for state in ("defective", "corrected"):
        pool(state)
