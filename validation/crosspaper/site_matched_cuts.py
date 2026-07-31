#!/usr/bin/env python3
"""
site_matched_cuts.py — recompute every reported cut of the site-matched depth control.

Written: 2026-07-31 UTC — claude:prepub-antagonist
Answers P3's reproducibility request on tf264de0: pin exactly which models and which
filters produce each published cell count, so the numbers are re-derivable without
reading anyone's notes.

Source of record: HF james-ra-henry/Rosetta-Activations
                  paper_n250/_gem_depth_matched/<slug>_depth_matched_control.json
(NOT paper_n250/_p2_depth_matched_control_superseded_20260727/, which is the
2026-07-27 M6 tree — legacy arm only, no site_matched_control key.)

Usage:  python site_matched_cuts.py
Needs:  huggingface_hub, scipy.  No GPU, no model weights, ~30 s.
"""
from __future__ import annotations

import json
import statistics as st

from huggingface_hub import HfApi, hf_hub_download
from scipy.stats import binomtest, wilcoxon

REPO = "james-ra-henry/Rosetta-Activations"
TREE = "paper_n250/_gem_depth_matched/"

# P3's Table 1 base-model corpus. 28 models x 17 concepts = 476 cells, the population
# §6.5's 304/442 headline is drawn from.
BASE_28 = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    "google/gemma-2-2b", "google/gemma-2-9b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3", "microsoft/phi-2",
]
assert len(BASE_28) == 28

# The round-3 GPU sub-roster: `papers/shared/round3_gpu/common.py::BASE_28`, which despite
# its name holds 25 entries. Three models are dropped from Table 1's 28, each for a
# documented reason (see that file's header):
#   facebook/opt-350m  - only OPT variant with word_embed_proj_dim (512) != hidden_size
#                        (1024), so its embedding row is 512-dim while block rows are
#                        1024-dim, breaking the uniform-dimension assumption.
#   google/gemma-2-2b  - EXCLUDED 2026-07-16 (James): DOM directions do not stabilise
#   google/gemma-2-9b    across pair subsamples (split-half cos 0.52-0.60 vs >=0.96 for
#                        gpt2/Qwen controls; extraction itself bit-deterministic).
ROUND3_EXCLUSIONS = ["facebook/opt-350m", "google/gemma-2-2b", "google/gemma-2-9b"]
BASE_25 = [m for m in BASE_28 if m not in ROUND3_EXCLUSIONS]
assert len(BASE_25) == 25

# P2's roster: BASE_28 + Llama-3.1-8B. 29 x 17 = 493 cells.
P2_MODELS = BASE_28 + ["meta-llama/Llama-3.1-8B"]
assert len(P2_MODELS) == 29


def load() -> list[dict]:
    """Every cell in the site-matched tree, with its model id attached."""
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.startswith(TREE) and f.endswith("_depth_matched_control.json")]
    cells = []
    for f in sorted(files):
        doc = json.load(open(hf_hub_download(REPO, f, repo_type="dataset")))
        for cell in doc["results"]:
            cell["_model"] = doc["model_id"]
            cells.append(cell)
    return cells


def report(cells: list[dict], label: str, agg=st.median) -> None:
    """One cut. `agg` is the per-model aggregator fed to the model-level Wilcoxon."""
    if not cells:
        print(f"{label}: no cells")
        return
    sm = [c["site_matched_control"] for c in cells]
    better = sum(c["handoff_better"] for c in sm)
    deltas = [c["delta_pp"] for c in sm]
    models = sorted({c["_model"] for c in cells})
    per_model = [agg([c["site_matched_control"]["delta_pp"]
                      for c in cells if c["_model"] == m]) for m in models]
    w, p = wilcoxon(per_model)
    print(f"{label}")
    print(f"   {len(cells)} cells | handoff better {better} = {100 * better / len(cells):.1f}% "
          f"(sign test p={binomtest(better, len(cells), 0.5).pvalue:.3f})")
    print(f"   mean {st.mean(deltas):+.2f}pp | median {st.median(deltas):+.2f}pp")
    print(f"   model-level Wilcoxon on per-model {agg.__name__}: "
          f"N={len(models)}, W={w:.0f}, p={p:.3f}")


def main() -> None:
    cells = load()
    print(f"loaded {len(cells)} cells from {TREE}\n")

    matched = [c for c in cells if c["site_matched_control"].get("site_matched")]
    print(f"filter 1 - site_matched is True: {len(matched)}/493\n")

    # The multi-node filter. n_targets_total is the number of GEMs the handoff arm
    # ablates; >=2 is "multi-node", the population F8 is about. Single-GEM cells
    # (n_targets_total == 1) are the separate P2 crux class.
    multi = [c for c in matched if c["site_matched_control"]["n_targets_total"] >= 2]
    print(f"filter 2 - n_targets_total >= 2: {len(multi)}/{len(matched)}\n")

    report([c for c in multi if c["_model"] in BASE_28],
           "A. 28-model Table 1 roster - MATCHES §6.5's population (476 = 28x17)")
    print()
    report([c for c in multi if c["_model"] in BASE_28], "   (same cut, per-model MEAN)",
           agg=st.mean)
    print()
    report([c for c in multi if c["_model"] in BASE_25],
           "B. 25-model round-3 GPU sub-roster - matches §8.7's TNCONF control")
    print()
    report(multi, "C. 29-model P2 roster (P2_MODELS)")
    print()
    report([c for c in matched if c["site_matched_control"]["n_targets_total"] == 1],
           "D. single-GEM crux class (P2 §5.5; was n=2 before the coverage fix)")


if __name__ == "__main__":
    main()
