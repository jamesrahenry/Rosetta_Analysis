#!/usr/bin/env python3
"""C5 leg 2 — split-half tau reliability ceiling (review item 6).

§3.2 rebuts the frequency confound with: "if frequency were the primary driver,
all 28 models would produce the same ordering (tau ~ 1.0); observed median tau
= 0.417 is far below that ceiling." The review calls this a strawman —
measurement noise attenuates tau below 1.0 no matter what, so 1.0 is not the
ceiling. The real ceiling is set by how reliably a single model's ordering is
measured at all, which the 250 calibration pairs let us estimate directly.

Recomputes each model's 17-concept ordering independently on each half of its
250 pairs (pipeline verified to reproduce the stored caz JSONs to 2e-16), then
runs two tests:

TEST 1 (what the review asked for) — classical attenuation ceiling.
  rel_125 = tau(orderingA, orderingB) per model; Spearman-Brown up to the full
  250 pairs; reliability of the 28-model grand mean by the k-item S-B formula;
  ceiling = sqrt(rel_model * rel_grandmean) = the largest tau we could expect to
  observe IF every model shared one true ordering. Compare to observed 0.417.
  Caveat: the attenuation formula is Pearson-derived, so it is an approximation
  on tau; reported for Spearman rho as well, where rank attenuation is better
  behaved.

TEST 2 (assumption-free) — within-model vs between-model agreement.
  Under "all models share one ordering + noise", model m's half A and model n's
  half B are two noisy reads of the SAME thing, so within-model agreement
  tau(A_m, B_m) should equal between-model agreement tau(A_m, B_n). If each
  model has its own reliable ordering, within > between. This needs no
  attenuation formula and no Spearman-Brown, and it is the cleaner answer.

Split: odd/even by pair index. The RCP jsonl files ARE blocked by topic (107
topics, 119 runs over 2666 records — generators are interleaved but topics are
contiguous), so a first/second-half split of the *file* would be topic-disjoint
rather than a noise split. It does not reach us that way: load_concept_pairs
draws via rng.sample(), which returns the 250 pairs in random selection order,
so the npy row order is already shuffled w.r.t. topic and any index split is a
pure measurement-noise split. A first/second variant was run on the first 57
(model, concept) cells and agreed with odd/even; it was then dropped as a
redundant second random split rather than the topic-generalization contrast it
would be on the raw file. (A topic-disjoint reliability split is a genuinely
different and interesting measurement — it needs a re-extraction against
topic-partitioned pairs, and is noted as follow-up, not done here.)

Streams one .npy at a time and deletes it (67.8 GB total, ~10 GB free), with a
per-(model, concept) checkpoint so it resumes. Compute-bound, not
download-bound (~5.6 s download vs ~31 s for 5 profile computations on a
24x2048 model), hence only the two half-orderings are computed here: the
full-sample ordering is read from the stored caz artifacts instead, which this
pipeline reproduces to 2.2e-16.

Written: 2026-07-16 UTC
"""
import argparse
import json
import shutil
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path.home() / "rosetta_tools"))
for _p in (str(Path.home()/"rosetta_tools"), str(Path.home()/"Source"/"Rosetta_Program"/"rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from huggingface_hub import hf_hub_download  # noqa: E402
from rosetta_tools.caz import compute_layer_metrics, find_caz_regions_scored  # noqa: E402

REPO = "james-ra-henry/Rosetta-Activations"
REVISION = "paper-n250"
SCRATCH = Path(__file__).parent / "_scratch_c5_ceiling"
CKPT = Path(__file__).parent / "results" / "c5_splithalf_depths.json"
OUT = Path(__file__).parent / "results" / "c5_splithalf_tau_ceiling.json"

MODELS_28 = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b", "EleutherAI_pythia_12b",
    "openai_community_gpt2", "openai_community_gpt2_medium",
    "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_125m", "facebook_opt_350m", "facebook_opt_1.3b",
    "facebook_opt_2.7b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B",
    "Qwen_Qwen2.5_7B", "Qwen_Qwen2.5_14B",
    "google_gemma_2_2b", "google_gemma_2_9b",
    "meta_llama_Llama_3.2_1B", "meta_llama_Llama_3.2_3B",
    "mistralai_Mistral_7B_v0.3", "microsoft_phi_2",
]
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]


def dominant_depth(pos, neg):
    """Full paper pipeline on one subsample: metrics -> scored detector ->
    dominant region depth_pct. Mirrors load_scored_region_df (default
    attention_paradigm; dominant is chosen on the paradigm-independent
    caz_score)."""
    metrics = compute_layer_metrics(
        [(pos[l].astype(np.float64), neg[l].astype(np.float64))
         for l in range(pos.shape[0])]
    )
    prof = find_caz_regions_scored(metrics, min_prominence_frac=0.005)
    return float(prof.dominant.depth_pct)


def process(slug, concept):
    """Download one calibration_alllayer npy, compute half-orderings, delete."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fn = f"calibration_alllayer_{concept}.npy"
    if concept == "exfiltration":
        # corrected N=249 rerun artifact (main revision), not the defective paper-n250 npy
        path = hf_hub_download(REPO, f"paper_n250/_round3_gpu/exfiltration_rerun/{slug}/{fn}",
                               repo_type="dataset", local_dir=str(SCRATCH))
    else:
        path = hf_hub_download(REPO, f"paper_n250/{slug}/{fn}", repo_type="dataset",
                               revision=REVISION, local_dir=str(SCRATCH))
    try:
        a = np.load(path)
        n = a.shape[1] // 2
        pos_all, neg_all = a[:, :n, :], a[:, n:, :]
        idx = np.arange(n)
        ia, ib = idx[0::2], idx[1::2]
        return {
            "n_pairs": int(n),
            "n_layers": int(a.shape[0]),
            "oddeven_A": dominant_depth(pos_all[:, ia, :], neg_all[:, ia, :]),
            "oddeven_B": dominant_depth(pos_all[:, ib, :], neg_all[:, ib, :]),
        }
    finally:
        Path(path).unlink(missing_ok=True)
        shutil.rmtree(SCRATCH / "paper_n250", ignore_errors=True)


def spearman_brown(r, k):
    """Reliability of a k-fold-longer test (k=2 for half -> full)."""
    if r <= -1 / (k - 1) if k > 1 else False:
        return np.nan
    return k * r / (1 + (k - 1) * r)


def collect(args):
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    todo = [(m, c) for m in MODELS_28 for c in CONCEPTS if f"{m}|{c}" not in ck]
    print(f"{len(ck)} cached, {len(todo)} to fetch")
    for i, (m, c) in enumerate(todo, 1):
        try:
            ck[f"{m}|{c}"] = process(m, c)
            print(f"[{i}/{len(todo)}] {m:32s} {c:16s} "
                  f"oddeven A={ck[f'{m}|{c}']['oddeven_A']:5.1f} "
                  f"B={ck[f'{m}|{c}']['oddeven_B']:5.1f}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAIL {m} {c}: {type(e).__name__}: {e}", flush=True)
            ck[f"{m}|{c}"] = {"error": f"{type(e).__name__}: {e}"}
        if i % 10 == 0 or i == len(todo):
            CKPT.write_text(json.dumps(ck, indent=2))
    CKPT.write_text(json.dumps(ck, indent=2))
    return ck


def artifact_pivot():
    """Full-sample 28x17 dominant-depth pivot straight from the stored caz
    artifacts — the published statistic's own path (reproduces median tau =
    0.417). Recomputing it from the npys is redundant: this pipeline matches
    the artifacts to 2.2e-16 on the full sample."""
    from rosetta_tools.reporting import load_results_dir, load_scored_region_df
    root = Path.home() / "rosetta_data" / "paper_n250"
    layer_df = load_results_dir([root / m for m in MODELS_28])
    layer_df = layer_df[layer_df["concept"].isin(CONCEPTS)]
    region_df = load_scored_region_df(layer_df, min_prominence_frac=0.005)
    dom = region_df[region_df["is_dominant"]]
    piv = dom.pivot_table(index="model_id", columns="concept", values="depth_pct")
    piv = piv.reindex(columns=CONCEPTS)
    # load_results_dir keys by HF id; map back to slug
    piv.index = [i.replace("/", "_").replace("-", "_") for i in piv.index]
    return piv


def analyse(ck):
    def order(slug, key):
        v = []
        for c in CONCEPTS:
            d = ck.get(f"{slug}|{c}")
            if not d or "error" in d or key not in d:
                return None
            v.append(d[key])
        return np.array(v)

    piv = artifact_pivot()
    res = {}
    for scheme in ("oddeven",):
        A = {m: order(m, f"{scheme}_A") for m in MODELS_28}
        B = {m: order(m, f"{scheme}_B") for m in MODELS_28}
        ok = [m for m in MODELS_28 if A[m] is not None and B[m] is not None
              and m in piv.index]

        # --- TEST 1: reliability + attenuation ceiling ---
        rel_t = {m: kendalltau(A[m], B[m]).correlation for m in ok}
        rel_r = {m: spearmanr(A[m], B[m]).correlation for m in ok}
        full = {m: piv.loc[m].values.astype(float) for m in ok}
        grand = np.mean([full[m] for m in ok], axis=0)
        obs_t = {m: kendalltau(full[m], grand).correlation for m in ok}

        med_rel_t = float(np.median(list(rel_t.values())))
        med_rel_r = float(np.median(list(rel_r.values())))
        rel250_t = spearman_brown(med_rel_t, 2)
        rel250_r = spearman_brown(med_rel_r, 2)
        relD_t = spearman_brown(rel250_t, len(ok))
        relD_r = spearman_brown(rel250_r, len(ok))
        ceil_t = float(np.sqrt(max(rel250_t, 0) * max(relD_t, 0)))
        ceil_r = float(np.sqrt(max(rel250_r, 0) * max(relD_r, 0)))

        # --- TEST 2: within vs between ---
        within = [kendalltau(A[m], B[m]).correlation for m in ok]
        between = [kendalltau(A[m], B[n]).correlation for m, n in permutations(ok, 2)]
        res[scheme] = {
            "n_models": len(ok),
            "median_splithalf_tau_125": med_rel_t,
            "median_splithalf_rho_125": med_rel_r,
            "spearman_brown_rel_250_tau": float(rel250_t),
            "spearman_brown_rel_250_rho": float(rel250_r),
            "rel_grandmean_tau": float(relD_t),
            "rel_grandmean_rho": float(relD_r),
            "attenuation_ceiling_tau": ceil_t,
            "attenuation_ceiling_rho": ceil_r,
            "observed_median_tau_vs_grandmean": float(np.median(list(obs_t.values()))),
            "within_model_mean_tau": float(np.mean(within)),
            "within_model_median_tau": float(np.median(within)),
            "between_model_mean_tau": float(np.mean(between)),
            "between_model_median_tau": float(np.median(between)),
            "within_minus_between": float(np.mean(within) - np.mean(between)),
            "per_model_splithalf_tau": {m: float(rel_t[m]) for m in ok},
            "per_model_observed_tau": {m: float(obs_t[m]) for m in ok},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    ck = json.loads(CKPT.read_text()) if args.analyse_only else collect(args)
    res = analyse(ck)

    for scheme, r in res.items():
        print(f"\n=== {scheme} ({r['n_models']} models) ===")
        print(f"  split-half tau (125 pairs)      : {r['median_splithalf_tau_125']:.3f}")
        print(f"  Spearman-Brown rel @250         : {r['spearman_brown_rel_250_tau']:.3f}")
        print(f"  rel of 28-model grand mean      : {r['rel_grandmean_tau']:.3f}")
        print(f"  ATTENUATION CEILING (tau)       : {r['attenuation_ceiling_tau']:.3f}")
        print(f"  ATTENUATION CEILING (rho)       : {r['attenuation_ceiling_rho']:.3f}")
        print(f"  observed median tau             : {r['observed_median_tau_vs_grandmean']:.3f}")
        print(f"  --- within vs between ---")
        print(f"  within-model  mean tau          : {r['within_model_mean_tau']:.3f}")
        print(f"  between-model mean tau          : {r['between_model_mean_tau']:.3f}")
        print(f"  within - between                : {r['within_minus_between']:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "written_utc": __import__("subprocess").run(
            ["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True
        ).stdout.strip(),
        "job": "C5 leg 2 — split-half tau reliability ceiling (review item 6)",
        "paper_baseline": {"median_tau": 0.417, "rebuttal_ceiling_claimed": 1.0},
        "results": res,
        "method_notes": (
            "Per model x concept, the 250-pair calibration_alllayer npy (paper-n250 "
            "revision) is split and each half run through the full paper pipeline: "
            "compute_layer_metrics -> find_caz_regions_scored(min_prominence_frac=0.005) "
            "-> dominant region depth_pct, matching load_scored_region_df. Pipeline "
            "verified against the stored caz JSONs at 2.2e-16 on the full sample. "
            "TEST 1 ceiling = sqrt(rel_model * rel_grandmean) with Spearman-Brown "
            "corrections; the attenuation formula is Pearson-derived so the tau version "
            "is an approximation (rho reported alongside). TEST 2 (within vs between) "
            "needs no such assumption: under a shared-ordering null, tau(A_m,B_m) and "
            "tau(A_m,B_n) estimate the same quantity."
        ),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
