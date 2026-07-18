"""Prefect flow: cluster-bootstrap reanalysis of P3's ablation-enrichment
statistic (3.68x, Mann-Whitney p=7.45e-135, Table 11), resampling at the
MODEL level instead of the layer-measurement level, to get a properly
adjusted CI that accounts for within-model layer autocorrelation and
shared-family non-independence."""
import json
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from prefect import flow, task

CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]
MODELS = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b",
    "openai_community_gpt2", "openai_community_gpt2_medium",
    "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_125m", "facebook_opt_350m", "facebook_opt_1.3b",
    "facebook_opt_2.7b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B", "Qwen_Qwen2.5_7B",
    "google_gemma_2_2b", "google_gemma_2_9b",
    "meta_llama_Llama_3.2_1B", "meta_llama_Llama_3.2_3B",
    "mistralai_Mistral_7B_v0.3", "microsoft_phi_2",
]
DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"


@task(retries=2)
def fetch_sweep_file(model: str, concept: str):
    local = DATA_ROOT / model / f"ablation_global_sweep_{concept}.json"
    if local.exists():
        return str(local)
    try:
        return hf_hub_download(
            repo_id="james-ra-henry/Rosetta-Activations", repo_type="dataset",
            filename=f"paper_n250/{model}/ablation_global_sweep_{concept}.json",
            local_dir=str(Path.home() / "rosetta_data"),
        )
    except Exception:
        return None


@task
def extract_measurements(path: str, model: str):
    """Return list of (is_caz_peak, global_sep_reduction) for this model/concept."""
    if path is None:
        return []
    with open(path) as f:
        d = json.load(f)
    peak = d.get("caz_peak")
    if peak is None:
        return []
    out = []
    for row in d["layers"]:
        l = row["layer"]
        red = row.get("global_sep_reduction")
        if red is None:
            continue
        is_peak = (l == peak)
        is_non_caz = abs(l - peak) > 3
        if is_peak or is_non_caz:
            out.append({"model": model, "is_peak": is_peak, "reduction": red})
    return out


@task
def cluster_bootstrap(records: list, n_boot: int = 2000, seed: int = 42):
    models = sorted(set(r["model"] for r in records))
    by_model = {m: [r for r in records if r["model"] == m] for m in models}
    rng = np.random.default_rng(seed)

    def ratio_for(recs):
        peak_vals = [r["reduction"] for r in recs if r["is_peak"]]
        non_vals = [r["reduction"] for r in recs if not r["is_peak"]]
        if not peak_vals or not non_vals or np.mean(non_vals) == 0:
            return None
        return np.mean(peak_vals) / np.mean(non_vals)

    observed = ratio_for(records)

    boot_ratios = []
    for _ in range(n_boot):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        boot_records = []
        for m in sampled_models:
            boot_records.extend(by_model[m])
        r = ratio_for(boot_records)
        if r is not None:
            boot_ratios.append(r)
    boot_ratios = np.array(boot_ratios)

    # model-level permutation test: shuffle peak/non-CAZ labels WITHIN each model,
    # recompute ratio, see how often permuted ratio >= observed
    perm_ratios = []
    for _ in range(2000):
        perm_records = []
        for m in models:
            recs = by_model[m]
            labels = [r["is_peak"] for r in recs]
            rng.shuffle(labels)
            for r, lab in zip(recs, labels):
                perm_records.append({"model": m, "is_peak": lab, "reduction": r["reduction"]})
        r = ratio_for(perm_records)
        if r is not None:
            perm_ratios.append(r)
    perm_ratios = np.array(perm_ratios)
    perm_p = float(np.mean(perm_ratios >= observed)) if len(perm_ratios) else None

    return {
        "n_models": len(models),
        "n_measurements": len(records),
        "observed_ratio": observed,
        "cluster_bootstrap_ci_95": [float(np.percentile(boot_ratios, 2.5)),
                                     float(np.percentile(boot_ratios, 97.5))],
        "cluster_bootstrap_mean": float(boot_ratios.mean()),
        "cluster_bootstrap_std": float(boot_ratios.std()),
        "within_model_permutation_p": perm_p,
        "n_bootstrap": n_boot,
    }


@flow(name="p3-cluster-bootstrap-ablation")
def p3_cluster_bootstrap_ablation():
    all_records = []
    for model in MODELS:
        for concept in CONCEPTS:
            path = fetch_sweep_file(model, concept)
            recs = extract_measurements(path, model)
            all_records.extend(recs)
    result = cluster_bootstrap(all_records)
    out_path = Path.home() / "rosetta_data" / "results" / "p3_cluster_bootstrap_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p3_cluster_bootstrap_ablation()
