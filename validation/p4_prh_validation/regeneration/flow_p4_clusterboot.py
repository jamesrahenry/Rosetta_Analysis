"""Prefect flow: cluster-bootstrap reanalysis of P4's depth-stratification
result (Delta=+0.195, 1,666/1,666 positive), resampling at the MODEL level
(not the trial level) to get a properly adjusted CI."""
import json
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from prefect import flow, task


@task(retries=2)
def fetch_trial_data():
    local = Path.home() / "rosetta_data" / "paper_n250" / "_alignment" / "p5_propdepth_xfam_C17_AE.json"
    if local.exists():
        return str(local)
    return hf_hub_download(
        repo_id="james-ra-henry/Rosetta-Activations", repo_type="dataset",
        filename="paper_n250/_alignment/p5_propdepth_xfam_C17_AE.json",
        local_dir=str(Path.home() / "rosetta_data"),
    )


@task
def cluster_bootstrap_depth(path: str, n_boot: int = 5000, seed: int = 42):
    with open(path) as f:
        d = json.load(f)
    trials = d["pair_results"]
    naive_ci = d["summary"]["grand"]["bootstrap_ci_95"]
    naive_mean = d["summary"]["grand"]["mean_delta"]

    models = sorted(set(t["model_a"] for t in trials) | set(t["model_b"] for t in trials))
    rng = np.random.default_rng(seed)

    boot_means = []
    boot_positive_fracs = []
    for _ in range(n_boot):
        sampled = set(rng.choice(models, size=len(models), replace=True).tolist())
        subset = [t for t in trials if t["model_a"] in sampled and t["model_b"] in sampled]
        if not subset:
            continue
        deltas = np.array([t["obs_delta"] for t in subset])
        boot_means.append(deltas.mean())
        boot_positive_fracs.append(float((deltas > 0).mean()))

    boot_means = np.array(boot_means)
    boot_positive_fracs = np.array(boot_positive_fracs)

    # also: within-model-set trial count distribution, to show how much
    # effective sample size shrinks under model-level resampling
    subset_sizes = []
    for _ in range(500):
        sampled = set(rng.choice(models, size=len(models), replace=True).tolist())
        subset_sizes.append(sum(1 for t in trials if t["model_a"] in sampled and t["model_b"] in sampled))

    return {
        "n_trials_original": len(trials),
        "n_models": len(models),
        "naive_mean_delta": naive_mean,
        "naive_bootstrap_ci_95": naive_ci,
        "cluster_bootstrap_mean_delta": float(boot_means.mean()),
        "cluster_bootstrap_ci_95": [float(np.percentile(boot_means, 2.5)),
                                     float(np.percentile(boot_means, 97.5))],
        "cluster_bootstrap_std": float(boot_means.std()),
        "cluster_bootstrap_mean_frac_positive": float(boot_positive_fracs.mean()),
        "cluster_bootstrap_min_frac_positive": float(boot_positive_fracs.min()),
        "cluster_bootstrap_n_resamples_with_zero_subset": int((np.array([1 if s == 0 else 0 for s in subset_sizes])).sum()),
        "effective_n_trials_mean_under_resampling": float(np.mean(subset_sizes)),
        "n_bootstrap": n_boot,
    }


@flow(name="p4-cluster-bootstrap-depth")
def p4_cluster_bootstrap_depth():
    path = fetch_trial_data()
    result = cluster_bootstrap_depth(path)
    out_path = Path.home() / "rosetta_data" / "results" / "p4_cluster_bootstrap_depth.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p4_cluster_bootstrap_depth()
