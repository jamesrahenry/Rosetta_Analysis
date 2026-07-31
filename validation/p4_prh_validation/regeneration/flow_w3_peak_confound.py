"""Prefect flow: tests whether Fisher-peak layer selection is aligning well
merely because high-separation layers are geometrically "easy" to align
under any similarity metric, independent of true cross-architecture
correspondence (P4 round-2 review, W3).

Design: for each of the 7 cross-family same-dim pairs (reusing the
leave-one-generator-out pilot's pair set), for each concept, find the
Fisher-peak layer AND a "matched non-peak" layer whose separation score is
closest to the peak's score (excluding the peak +/-2 layers). Compute
Procrustes-aligned cosine at both. If peak alignment is not meaningfully
higher than score-matched non-peak alignment, that supports the "easy
geometry" confound; if peak alignment is clearly higher, that's evidence
against it (genuine correspondence, not just separation-driven ease)."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes
from huggingface_hub import hf_hub_download
from prefect import flow, task

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"
PAIRS = [
    ("EleutherAI_pythia_1.4b", "facebook_opt_1.3b"),
    ("EleutherAI_pythia_1.4b", "meta_llama_Llama_3.2_1B"),
    ("EleutherAI_pythia_1.4b", "Qwen_Qwen2.5_3B"),
    ("facebook_opt_1.3b", "meta_llama_Llama_3.2_1B"),
    ("facebook_opt_1.3b", "Qwen_Qwen2.5_3B"),
    ("meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B"),
    ("EleutherAI_pythia_2.8b", "facebook_opt_2.7b"),
]
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]


def fisher_separation(pos, neg):
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    dist = np.linalg.norm(mu_pos - mu_neg)
    var_pos = pos.var(0, ddof=1).sum() if len(pos) > 1 else 0.0
    var_neg = neg.var(0, ddof=1).sum() if len(neg) > 1 else 0.0
    denom = np.sqrt(0.5 * (var_pos + var_neg))
    return dist / denom if denom > 0 else 0.0


@task(retries=2)
def ensure_local(model: str, concept: str):
    meta = DATA_ROOT / model / f"calibration_{concept}_meta.json"
    arr = DATA_ROOT / model / f"calibration_alllayer_{concept}.npy"
    if meta.exists() and arr.exists():
        return str(meta), str(arr)
    hf_hub_download(repo_id="james-ra-henry/Rosetta-Activations", repo_type="dataset",
                     filename=f"paper_n250/{model}/calibration_{concept}_meta.json",
                     local_dir=str(Path.home() / "rosetta_data"))
    hf_hub_download(repo_id="james-ra-henry/Rosetta-Activations", repo_type="dataset",
                     filename=f"paper_n250/{model}/calibration_alllayer_{concept}.npy",
                     local_dir=str(Path.home() / "rosetta_data"))
    return str(meta), str(arr)


def load_all_layer_scores(arr_path: str, n_pairs_hint=250):
    arr = np.load(arr_path)  # (n_layers, 2*n_pairs, hidden_dim)
    n_layers = arr.shape[0]
    n_pairs = arr.shape[1] // 2
    scores = np.empty(n_layers)
    for l in range(n_layers):
        pos = arr[l, :n_pairs, :]
        neg = arr[l, n_pairs:, :]
        scores[l] = fisher_separation(pos, neg)
    return arr, scores, n_pairs


def dom_and_calib(arr, n_pairs, layer):
    pos = arr[layer, :n_pairs, :]
    neg = arr[layer, n_pairs:, :]
    dom = pos.mean(0) - neg.mean(0)
    dom = dom / (np.linalg.norm(dom) + 1e-12)
    calib = np.vstack([pos, neg])
    return dom, calib


def aligned_cosine(dom_s, calib_s, dom_t, calib_t):
    n = min(len(calib_s), len(calib_t))
    R, _ = orthogonal_procrustes(calib_t[:n], calib_s[:n])
    dom_t_rot = dom_t @ R
    dom_t_rot = dom_t_rot / (np.linalg.norm(dom_t_rot) + 1e-12)
    return float(np.dot(dom_s, dom_t_rot))


@task
def process_pair_concept(model_a: str, model_b: str, concept: str):
    _, arr_a_path = ensure_local(model_a, concept)
    _, arr_b_path = ensure_local(model_b, concept)
    arr_a, scores_a, np_a = load_all_layer_scores(arr_a_path)
    arr_b, scores_b, np_b = load_all_layer_scores(arr_b_path)

    peak_a, peak_b = int(np.argmax(scores_a)), int(np.argmax(scores_b))

    def matched_nonpeak(scores, peak, exclude_radius=2):
        candidates = [l for l in range(len(scores)) if abs(l - peak) > exclude_radius]
        if not candidates:
            return None
        target = scores[peak]
        best = min(candidates, key=lambda l: abs(scores[l] - target))
        return best

    match_a = matched_nonpeak(scores_a, peak_a)
    match_b = matched_nonpeak(scores_b, peak_b)
    if match_a is None or match_b is None:
        return None

    dom_a_peak, calib_a_peak = dom_and_calib(arr_a, np_a, peak_a)
    dom_b_peak, calib_b_peak = dom_and_calib(arr_b, np_b, peak_b)
    peak_cos = aligned_cosine(dom_a_peak, calib_a_peak, dom_b_peak, calib_b_peak)

    dom_a_m, calib_a_m = dom_and_calib(arr_a, np_a, match_a)
    dom_b_m, calib_b_m = dom_and_calib(arr_b, np_b, match_b)
    matched_cos = aligned_cosine(dom_a_m, calib_a_m, dom_b_m, calib_b_m)

    return {
        "model_a": model_a, "model_b": model_b, "concept": concept,
        "peak_a": peak_a, "peak_b": peak_b,
        "score_peak_a": float(scores_a[peak_a]), "score_peak_b": float(scores_b[peak_b]),
        "match_a": match_a, "match_b": match_b,
        "score_match_a": float(scores_a[match_a]), "score_match_b": float(scores_b[match_b]),
        "peak_aligned_cosine": peak_cos,
        "matched_nonpeak_aligned_cosine": matched_cos,
        "peak_minus_matched": peak_cos - matched_cos,
    }


@flow(name="w3-peak-selection-confound")
def w3_peak_selection_confound():
    results = []
    for model_a, model_b in PAIRS:
        for concept in CONCEPTS:
            r = process_pair_concept(model_a, model_b, concept)
            if r is not None:
                results.append(r)

    diffs = np.array([r["peak_minus_matched"] for r in results])
    summary = {
        "n_comparisons": len(results),
        "mean_peak_minus_matched": float(diffs.mean()),
        "std_peak_minus_matched": float(diffs.std()),
        "median_peak_minus_matched": float(np.median(diffs)),
        "frac_peak_better": float((diffs > 0).mean()),
        "frac_within_0.02": float((np.abs(diffs) < 0.02).mean()),
        "min_diff": float(diffs.min()),
        "max_diff": float(diffs.max()),
    }
    out = {"summary": summary, "detail": results}
    out_path = Path.home() / "rosetta_data" / "results" / "w3_peak_selection_confound.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    w3_peak_selection_confound()
