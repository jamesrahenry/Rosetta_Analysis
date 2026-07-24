"""Prefect flow — re-freeze P2 §5.4 permutation-ablation results on the current
gem node vintage (rosetta_tools 1.3.1).

Companion to gem/regenerate_table3_handoff_depth.py (which re-froze §4.3 Table 3
and the §5.4 node-count on the current store). Table 3 and the node counts are
CPU-derivable and already done; the §5.4 permutation-ablation *results*
(deepest-node-dominates, cross-disruption, synergy) require model forward
passes and were left on the pre-1.3.1 node vintage. This flow re-runs them.

Closes Hopper t7f9ce62.

What it does
------------
1. VINTAGE GUARD — verify the gem store on the run node is the current 1.3.1
   vintage, not stale gem (canary: pythia-6.9b's 3-node concepts must be
   {negation, sentiment}, NOT the old {causation, exfiltration, negation}).
   Aborts before spending GPU time if the store is stale.
2. Run gem/ablate_gem_permutation.py on the three §5.4 models, sequentially
   (one model on the GPU at a time).
3. Aggregate the deepest-node-dominates / cross-disruption / synergy statistics
   over each model's multi-node concepts.
4. Diff against the values currently in the preprint and emit ready-to-paste
   §5.4 sentences + a JSON report. Does NOT edit the paper — a human applies
   the swap and drops the inline "Regeneration note."

GPU / timing
------------
A single 24 GB card is sufficient (models run sequentially; peak ~16 GB for the
7B pair in bf16 — no sharding needed). ~30-60 min on an A10/L4, ~15-30 min on an
A100, plus ~10-20 min of weight download (~31 GB) if the node is cold.

Deployment is handled separately (Eigan/Prefect). Run locally with:
    python validation/p2_gem/regeneration/flow_p2_permutation_refreeze.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from prefect import flow, task, get_run_logger

REPO = Path(__file__).resolve().parents[3]          # Rosetta_Analysis root
ABLATE_SCRIPT = REPO / "gem" / "ablate_gem_permutation.py"
DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"
OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_permutation"

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# (HF id, extraction-dir slug, display name) — the three §5.4 models.
MODELS = [
    ("EleutherAI/pythia-6.9b", "EleutherAI_pythia_6.9b", "pythia-6.9b"),
    ("openai-community/gpt2-xl", "openai_community_gpt2_xl", "GPT-2-XL"),
    ("Qwen/Qwen2.5-7B", "Qwen_Qwen2.5_7B", "Qwen2.5-7B"),
]

# Current (1.3.1) canary: pythia-6.9b's 3-node concepts. Established by
# gem/regenerate_table3_handoff_depth.py. If the store still shows the old
# {causation, exfiltration, negation}, the run node has stale gem — abort.
CURRENT_3NODE_PYTHIA69 = {"negation", "sentiment"}
STALE_3NODE_PYTHIA69 = {"causation", "exfiltration", "negation"}

CROSS_THRESH = 0.05  # §5.4's cross-disruption cutoff

# Values currently in the preprint (pre-1.3.1 vintage) — the diff baseline.
PREPRINT_5_4 = {
    "pythia-6.9b": {"multinode": 15, "deepest_dominates": 15,
                    "cross_disrupt_n": 9, "cross_disrupt_mean": 0.11},
    "GPT-2-XL":    {"multinode": 12, "cross_disrupt_n": 10, "cross_disrupt_mean": 0.122,
                    "synergy_mean": -0.040, "synergy_pos": 0},
    "Qwen2.5-7B":  {"multinode": 14, "cross_disrupt_n": 13, "cross_disrupt_mean": 0.324,
                    "synergy_mean": 0.047, "synergy_pos": 10},
}


@task
def vintage_guard() -> None:
    """Abort before GPU time if the gem store is not the current 1.3.1 vintage."""
    log = get_run_logger()
    slug = "EleutherAI_pythia_6.9b"
    three_node = set()
    missing = []
    for c in CONCEPTS:
        p = DATA_ROOT / slug / f"gem_{c}.json"
        if not p.exists():
            missing.append(c)
            continue
        n = len(json.loads(p.read_text()).get("nodes", []))
        if n == 3:
            three_node.add(c)
    if missing:
        raise FileNotFoundError(
            f"pythia-6.9b gem missing for {missing} under {DATA_ROOT/slug}. "
            "Sync the current 1.3.1 paper_n250 gem store to the run node first."
        )
    if three_node == STALE_3NODE_PYTHIA69:
        raise RuntimeError(
            "STALE GEM VINTAGE: pythia-6.9b 3-node concepts are the pre-1.3.1 set "
            f"{sorted(three_node)}. The run node must carry the CURRENT store "
            f"(expected 3-node = {sorted(CURRENT_3NODE_PYTHIA69)}). Re-sync gem before running."
        )
    if three_node != CURRENT_3NODE_PYTHIA69:
        log.warning("pythia-6.9b 3-node set %s != expected current %s — proceeding, "
                    "but verify the store is the intended vintage.",
                    sorted(three_node), sorted(CURRENT_3NODE_PYTHIA69))
    else:
        log.info("Vintage guard OK — current 1.3.1 gem store confirmed.")


@task(retries=1, retry_delay_seconds=60)
def run_permutation(hf_id: str, slug: str) -> str:
    """Run ablate_gem_permutation.py for one model; return its output JSON path."""
    log = get_run_logger()
    out_path = OUT_DIR / f"{slug}_permutation.json"
    cmd = [sys.executable, str(ABLATE_SCRIPT), "--model", hf_id,
           "--device", "auto", "--dtype", "auto", "--overwrite"]
    log.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO))
    if not out_path.exists():
        raise FileNotFoundError(f"ablation did not write {out_path}")
    return str(out_path)


def _deepest_idx(concept: dict) -> int | None:
    hls = [n.get("handoff_layer") for n in concept.get("nodes", [])]
    if not hls or any(h is None for h in hls):
        return None
    return int(np.argmax(hls))


@task
def aggregate(path: str, name: str) -> dict:
    """Compute the §5.4 statistics over this model's multi-node concepts."""
    log = get_run_logger()
    results = json.loads(Path(path).read_text())
    multi = [c for c in results if c.get("n_nodes", 0) >= 2]

    dominates = 0
    for c in multi:
        di = _deepest_idx(c)
        if di is not None and c.get("derived", {}).get("best_single_node") == di:
            dominates += 1

    cross = [c["derived"]["cross_disruption_at_deepest_hl"] for c in multi
             if c.get("derived", {}).get("cross_disruption_at_deepest_hl") is not None]
    cross_hits = [x for x in cross if x > CROSS_THRESH]

    syn = [c["derived"]["synergy_over_best_single"] for c in multi
           if c.get("derived", {}).get("synergy_over_best_single") is not None]

    agg = {
        "model": name,
        "multinode": len(multi),
        "deepest_dominates": dominates,
        "cross_disrupt_n": len(cross_hits),
        "cross_disrupt_mean": round(float(np.mean(cross_hits)), 3) if cross_hits else None,
        "synergy_mean": round(float(np.mean(syn)), 3) if syn else None,
        "synergy_pos": sum(1 for x in syn if x > 0),
    }
    log.info("AGG %s: %s", name, agg)
    return agg


@task
def diff_and_report(aggs: list[dict]) -> dict:
    """Diff vs the preprint and emit ready-to-paste §5.4 sentences."""
    log = get_run_logger()
    lines, report = [], {}
    for a in aggs:
        name = a["model"]
        old = PREPRINT_5_4.get(name, {})
        changed = {k: {"old": old[k], "new": a[k]}
                   for k in old if a.get(k) is not None and a[k] != old[k]}
        report[name] = {"new": a, "preprint": old, "changed": changed}
        n = a["multinode"]
        if name == "pythia-6.9b":
            lines.append(
                f"pythia-6.9b — deepest-node-dominates {a['deepest_dominates']}/{n}; "
                f"cross-disruption {a['cross_disrupt_n']}/{n} (mean {a['cross_disrupt_mean']})")
        else:
            sm = a["synergy_mean"]
            lines.append(
                f"{name} — cross-disruption {a['cross_disrupt_n']}/{n} "
                f"(mean {a['cross_disrupt_mean']}); synergy mean "
                f"{sm:+.3f} ({a['synergy_pos']}/{n} positive)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "p2_section5_4_refreeze_report.json"
    report_path.write_text(json.dumps(
        {"section_5_4_update": lines, "diff_vs_preprint": report}, indent=2))

    changed_any = any(report[m]["changed"] for m in report)
    log.info("\n=== §5.4 UPDATE (apply to gem/preprint.md, then drop the 'Regeneration note') ===\n%s",
             "\n".join(lines))
    if changed_any:
        log.info("Numbers moved from the preprint vintage — see 'changed' in %s", report_path)
    else:
        log.info("Numbers unchanged vs preprint — only the 'Regeneration note' needs dropping.")

    # Optional MLflow logging (no-op if MLflow/URI not configured on the node).
    try:
        import mlflow  # noqa: WPS433
        if mlflow.get_tracking_uri():
            mlflow.set_experiment("p2_gem_section5_4_refreeze")
            with mlflow.start_run(run_name="permutation-refreeze"):
                for a in aggs:
                    for k, v in a.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{a['model']}.{k}", v)
                mlflow.log_artifact(str(report_path))
    except Exception as exc:  # tracking is best-effort
        log.info("MLflow logging skipped (%s)", exc)

    return {"section_5_4_update": lines, "report_path": str(report_path), "changed": changed_any}


@flow(name="p2-permutation-refreeze")
def p2_permutation_refreeze() -> dict:
    vintage_guard()
    aggs = []
    for hf_id, slug, name in MODELS:  # sequential — one model on the GPU at a time
        path = run_permutation(hf_id, slug)
        aggs.append(aggregate(path, name))
    return diff_and_report(aggs)


if __name__ == "__main__":
    p2_permutation_refreeze()
