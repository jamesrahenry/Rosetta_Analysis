"""
regenerate_table3_handoff_depth.py — regenerate P2 (GEM) §4.3 Table 3
("Mean handoff depth by concept") and the §5.4 node-count distribution
from the frozen paper_n250 gem artifacts.

WHY THIS EXISTS
---------------
P2's Table 3 and §5.4 node-count sentence were produced by a one-off that was
never committed. When the exfiltration corpus was re-extracted with corrected
labels (2026-07-17, rosetta_tools 1.3.1) the gem store drifted from the vintage
that generated those numbers — and with no committed generator there was no way
to re-derive them. This script closes that reproducibility gap: it is the
canonical, documented definition going forward.

METRIC (matches the paper's stated definition)
----------------------------------------------
§4.3: "Table 3 shows mean handoff layer relative depth (L_H / N) per concept
type, across all 23 models." §5.4 identifies L_H as the handoff of the dominant
CAZ region ("GEM's handoff layer ... the deepest detected node ... the most
causally concentrated single intervention point"). Operationally the dominant
node is the highest-caz_score node in gem_<concept>.json; L_H is its
`handoff_layer`; relative depth is L_H / N where N = n_layers (from
caz_<concept>.json). Per concept we report mean and median of that fraction
across the 23-model primary corpus (Appendix A).

Node-count distribution (§5.4) is reported for pythia-6.9b — the permutation-
ablation model that section analyses.

STATUS (2026-07-23) — this is the ADOPTED canonical definition
--------------------------------------------------------------
P2's original Table 3 was produced by an uncommitted one-off against a gem
vintage that no longer exists (the gem JSONs were rebuilt at rosetta_tools 1.3.1
after the paper's vintage; the node structure changed — e.g. pythia-6.9b's
3-node concepts went from {causation, exfiltration, negation} to
{negation, sentiment}). Because that vintage is gone, the original numbers are
not reproducible from the frozen store by ANY selection, so P2 §4.3/§5.4 were
re-frozen on THIS generator's output (dominant-node/N over the current store) —
now the single reproducible source of truth. The values differ from the
pre-1.3.1 preprint for concepts whose nodes moved (notably the bimodal ones:
negation 0.696->0.814, deception 0.874->0.782); exfiltration is essentially
unchanged (0.831->0.825; the label correction is sign-invariant for
angular-velocity handoff detection). The §5.4 permutation-ablation *results*
(deepest-node-dominates, cross-disruption) still need a GPU re-run on the
current 15 multi-node concepts — see the Hopper task / EXFIL_RECOMPUTE_PLAN.

USAGE
-----
    python gem/regenerate_table3_handoff_depth.py
    python gem/regenerate_table3_handoff_depth.py --data-dir ~/rosetta_data/paper_n250

Reads gem_<concept>.json + caz_<concept>.json under <data-dir>/<model_slug>/.
Emits the Table 3 markdown, the §5.4 node-count line, and a machine-readable
JSON alongside for provenance.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# 23-model primary corpus (Appendix A), slug -> a display name is derived below.
ROSTER = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b", "EleutherAI_pythia_12b",
    "openai_community_gpt2", "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_1.3b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B",
    "Qwen_Qwen2.5_7B", "Qwen_Qwen2.5_14B",
    "mistralai_Mistral_7B_v0.3", "meta_llama_Llama_3.1_8B",
    "google_gemma_2_2b", "google_gemma_2_9b", "microsoft_phi_2",
]

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

NODE_MODEL = "EleutherAI_pythia_6.9b"  # §5.4 permutation-ablation model


def dominant_handoff_frac(model_dir: Path, concept: str) -> float | None:
    """L_H / N for the dominant (max-caz_score) node, or None if unavailable."""
    gem_p = model_dir / f"gem_{concept}.json"
    caz_p = model_dir / f"caz_{concept}.json"
    if not (gem_p.exists() and caz_p.exists()):
        return None
    gem = json.loads(gem_p.read_text())
    nodes = gem.get("nodes") or []
    if not nodes:
        return None
    n_layers = json.loads(caz_p.read_text()).get("n_layers")
    if not n_layers:
        return None
    dom = max(nodes, key=lambda n: n.get("caz_score", 0.0))
    return dom["handoff_layer"] / n_layers


def build_table3(data_dir: Path) -> list[dict]:
    rows = []
    for concept in CONCEPTS:
        fracs = []
        missing = []
        for slug in ROSTER:
            f = dominant_handoff_frac(data_dir / slug, concept)
            if f is None:
                missing.append(slug)
            else:
                fracs.append(f)
        rows.append({
            "concept": concept,
            "mean": round(statistics.mean(fracs), 3) if fracs else None,
            "median": round(statistics.median(fracs), 3) if fracs else None,
            "n": len(fracs),
            "missing": missing,
        })
    rows.sort(key=lambda r: (r["mean"] is None, r["mean"]))
    return rows


def node_count_distribution(data_dir: Path, model_slug: str) -> dict:
    dist: dict[int, list[str]] = {}
    for concept in CONCEPTS:
        gem_p = data_dir / model_slug / f"gem_{concept}.json"
        if not gem_p.exists():
            continue
        n = len(json.loads(gem_p.read_text()).get("nodes") or [])
        dist.setdefault(n, []).append(concept)
    return {str(k): sorted(v) for k, v in sorted(dist.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "rosetta_data" / "paper_n250"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "results" / "p2_table3_handoff_depth.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir).expanduser()

    rows = build_table3(data_dir)
    nodes = node_count_distribution(data_dir, NODE_MODEL)

    print("**Table 3: Mean handoff depth by concept (23 models)**\n")
    print("| Concept | Mean rel. depth | Median rel. depth |")
    print("|---------|-----------------|-------------------|")
    for r in rows:
        print(f"| {r['concept']} | {r['mean']} | {r['median']} |")
    incomplete = [r for r in rows if r["n"] < len(ROSTER)]
    if incomplete:
        print(f"\n> coverage note: {', '.join(f'{r['concept']} n={r['n']}' for r in incomplete)}")

    print(f"\n§5.4 node-count distribution ({NODE_MODEL.split('_')[-1]}):")
    for k, v in nodes.items():
        print(f"  {k} node(s): {len(v)} concepts -> {v}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "metric": "dominant-node (max caz_score) handoff_layer / n_layers",
        "roster": ROSTER,
        "table3": rows,
        "node_counts_model": NODE_MODEL,
        "node_counts": nodes,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
