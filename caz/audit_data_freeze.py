#!/usr/bin/env python3
"""
audit_data_freeze.py — Data freeze audit for the CAZ validation paper.

Checks every quantitative claim in the paper against the actual data files.
Run this after the final GPU jobs complete (tc7d8a32, tee0f0cc, tf3a41d3)
and before submission.

Usage:
    cd caz_scaling/
    python src/audit_data_freeze.py

Exits 0 if all checks pass. Exits 1 if any FAIL or WARN.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT        = Path(__file__).resolve().parents[1]
RESULTS     = ROOT / "results"
LIB_CAZS    = ROOT / "feature_library" / "cazs"
SCORED_CSV  = ROOT / "SCORED_CAZ_ANALYSIS.csv"
CONTROL_JSON = ROOT / "CAZ_ABLATION_CONTROL.json"

# ── 26 base models expected in the paper ──────────────────────────────────────
BASE_MODELS = {
    # Pythia family (7)
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",  "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # GPT-2 family (4)
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    # OPT family (5)
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    # Qwen 2.5 base (4)
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",   "Qwen/Qwen2.5-7B",
    # Gemma 2 base (2)
    "google/gemma-2-2b", "google/gemma-2-9b",
    # Llama 3.2 base (2)
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    # Mistral (1)
    "mistralai/Mistral-7B-v0.3",
    # Phi (1)
    "microsoft/phi-2",
}

# ── 20 models in the §7.2 dark matter census ──────────────────────────────────
# Table 12 confirms 4 architecture families: Pythia, GPT-2, OPT, Qwen (7+4+5+4).
# Gemma/Llama/Mistral/Phi-2 deepdives ran after the census section was written.
DARK_MATTER_MODELS = {
    # Pythia family (7)
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",  "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # GPT-2 family (4)
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    # OPT family (5)
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    # Qwen 2.5 base (4)
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",   "Qwen/Qwen2.5-7B",
}

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

PASS, WARN, FAIL = "✓ PASS", "⚠ WARN", "✗ FAIL"

issues = []


def check(label: str, condition: bool, found, expected, warn_only: bool = False) -> str:
    status = PASS if condition else (WARN if warn_only else FAIL)
    line   = f"  {status}  {label}"
    if not condition:
        line += f"\n         expected: {expected}\n         found:    {found}"
        issues.append((status, label, expected, found))
    print(line)
    return status


def latest_run(prefix: str) -> Path | None:
    """Return the most recent results dir matching a prefix."""
    matches = sorted([d for d in RESULTS.iterdir() if d.name.startswith(prefix)], reverse=True)
    return matches[0] if matches else None


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").replace(".", "_")


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMPLETENESS — extraction data for all 26 base models × 7 concepts
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Extraction completeness (26 models × 7 concepts) ─────────────────")

missing_extraction = []
for mid in sorted(BASE_MODELS):
    short = mid.split("/")[-1]
    # Find the canonical run directory
    run = latest_run(f"gpt2_{model_slug(mid)}") or \
          latest_run(f"pythia_{model_slug(mid)}") or \
          latest_run(f"custom_{model_slug(mid)}")

    # More flexible: any dir with run_summary.json matching this model_id
    run = None
    for d in sorted(RESULTS.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if sf.exists():
            try:
                if json.load(open(sf)).get("model_id") == mid:
                    run = d
                    break
            except Exception:
                continue

    if run is None:
        missing_extraction.append(short)
        continue

    for concept in CONCEPTS:
        caz_file = run / f"caz_{concept}.json"
        if not caz_file.exists():
            missing_extraction.append(f"{short}/{concept}")

check(
    f"All 26 models × 7 concepts have extraction data",
    len(missing_extraction) == 0,
    f"Missing: {missing_extraction}" if missing_extraction else "none",
    "none missing",
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TABLE 2 — CAZ score distribution (should total 623 across 26 base models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Table 2 — CAZ score distribution ─────────────────────────────────")

# Load scored CAZ CSV; count only base models
cat_counts = defaultdict(int)
total_cazs = 0
scored_models = set()

with open(SCORED_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        mid = row["model_id"]
        if mid not in BASE_MODELS:
            continue
        scored_models.add(mid)
        score = float(row["caz_score"])
        if score > 0.5:
            cat_counts["black_hole"] += 1
        elif score > 0.2:
            cat_counts["strong"] += 1
        elif score > 0.05:
            cat_counts["moderate"] += 1
        else:
            cat_counts["gentle"] += 1
        total_cazs += 1

print(f"  Models counted: {len(scored_models)}/26")
print(f"  Black hole (>0.5):  {cat_counts['black_hole']:4d}   (paper says 103)")
print(f"  Strong  (0.2–0.5):  {cat_counts['strong']:4d}   (paper says  86)")
print(f"  Moderate(0.05–0.2): {cat_counts['moderate']:4d}   (paper says 117)")
print(f"  Gentle  (<0.05):    {cat_counts['gentle']:4d}   (paper says 317)")
print(f"  Total:              {total_cazs:4d}   (paper says 623)")

check("Table 2 total matches paper (623)", total_cazs == 623, total_cazs, 623)
check("Table 2 gentle count",    cat_counts["gentle"]    == 317, cat_counts["gentle"],    317, warn_only=True)
check("Table 2 black_hole count",cat_counts["black_hole"] == 103, cat_counts["black_hole"], 103, warn_only=True)
check("All 26 base models in scored CSV", len(scored_models) == 26, len(scored_models), 26)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TABLE 9 — Ablation efficacy by CAZ score category
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. Table 9 — Ablation efficacy by category ──────────────────────────")

# Load ablation results from feature library + per-model ablation files
# Strategy: use the scored CAZ CSV (which has caz_score) and match to ablation files

abl_by_cat = defaultdict(list)   # cat → list of (self_retained_pct, effective bool)
models_with_ablation = set()

for mid in sorted(BASE_MODELS):
    # Find run dir
    run = None
    for d in sorted(RESULTS.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if sf.exists():
            try:
                if json.load(open(sf)).get("model_id") == mid:
                    run = d
                    break
            except Exception:
                continue
    if run is None:
        continue

    for concept in CONCEPTS:
        # Load multimodal ablation results (N-CAZ per-CAZ format)
        # These contain self_retained_pct per CAZ — the authoritative source for Table 9.
        abl_file = run / f"ablation_multimodal_{concept}.json"
        if not abl_file.exists():
            continue

        models_with_ablation.add(mid)
        abl = json.load(open(abl_file))

        for caz in abl.get("cazs", []):
            score    = caz["caz_score"]
            retained = caz["self_retained_pct"]          # % of baseline sep retained
            effective = retained <= 80.0                  # ≥20% suppression → effective

            if score > 0.5:    cat = "black_hole"
            elif score > 0.2:  cat = "strong"
            elif score > 0.05: cat = "moderate"
            else:               cat = "gentle"

            abl_by_cat[cat].append((retained, effective))

print(f"  Models with ablation data: {len(models_with_ablation)}")
print()
for cat, label, paper_n, paper_eff, paper_ret in [
    ("black_hole", "Black hole (>0.5)",   73, 100, 30.7),
    ("strong",     "Strong (0.2–0.5)",    62,  95, 39.8),
    ("moderate",   "Moderate(0.05–0.2)",  87,  93, 45.6),
    ("gentle",     "Gentle (<0.05)",     245,  93, 48.7),
]:
    results = abl_by_cat[cat]
    if not results:
        print(f"  {WARN}  {label}: no data")
        continue
    n     = len(results)
    eff   = 100 * sum(1 for _, e in results if e) / n
    ret   = sum(r for r, _ in results) / n          # self_retained_pct (already %)
    print(f"  {label:25s}  n={n:4d} (paper={paper_n:4d})  "
          f"effective={eff:.0f}% (paper={paper_eff}%)  "
          f"self-retained={ret:.1f}% (paper={paper_ret}%)")

total_abl = sum(len(v) for v in abl_by_cat.values())
check("Table 10 total ablations near 467", abs(total_abl - 467) <= 20,
      total_abl, "467 ±20", warn_only=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DARK MATTER CENSUS — §7.2 numbers
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Dark matter census (§7.2) ────────────────────────────────────────")

# Paper claims: 20 models, 2096 total features, 414 persistent, 42 labeled (2%), 12 relay
# §7.2 census covers Pythia×7 + GPT-2×4 + OPT×5 + Qwen×4 = 20 models (DARK_MATTER_MODELS).
RELAY_THRESHOLD_COS = 0.5
RELAY_THRESHOLD_LAYERS = 2   # different concept at different depth

census_models = 0
total_features = 0
persistent_features = 0
labeled_features = 0
relay_features = 0


def find_deepdive(mid: str) -> Path | None:
    """Return the most recent deepdive dir for a model, without matching variants.

    Uses a timestamp guard: after the slug prefix, the next chars must be 8 digits
    (the YYYYMMDD timestamp). This avoids matching gpt2_large when looking for gpt2,
    or Qwen2.5-0.5B-Instruct when looking for Qwen2.5-0.5B.
    """
    slug   = mid.replace("/", "_").replace("-", "_")
    prefix = f"deepdive_{slug}_"
    matches = sorted(
        [d for d in RESULTS.iterdir()
         if d.name.startswith(prefix) and d.name[len(prefix):len(prefix) + 8].isdigit()],
        reverse=True,
    )
    return matches[0] if matches else None


for mid in sorted(DARK_MATTER_MODELS):
    runs_dir = find_deepdive(mid)
    if runs_dir is None:
        runs = []
    else:
        runs = [runs_dir]
    if not runs:
        continue
    fm_file = runs_dir / "feature_map.json"
    if not fm_file.exists():
        continue

    fm = json.load(open(fm_file))
    census_models += 1
    features = fm["features"]
    total_features    += len(features)
    persistent_features += sum(1 for f in features if f.get("is_persistent", False))

    for f in features:
        align = f.get("concept_alignment", {})
        if any(v >= 0.5 for v in align.values()):
            labeled_features += 1

        # Relay: aligns with ≥2 different concepts at different layer ranges
        traj = f.get("concept_alignment_trajectory", {})
        if len(traj) >= 2:
            concepts_by_layer = defaultdict(list)
            for concept, by_layer in traj.items():
                for layer_str, cos in by_layer.items():
                    if float(cos) >= RELAY_THRESHOLD_COS:
                        concepts_by_layer[int(layer_str)].append(concept)
            # Find layers where dominant concept changes
            layer_concepts = {}
            for layer, concepts in sorted(concepts_by_layer.items()):
                if concepts:
                    layer_concepts[layer] = concepts[0]
            unique_concepts = list(dict.fromkeys(layer_concepts.values()))
            if len(unique_concepts) >= 2:
                relay_features += 1

pct_labeled = 100 * labeled_features / total_features if total_features else 0

print(f"  Models with deepdive:   {census_models:4d}  (paper says 20)")
print(f"  Total features:         {total_features:4d}  (paper says 2,096)")
print(f"  Persistent (5+ layers): {persistent_features:4d}  (paper says 414)")
print(f"  Labeled (cos>0.5):      {labeled_features:4d}  ({pct_labeled:.1f}%, paper says 42, 2.0%)")
print(f"  Relay features:         {relay_features:4d}  (paper says 12)")

check("Dark matter census: 20 models", census_models == 20, census_models, 20)
check("Dark matter census: ~2096 features",
      abs(total_features - 2096) <= 100, total_features, "2096 ±100", warn_only=True)
check("Dark matter labeled ≈2%",
      abs(pct_labeled - 2.0) <= 1.0, f"{pct_labeled:.1f}%", "2.0% ±1.0", warn_only=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONTROL ABLATION — §6.1 (83.3% aligned vs 94.9% dark matter)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. Control ablation (§6.1) ───────────────────────────────────────────")

ctrl = json.load(open(CONTROL_JSON))
n_features  = ctrl["n_features"]
n_models    = ctrl["n_models"]
aligned_ret = ctrl["alignment"]["overall"]["aligned"]["mean"]
dark_ret    = ctrl["alignment"]["overall"]["dark_matter"]["mean"]
gap         = dark_ret - aligned_ret

print(f"  Features ablated:      {n_features}  (paper says 286)")
print(f"  Models:                {n_models}  (paper says 32)")
print(f"  Aligned mean retained: {aligned_ret:.1f}%  (paper says 83.3%)")
print(f"  Dark matter retained:  {dark_ret:.1f}%  (paper says 94.9%)")
print(f"  Gap:                   {gap:.1f}pp  (paper says 11.6pp)")

check("Control: n_features == 286", n_features == 286, n_features, 286)
check("Control: n_models == 32",    n_models   == 32,  n_models,   32)
check("Control: aligned mean ≈83.3%", abs(aligned_ret - 83.3) < 0.5,
      f"{aligned_ret:.1f}%", "83.3%")
check("Control: dark mean ≈94.9%",   abs(dark_ret - 94.9) < 0.5,
      f"{dark_ret:.1f}%", "94.9%")
check("Control: gap ≈11.6pp",        abs(gap - 11.6) < 0.5,
      f"{gap:.1f}pp", "11.6pp")


# ══════════════════════════════════════════════════════════════════════════════
# 6. POST-BUG GPT-2 RUNS — confirm no pre-fix data remains
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. GPT-2 extraction directory bug (fixed 2026-04-02) ─────────────────")

gpt2_models = [mid for mid in BASE_MODELS if "gpt2" in mid.lower()]
for mid in sorted(gpt2_models):
    run = None
    for d in sorted(RESULTS.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if sf.exists():
            try:
                data = json.load(open(sf))
                if data.get("model_id") == mid:
                    run = d
                    break
            except Exception:
                continue
    if run is None:
        print(f"  {WARN}  {mid.split('/')[-1]}: no run found")
        continue
    # Bug was fixed on 2026-04-02; run timestamp should be after that
    ts = run.name.split("_")[-2] + "_" + run.name.split("_")[-1]  # YYYYMMDD_HHMMSS
    date_str = ts[:8]
    ok = date_str >= "20260402"
    status = PASS if ok else FAIL
    print(f"  {status}  {mid.split('/')[-1]:15s}  run={ts[:8]}  {'post-fix' if ok else 'PRE-FIX — RERUN NEEDED'}")
    if not ok:
        issues.append((FAIL, f"GPT-2 pre-fix run: {mid}", ">=20260402", date_str))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
fails = [i for i in issues if i[0] == FAIL]
warns = [i for i in issues if i[0] == WARN]

if not issues:
    print("ALL CHECKS PASSED — data is consistent with paper claims.")
    print("Safe to freeze dataset and proceed to submission.")
else:
    if fails:
        print(f"FAILS ({len(fails)}) — resolve before submission:")
        for _, label, exp, found in fails:
            print(f"  ✗  {label}")
            print(f"       expected {exp}, found {found}")
    if warns:
        print(f"\nWARNINGS ({len(warns)}) — review and update paper text if needed:")
        for _, label, exp, found in warns:
            print(f"  ⚠  {label}")
            print(f"       expected {exp}, found {found}")

print()
sys.exit(1 if fails else 0)
