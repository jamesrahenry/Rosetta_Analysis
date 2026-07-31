#!/usr/bin/env python3
"""Number-provenance verifier: assert manuscript numbers against their artifacts.

WHY THIS EXISTS
---------------
On 2026-07-29 a pre-publication sweep found that P3's supplementary §E reported
opt-350m's split-half tau as 0.491 while the main text said 0.310. The correct
value (0.310) was sitting in a tracked artifact — `scripts/results/
c5_splithalf_tau_ceiling.json` — the whole time. Three prior review passes,
including one that explicitly fixed the same statistic in the main text, missed
the supplement. Nothing in the pipeline compared manuscript prose to the JSON
it came from, so the only way to catch it was for a human to notice two numbers
disagreeing across a 600-line file boundary.

This script closes that gap. Each paper declares a MANIFEST of
(claim -> artifact -> expected value) triples. The script reads the artifact,
recomputes/extracts the value, and asserts the manuscript text contains it. A
number that has drifted from its artifact fails here instead of at a referee.

WHAT IT CATCHES
---------------
1. Manuscript number != artifact value (the opt-350m class).
2. A value present in one file but stale in the other (main vs supplementary).
3. Values whose artifact no longer exists / no longer contains the key
   (the "script became unrecoverable" class, P3 §8.7).
4. Superseded-vintage values that must appear NOWHERE (purge lists: P4's
   descoped numbers, retracted framings, withdrawn scale claims).

WHAT IT DOES NOT CATCH
----------------------
Prose claims with no numeric anchor, wrong-but-self-consistent reasoning, and
inverted arguments (P2's §9.2 "rotate least" error was a *correct* number
attached to the wrong concept names). Those still need a reading reviewer.
Numeric agreement is a floor, not a substitute for review.

USAGE
    python3 papers/shared/scripts/verify_claims.py             # all papers
    python3 papers/shared/scripts/verify_claims.py --paper p3
Exit code 0 = all checks pass; 1 = at least one failure.

ADDING A CHECK  (do this whenever you fix or cite a number)
    check("<label>", artifact="<path under paper dir>", value=<fn or literal>,
          text="<regex or literal as it appears in prose>", files=[...])
Rule of thumb: if a reviewer had to open a JSON to settle a number, that number
belongs here before the session ends.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, median

PAPERS_DIR = Path(__file__).resolve().parents[2]  # .../papers/
FAILURES: list[str] = []
PASSES = 0

# P2 §8.1's authoritative artifact tree. Lives in the frozen store rather than
# under papers/ because it is published as part of the Rosetta Activations
# dataset; the superseded non-pair-aware run in ~/rosetta_data/results/ is NOT
# this and must never be read for §8.1 values.
P2_ESTIMATOR_DIR = Path.home() / "rosetta_data" / "paper_n250" / "_p2_direction_estimator"


P2_EXFIL_DIR = Path.home() / "rosetta_data" / "paper_n250" / "_p2_exfil_width1"


def _p2_headline():
    """(wins, n, widths, near_final_count) over the 29-model roster, using the
    CURRENT exfiltration artifacts rather than the superseded main-tree ones."""
    root = Path.home() / "rosetta_data" / "paper_n250"
    if not P2_EXFIL_DIR.is_dir() or not root.is_dir():
        return None
    excl = {"EleutherAI_gpt_neo_125m"}
    gem = {}
    for f in root.glob("*/gem_*.json"):
        d = json.loads(f.read_text())
        gem[(f.parent.name, d["concept"])] = d
    rows = []
    for f in root.glob("*/ablation_gem_*.json"):
        if f.parent.name in excl or "superseded" in f.name:
            continue
        d = json.loads(f.read_text())
        if d["concept"] == "exfiltration":
            continue
        rows.append((f.parent.name, d))
    for f in P2_EXFIL_DIR.glob("*/ablation_gem_exfiltration.json"):
        rows.append((f.parent.name, json.loads(f.read_text())))
    if len(rows) != 493:
        return None
    wins = sum(1 for _, d in rows if d["comparison"]["handoff_better"])
    widths = {d["handoff"].get("width") for _, d in rows}
    near = 0
    for s, d in rows:
        ns = gem.get((s, d["concept"]), {}).get("nodes") or []
        L = d["handoff"].get("ablation_layers") or []
        if ns and L and min(L) / ns[0]["n_layers_total"] > 0.85:
            near += 1
    return wins, len(rows), widths, near


P2_SITE_MATCHED_DIR = Path.home() / "rosetta_data" / "paper_n250" / "_gem_depth_matched"


def _p2_site_matched():
    """Per-cell rows of the site-matched depth control, or None."""
    if not P2_SITE_MATCHED_DIR.is_dir():
        return None
    rows = []
    for f in sorted(P2_SITE_MATCHED_DIR.glob("*_depth_matched_control.json")):
        d = json.loads(f.read_text())
        model = d["model_id"].split("/")[-1]
        for r in d.get("results", []):
            s = r.get("site_matched_control") or {}
            if s.get("site_matched"):
                rows.append({"model": model, "win": bool(s["handoff_better"]),
                             "delta": s["delta_pp"]})
    return rows or None


def _p2_estimator_cells():
    """Per-cell rows from the pair-aware supervised comparison, or None."""
    if not P2_ESTIMATOR_DIR.is_dir():
        return None
    rows = []
    for f in sorted(P2_ESTIMATOR_DIR.glob("*_direction_estimator.json")):
        d = json.loads(f.read_text())
        pair_aware = bool(d.get("pair_aware_split"))
        for r in d.get("detail", []):
            rows.append({**r, "_pair_aware": pair_aware})
    return rows or None


def _load(paper: str, rel: str):
    p = PAPERS_DIR / paper / rel
    if not p.exists():
        return None, f"artifact missing: {paper}/{rel}"
    try:
        return json.loads(p.read_text()), None
    except Exception as e:  # noqa: BLE001
        return None, f"artifact unreadable: {paper}/{rel} ({e})"


def _text(paper: str, files: list[str]) -> dict[str, str]:
    out = {}
    for f in files:
        p = PAPERS_DIR / paper / f
        out[f] = p.read_text() if p.exists() else ""
    return out


def check(label, paper, files, text, artifact=None, value=None, expect_absent=False):
    """Assert `text` appears in `files` (or is absent), consistent with `artifact`.

    `value` may be a literal or a callable taking the loaded artifact. When
    given, the rendered `text` must contain the artifact-derived value, so the
    artifact is the source of truth rather than a second place to drift.
    """
    global PASSES
    art = None
    if artifact:
        art, err = _load(paper, artifact)
        if err:
            FAILURES.append(f"[{paper}] {label}: {err}")
            return
    if value is not None:
        got = value(art) if callable(value) else value
        if got not in text:
            FAILURES.append(
                f"[{paper}] {label}: artifact gives {got!r} but the check "
                f"string is {text!r} — manifest and artifact disagree"
            )
            return
    bodies = _text(paper, files)
    hits = {f: len(re.findall(re.escape(text), b)) for f, b in bodies.items()}
    total = sum(hits.values())
    if expect_absent and total:
        where = ", ".join(f"{f}x{n}" for f, n in hits.items() if n)
        FAILURES.append(f"[{paper}] {label}: SUPERSEDED value {text!r} still present ({where})")
    elif not expect_absent and not total:
        FAILURES.append(
            f"[{paper}] {label}: {text!r} not found in {', '.join(files)} "
            f"(artifact says it should be there)"
        )
    else:
        PASSES += 1


# ---------------------------------------------------------------------------
# P3 (caz-validation) — seeded from the 2026-07-29 sweep. Extend as you fix.
# ---------------------------------------------------------------------------
def manifest_p3():
    P, MD = "caz-validation", ["preprint.md", "supplementary.md"]
    C5 = "scripts/results/c5_splithalf_tau_ceiling.json"

    def half(model):
        return lambda a: f"{a['results']['oddeven']['per_model_splithalf_tau'][model]:.3f}"

    def agg(key, nd=3):
        return lambda a: f"{a['results']['oddeven'][key]:.{nd}f}"

    # The exact bug this file was written for.
    check("C5 opt-350m split-half tau", P, MD, "0.310", C5, half("facebook_opt_350m"))
    check("C5 opt-350m PRE-CORRECTION value purged", P, MD, "0.491", expect_absent=True)
    check("C5 gpt2-medium split-half tau", P, MD, "0.895", C5, half("openai_community_gpt2_medium"))
    check("C5 gpt2-medium PRE-CORRECTION value purged", P, MD, "0.688 (gpt2-medium)", expect_absent=True)
    check("C5 corpus median split-half", P, MD, "0.717", C5, agg("median_splithalf_tau_125"))
    check("C5 reliability ceiling", P, MD, "0.911", C5, agg("attenuation_ceiling_tau"))
    check("C5 ceiling (rho)", P, MD, "0.921", C5, agg("attenuation_ceiling_rho"))
    check("C5 Spearman-Brown at 250", P, MD, "0.835", C5, agg("spearman_brown_rel_250_tau"))
    check("C5 grand-mean reliability", P, MD, "0.993", C5, agg("rel_grandmean_tau"))
    check("C5 observed median tau", P, MD, "0.404", C5, agg("observed_median_tau_vs_grandmean"))
    check("C5 within-model mean tau", P, MD, "0.651", C5, agg("within_model_mean_tau"))
    check("C5 between-model mean tau", P, MD, "0.254", C5, agg("between_model_mean_tau"))
    check("C5 gemma-2-2b split-half", P, MD, "0.176", C5, half("google_gemma_2_2b"))
    check("C5 gemma-2-9b split-half", P, MD, "0.184", C5, half("google_gemma_2_9b"))
    check("C5 Qwen2.5-3B split-half (highest)", P, MD, "0.992", C5, half("Qwen_Qwen2.5_3B"))

    # LOO: recomputed 2026-07-29 from depth_pivot.csv -> 0.3725 -> 0.373.
    check("LOO median tau (recomputed)", P, MD, "0.373")
    check("LOO stale 0.372 purged", P, MD, "median τ is 0.372", expect_absent=True)

    # Provenance-flagged statistics that must stay withdrawn (ledger rows 23/24).
    check("6.4 divergence magnitude withdrawn", P, ["preprint.md"],
          "by a mean 41.3 pp", expect_absent=True)
    check("6.7 CKA direction not published as a result", P, ["preprint.md"],
          "The result is the opposite (within-CAZ mean: 0.962", expect_absent=True)

    # P4 descope purge list (arbiter decision 2026-07-29).
    for v in ["0.9750", "Δ=+0.195", "1,666/1,666", "universality 0.204", "0.0279"]:
        check(f"P4 purge: {v}", P, MD, v, expect_absent=True)
    check("2026d pointer wording", P, MD, "companion submission; arXiv pending", expect_absent=True)

    # --- Round 2: the scripts/*_results.json tier, never diffed before. The BoW
    # ordering control is the surface-lexical confound control for the headline
    # ordering result, and its artifact records SEVEN measures, one significant.
    BOW = "scripts/c12_bow_ordering_results.json"

    def bow_max_tau(a):
        return f"{max(v['tau'] for v in a['correlations_vs_depth'].values()):.3f}"

    check("BoW full tau range reported", P, MD, "0.18–0.38")
    check("BoW significant variant reported", P, MD, "τ = 0.382", BOW, bow_max_tau)
    check("BoW truncated range purged", P, MD, "0.18–0.24", expect_absent=True)
    check("BoW multiple-comparisons defence present", P, MD, "0.034 × 7")
    # Enrichment threshold sweep: supp §H's derivation is 1.7x-20x; the 1.2x-10x
    # form was wrong in P3 and had been propagated into P1.
    check("enrichment sweep matches supp §H", P, MD, "≈20× (50%)")
    check("wrong 1.2x-10x enrichment range purged", P, MD, "1.2× (5%) to 10× (50%)", expect_absent=True)
    # Values that were the defective-exfiltration branch or contradicted supp §B.
    check("divergence magnitude from corrected branch", P, MD, "≈21.9 pp")
    check("defective-branch 22.6 pp purged", P, MD, "≈22.6 pp", expect_absent=True)
    check("Qwen2.5-3B mean region score", P, MD, "lowest mean region score in the corpus (0.185")
    check("stale Qwen2.5-3B 0.177 purged", P, MD, "region score in the corpus (0.177", expect_absent=True)
    check("Qwen CAZ-count range", P, MD, "51–65 per model")
    check("stale Qwen 50-64 range purged", P, MD, "50–64 per model", expect_absent=True)
    check("score-formula gap range", P, MD, "gap 11.5–17.9 pp")
    check("coasting stability range", P, MD, "78–81% (per-model-concept mean)")
    # F3 (round 2): the §6.4 divergence headline had spliced a 360-cell causal-
    # deeper rate (stale p3_c3_results.json) with a 358-cell null. Reran
    # scripts/apply_exfil_c3.py — corrected branch harmonises both at n=358:
    # divergence 50.3%, deeper 94.4%, null 57.9%±2.5%, z=-3.02, p=0.0016, k̄≈2.38.
    C3 = "scripts/results/apply_exfil_c3.json"

    def c3_deeper(a):
        return f"{a['corrected']['causal_deeper_rate'] * 100:.1f}%"

    def c3_ncells(a):
        return str(a["corrected"]["n_cells"])

    check("F3 causal-deeper rate (corrected, n=358)", P, MD, "94.4%", C3, c3_deeper)
    check("F3 divergence population n=358", P, MD, "358 multimodal cases", C3, c3_ncells)
    check("F3 stale 93.9% deeper rate purged", P, MD, "93.9%", expect_absent=True)
    check("F3 spliced 360-cell divergence label purged", P, MD,
          "360 multimodal cases", expect_absent=True)
    check("F3 stale 360-cell divergence label purged (supp)", P, MD,
          "360 cases in §6.4", expect_absent=True)
    check("F3 k-bar corrected to 2.38", P, MD, r"\bar{k} \approx 2.38")
    check("F3 stale k-bar 2.2 purged", P, MD, r"\bar{k} \approx 2.2$", expect_absent=True)
    # F5 (round 2): §6.1 read as if clustering *raised* the 3.59x enrichment to
    # 4.29x. 4.29x is a mean-of-ratios on a simplified classification. Clustering
    # the actual Table 7 ratio-of-means is a no-op (3.60x [3.28, 3.97]).
    F5 = "scripts/results/f5_enrichment_clusterboot_results.json"

    def f5_boot(a):
        return f"{a['cluster_boot_mean']:.2f}×"

    def f5_ci(a):
        return f"[{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]"

    check("F5 ratio-of-means no-op reported", P, MD, "3.60×", F5, f5_boot)
    check("F5 no-op CI reported", P, MD, "[3.28, 3.97]", F5, f5_ci)
    check("F5 4.29x marked a different estimand", P, MD, "different estimand")
    # F13/F14 (round 2): Table 1 architecture facts. Gemma-2's activation is
    # GeGLU (gelu_pytorch_tanh, verified locally), not SwiGLU; Mistral-7B-v0.3
    # has sliding_window: null (SWA removed at v0.2) — it is NOT a sliding-window
    # model. No JSON artifact; these are config facts, guarded by text.
    check("F14 Gemma activation is GeGLU", P, MD, "GeGLU")
    check("F14 Gemma §B header now GeGLU (only Gemma is GeGLU)", P, MD, "RoPE, GeGLU")
    check("F13 Mistral GQA+SW cell purged", P, MD, "| GQA+SW |", expect_absent=True)
    check("F13 Mistral sliding-window attribution purged (main)", P, MD,
          "sliding window attention (Mistral)", expect_absent=True)
    check("F13 Mistral sliding-window GQA label purged", P, MD,
          "Mistral's sliding-window GQA", expect_absent=True)
    check("F13 Mistral §B sliding-window header purged", P, MD,
          "GQA, RoPE, sliding window", expect_absent=True)
    # F15-F18/F34/F35 (round 2): reference-list accuracy. No JSON artifacts —
    # these are citation facts verified against the sources (WebFetch 2026-07-30).
    check("F18 TimeBank->TimeML citation corrected", P, MD, "TimeML: Robust specification")
    check("F18 non-existent venue purged", P, MD,
          "AAAI Workshop on Temporal Inference", expect_absent=True)
    check("F15 Habernal year corrected to 2017", P, MD, "Habernal & Gurevych, 2017")
    check("F15 Habernal wrong year 2016 purged", P, MD,
          "Habernal & Gurevych, 2016", expect_absent=True)
    check("F16 Tenney no-interventions annotation", P, MD, "without causal interventions")
    check("F16 Tenney false causal claim purged", P, MD,
          "foundational evidence that geometric salience", expect_absent=True)
    check("F17 mean-shift attributed to activation-addition lineage", P, MD,
          "difference-of-means activation-addition lineage")
    check("F35 Phi-2 param count corrected", P, MD, "| 2.7B |")
    check("F35 Phi-2 wrong 2.8B purged", P, MD, "| 2.8B |", expect_absent=True)
    check("F35 Vig title corrected", P, MD,
          "Causal mediation analysis for interpreting neural NLP")
    check("F35 OpenAI Blog mis-parsed volume purged", P, MD,
          "*OpenAI Blog*, 1(8), 9", expect_absent=True)
    check("F35 UzZaman now cited (temporal ordering)", P, MD, "UzZaman et al., 2013")
    check("F35 uncited SentEval reference removed", P, MD, "SentEval", expect_absent=True)
    # F6/F8 (round 2, RESOLVED BY MEASUREMENT 2026-07-30): the site-matched depth
    # control (HF paper_n250/_gem_depth_matched/) is null — the handoff advantage
    # and Gemma's final-global-layer recovery are both depth effects. James ruled
    # concede both. Numbers are the run of record (SITE_MATCHED_DEPTH_CONTROL_RESULT
    # + patch.py per-layer output); local recompute reproduces the null (see ledger).
    check("F8 depth control §6.5-population rate (roster A)", P, MD, "handoff-better rate is 51.8%")
    check("F8 depth control §6.5-population p (median, N=28)", P, MD, "p = 0.779, N = 28")
    check("F8 depth control TNCONF sub-roster rate (roster B)", P, MD, "reads 55.2% (p = 0.791)")
    check("F8 depth-matched artifact named", P, MD, "_gem_depth_matched")
    check("F8 mislabelled 25-model number not cited as §6.5 population", P, MD,
          "population the depth-matched handoff-better rate is 55.2%", expect_absent=True)
    check("F8 settled-output inference withdrawn", P, MD,
          "behaves as if it holds the settled output", expect_absent=True)
    check("F6 universal near-readout recovery reported", P, MD,
          "MHA 1.002, GQA 1.003, Gemma-2 1.004")
    check("F6 Gemma localization demoted (§9)", P, MD,
          "The one architecture-specific localization that survives is Gemma", expect_absent=True)
    check("F6 coasting 'confirming' softened", P, MD,
          "confirming that the concept direction has reached its causally-active stable form",
          expect_absent=True)
    # F8 artifact anchor: the local cross-check reproduces the shallow-site
    # comparator (legacy arm) that §8.7 cites as 94.2%, proving the depth-control
    # artifact is the right one; the site-matched arm is null (51.8%, p=0.63 at
    # N=28, matching the run-of-record's 55.2%/p=0.791 at N=25 — same conclusion).
    F8DM = "scripts/results/f8_depth_matched_recompute_check.json"

    def f8_legacy_pct(a):
        return f"{a['legacy_arm_pct_rosterA']:.1f}%"

    def f8_rosterA_pct(a):
        return f"{a['rosterA_28model_sec65']['handoff_better_pct']:.1f}%"

    def f8_rosterA_p(a):
        return f"p = {a['rosterA_28model_sec65']['model_level_median']['p']:.3f}"

    check("F8 shallow-site comparator reproduces (artifact anchor)", P, MD,
          "94.2%", F8DM, f8_legacy_pct)
    check("F8 §6.5-population rate matches recompute", P, MD,
          "handoff-better rate is 51.8%", F8DM, f8_rosterA_pct)
    check("F8 §6.5-population p matches recompute (median)", P, MD,
          "p = 0.779", F8DM, f8_rosterA_p)


# ---------------------------------------------------------------------------
# P2 (gem)
# ---------------------------------------------------------------------------
def manifest_p2():
    P, MD = "gem", ["preprint.md"]
    # Frozen headline — must be identical at every site.
    check("frozen headline 341/493", P, MD, "341/493")
    check("frozen Wilcoxon W", P, MD, "W=356")
    check("superseded headline purged", P, MD, "350/493", expect_absent=True)
    check("superseded W purged", P, MD, "W=79", expect_absent=True)
    # Values §5.5 disavows must not be cited as evidence elsewhere.
    check("disavowed depth-matched advantage purged", P, MD, "+32.5pp mean advantage", expect_absent=True)
    check("disavowed depth-matched n purged", P, MD, "374 clean pairs", expect_absent=True)
    check("current depth-matched control", P, MD, "94.2%")
    # Withdrawn claims must not survive in the body.
    check("withdrawn scale threshold purged", P, MD,
          "consistent with a genuine scale threshold around 400–500M rather", expect_absent=True)
    # Version labelling (James ruling 2026-07-29).
    check("version labelled v2", P, MD, "Changes in version 2")
    check("v3 labelling purged", P, MD, "Version 3", expect_absent=True)
    # Strict drop of the descoped companion.
    check("2026d strictly dropped", P, MD, "2026d", expect_absent=True)
    # Companion values.
    check("P3 dependency split current", P, MD, "17.6%")
    check("P3 dependency stale purged", P, MD, "17.5% of testable CAZ pairs", expect_absent=True)

    # --- Round 2: values recomputed from ~/rosetta_data/paper_n250 (the frozen
    # store the paper cites). Counted directly from ablation_gem_*.json
    # handoff.final_retained_pct > 100 and gem_*.json node geometry.
    check("gpt2 ablation-pathology rate", P, MD, "In 12/17 pairs, gpt2's handoff ablation")
    check("gpt2-medium ablation-pathology rate", P, MD, "same pathology in 10/17 pairs (59%)")
    check("stale gpt2 13/17 pathology purged", P, MD, "In 13/17 pairs, gpt2's handoff ablation", expect_absent=True)
    check("stale gpt2-medium 6% pathology purged", P, MD,
          "the same pathology at the same rate (1/17 = 6%)", expect_absent=True)
    # The two "evaluable" single-GEM pairs are an opt-350m stored-series defect
    # (23 metric entries for a 24-layer model), not a segmentation edge case.
    check("single-GEM residual attributed to opt-350m defect", P, MD,
          "opt-350m plurality and opt-350m agency")
    check("bogus segmentation-edge-case hedge purged", P, MD,
          "unreconciled edge case in the segmentation", expect_absent=True)
    # Withdrawn framings must not survive in the body.
    check("no 'depth floor above' after its withdrawal", P, MD,
          "consistent with the depth floor above", expect_absent=True)
    check("gpt2 exemption not called pre-specified", P, MD,
          "documented structured failure for pre-specified reasons", expect_absent=True)

    # --- §8.1 supervised head-to-head, anchored to the pair-aware artifact.
    # Round 2 reported this section as a blocker after diffing it against the
    # SUPERSEDED non-pair-aware run in ~/rosetta_data/results/. The authoritative
    # artifact is the per-model tree published to HF under
    # paper_n250/_p2_direction_estimator/, and every §8.1 figure reproduces from
    # it exactly. These guards are store-derived so the manifest cannot drift
    # from the artifact, and they block the superseded run's values by literal.
    cells = _p2_estimator_cells()
    if cells is None:
        FAILURES.append(
            "[gem] §8.1 supervised baseline: store tree missing at "
            f"{P2_ESTIMATOR_DIR} — restore with: hf download "
            "james-ra-henry/Rosetta-Activations --repo-type dataset --local-dir "
            "~/rosetta_data/ --include 'paper_n250/_p2_direction_estimator/*'"
        )
    else:
        dom = [c["dom_auroc"] for c in cells]
        lr = [c["logreg_auroc"] for c in cells]
        d = [b - a for a, b in zip(dom, lr)]
        hi = [x for a, x in zip(dom, d) if a >= 0.99]
        lo = [x for a, x in zip(dom, d) if a < 0.95]
        check("§8.1 cell count", P, MD, "102 concept × model cells",
              value=f"{len(cells)} concept × model cells")
        check("§8.1 logreg mean AUROC", P, MD, "0.998 for logistic regression",
              value=f"{mean(lr):.3f} for logistic regression")
        check("§8.1 dom mean AUROC", P, MD, "0.965 for difference-of-means",
              value=f"{mean(dom):.3f} for difference-of-means")
        check("§8.1 median gap", P, MD, "the median gap is +0.008",
              value=f"the median gap is +{median(d):.3f}")
        check("§8.1 already-at-0.99 subset", P, MD,
              f"in the {len(hi)} of {len(cells)} cells where the unsupervised estimator already reaches 0.99")
        check("§8.1 already-at-0.99 delta", P, MD, "adds **+0.002**",
              value=f"adds **+{mean(hi):.3f}**")
        check("§8.1 below-0.95 subset", P, MD, f"in the {len(lo)} cells where it falls below 0.95")
        check("§8.1 below-0.95 delta", P, MD, "it adds **+0.119**",
              value=f"it adds **+{mean(lo):.3f}**")
        ceil_lr = sum(1 for x in lr if x == 1.0) / len(lr)
        ceil_dom = sum(1 for x in dom if x == 1.0) / len(dom)
        check("§8.1 ceiling rates", P, MD,
              f"exactly 1.0 in {ceil_lr:.0%} of cells against {ceil_dom:.0%} for difference-of-means")
        wins = sum(1 for x in d if x > 0)
        ties = sum(1 for x in d if x == 0)
        check("§8.1 win/tie count", P, MD, f"({wins}/{len(cells)}, with {ties} exact ties)")
        check("§8.1 pair-aware split asserted in every artifact", P, MD, "The split is pair-aware",
              value="The split is pair-aware" if all(c["_pair_aware"] for c in cells) else "NOT-PAIR-AWARE")
        check("§8.1 same-layer comparison holds in the artifact", P, MD, "**at the same layer**",
              value="**at the same layer**" if all(
                  c["layer"] == c["dom_layer"] == c["logreg_layer"] == c["lda_layer"] for c in cells
              ) else "LAYERS-DIFFER")
        check("§8.1 training-set size", P, MD, "with 400 training examples",
              value=f"with {sorted({c['n_train'] for c in cells})[0]} training examples")
    # The superseded non-pair-aware run's figures must appear nowhere.
    for label, v in [
        ("logreg mean", "0.9945"), ("win rate", "79.4%"), ("ceiling count", "65/102"),
        ("deception dom", "0.9121"), ("deception logreg", "0.9703"),
        ("win/tie split", "84/102, with 19 exact ties"),
    ]:
        check(f"superseded non-pair-aware {label} purged", P, MD, v, expect_absent=True)
    # The erratum said no number from this experiment was citable; the re-run
    # settled that and §8.1 quotes it, so the stale disclaimer must not return.
    check("stale 'not citable until re-run' erratum purged", P, MD,
          "why no number from it is citable in either direction until it is re-run", expect_absent=True)

    # --- External-citation audit (protocol rule 7), 2026-07-29. P2's reference
    # list had never been audited to the depth P1's and P3's were.
    # Verified against arXiv:2303.08112: the tuned lens trains a SEPARATE affine
    # probe per layer ("we train an affine probe for each block"), so describing
    # it as a fixed projection at every layer inverted its central contribution.
    check("tuned lens described as per-layer, not fixed", P, MD,
          "the tuned lens trains a separate affine translator per layer")
    check("'fixed projection at every layer' misreading purged", P, MD,
          "read a *fixed* projection at every layer", expect_absent=True)
    # §5.1's ablation operator h - (h·u)u is Arditi et al. 2024's directional
    # ablation; the paper used it unattributed.
    check("ablation operator attributed to Arditi", P, MD,
          "directional ablation as introduced by Arditi et al. [2024]")
    # SAE features are dictionary features, not circuits (a circuit is a
    # computational subgraph). The foundational SAE work was entirely absent.
    check("SAE work cited", P, MD, "[Cunningham et al., 2023; Bricken et al., 2023]")
    check("Gemma Scope cited", P, MD, "Gemma Scope [Lieberum et al., 2024]")
    check("SAE-features-are-circuits error purged", P, MD,
          "SAE-extracted features represent individual circuits", expect_absent=True)
    check("SAE 'circuit-level features' error purged", P, MD,
          "SAE decomposes it into circuit-level features", expect_absent=True)
    # Reference entries must exist for every newly-cited work.
    for surname in ["Arditi, A., Obeso, O.", "Bricken, T., Templeton, A.",
                    "Cunningham, H., Ewart, A.", "Lieberum, T., Rajamanoharan, S."]:
        check(f"reference entry present: {surname.split(',')[0]}", P, MD, f"- {surname}")

    # --- Round-2 bounded majors, all recomputed from ~/rosetta_data/paper_n250
    # on the 29-model / 493-cell roster (gpt-neo-125m excluded, as the paper does).
    # MAJOR 10: the headline ablation width. All 464 non-exfiltration cells ran
    # at w=1; the 28 w=3 cells are exfiltration for 28 of 29 models.
    # Width + headline are computed from the CURRENT exfiltration artifacts in
    # _p2_exfil_width1/ (w=1, n_pairs=249). The main-tree exfiltration files were
    # suffixed `superseded_*` on 2026-07-30; reading them gives w=3/n=50 and a
    # headline of 343/493, which is what round 2's MAJOR 14 was measuring.
    hl = _p2_headline()
    if hl is None:
        FAILURES.append(
            "[gem] headline/width: corrected exfiltration tree missing at "
            f"{P2_EXFIL_DIR} — restore with: hf download "
            "james-ra-henry/Rosetta-Activations --repo-type dataset --local-dir "
            "~/rosetta_data/ --include 'paper_n250/_p2_exfil_width1/*'"
        )
    else:
        wins, n, widths, near = hl
        check("frozen headline reproduces from the corrected store", P, MD,
              f"{wins}/{n}", value=f"{wins}/{n}")
        check("headline ablation width disclosed", P, MD,
              f"**$w = 1$ in all {n} cells**",
              value=f"**$w = 1$ in all {n} cells**" if widths == {1} else "WIDTHS-NOT-UNIFORM")
        check("near-final rule not credited for the w=1 corpus", P, MD,
              f"only {near} of the {n} have $L_H/N > 0.85$")
    check("superseded exfiltration vintage disclosed", P, MD,
          "An earlier state of the corpus had `exfiltration` alone at $w = 3$")
    check("stale 'default is w = 3' framing purged", P, MD,
          "The default is $w = 3$.", expect_absent=True)
    check("stale 465-of-493 width claim purged", P, MD, "465 of 493 cells", expect_absent=True)

    # --- Minors sweep 2026-07-31. The v1 errata must list every v1 claim this
    # revision withdraws; four were being withdrawn in the body with no entry.
    for label, txt in [
        ("depth-matched confirmation", "**The depth-matched control's confirmation of the settling criterion.**"),
        ("OPT high-EEC profile", "**OPT's high-EEC architectural profile.**"),
        ("Pythia ladder depth trend", "**The Pythia-ladder depth trend.**"),
        ("EEC/depth independence", '**"EEC and handoff depth are independent dimensions" (§9.2).**'),
    ]:
        check(f"errata lists withdrawal: {label}", P, MD, txt)
    # Acronyms the paper never expanded (MHA/GQA carry the cohort split and appear
    # in the abstract; LDA is an §8.1 estimator; RCP was never bound).
    check("MHA expanded on first use", P, MD, "multi-head attention (MHA)")
    check("GQA expanded on first use", P, MD, "grouped-query attention (GQA)")
    check("LDA expanded on first use", P, MD, "linear discriminant analysis (LDA)")
    check("RCP bound to its expansion", P, MD, "Rosetta Concept Pairs (RCP)")
    # Cross-reference into P3: the SAE analysis is its Supplementary §G, not §8.6
    # (P3 numbers main-text sections and letters its supplement).
    check("P3 SAE cross-ref resolves", P, MD, "its Supplementary §G)")
    check("malformed 'Supplementary §8.6' purged", P, MD, "Supplementary §8.6", expect_absent=True)
    # Table 3 became Table D2 in the appendix split.
    check("stale 'Tables 1–3' purged", P, MD, "Tables 1–3", expect_absent=True)
    # MAJOR 5: the atlas-position gradient must use the informative subset.
    check("atlas-position gradient on informative GEMs", P, MD,
          "0.844, 0.863 and 0.905 at successive positions")
    check("informative within-pair spread", P, MD, "mean within-pair spread of **0.106**")
    check("degenerate-inclusive gradient labelled as such", P, MD,
          "0.856 \u2192 0.936 \u2192 0.959 \u2192 0.976, spread 0.192")
    check("spliced gradient purged", P, MD,
          "rising to 0.936, 0.959 and 0.976 at successive GEMs", expect_absent=True)
    # MAJOR 6: pythia-2.8b's seventeen 1.000s are terminal GEMs.
    check("pythia-2.8b 1.000s attributed to terminal GEMs", P, MD,
          "**all seventeen are terminal GEMs**")
    check("pythia-2.8b informative range", P, MD, "0.677 to 0.957 (mean 0.827)")
    # MAJOR 9: the random-window null re-estimates the direction.
    check("random-window null does not hold direction fixed", P, MD,
          "It does not hold the direction fixed")
    check("stale 'holds the direction fixed' purged", P, MD,
          "holds the direction fixed and randomises", expect_absent=True)
    # MAJOR 11: OPT sits below the corpus mean EEC, i.e. rotates more.
    check("OPT high-EEC architectural claim withdrawn", P, MD,
          "The architectural claim is withdrawn")
    check("stale OPT architecture claim purged", P, MD,
          "This is a real property of OPT's architecture and training", expect_absent=True)
    check("temporal_order corrected as below-mean", P, MD, "but temporal_order (0.225) is **below** it")
    # MAJOR 12: pythia-70m meets the falsification criterion with no defence.
    check("pythia-70m falsification conceded", P, MD,
          "on pythia-70m the falsification criterion is met")
    check("pythia-70m has no ablation pathology", P, MD, "0 of 17 cells with retained > 100%")
    # MAJOR 13: 51.7% is roughly half, not "most concepts in most models".
    check("handoff-cosine survival stated as roughly half", P, MD,
          "in **roughly half** of the cases where that question can be asked")
    check("'most concepts in most models' inflation purged (§8.1)", P, MD,
          "it documents that for most concepts in most models", expect_absent=True)
    check("conclusion doubled-conjunction typo fixed", P, MD, "0.285) and and", expect_absent=True)
    # MAJOR 15: the Pythia ladder depth trend is not in the data.
    check("Pythia ladder trend withdrawn", P, MD, "**That trend is not in the data and is withdrawn.**")
    check("Pythia near-final count falls 1B->2.8B", P, MD,
          "from 6 concepts at pythia-1b to 5 at pythia-2.8b")
    check("stale 'most notable jump' purged", P, MD,
          "The most notable jump occurs between 1B and 2.8B", expect_absent=True)
    # MAJOR 17: Table D3 median-z synced to the artifact (3 of 4 were a
    # different vintage; only Phi-2 matched).
    for cohort, z in [("MHA", "1,231.7"), ("GQA", "1,771.3"), ("Gemma", "619.2"), ("Phi-2", "834.0")]:
        check(f"Table D3 median z ({cohort})", P, MD, f"| {z} | 100% |")
    for stale in ["1,317.1", "1,878.3", "622.8"]:
        check(f"stale Table D3 median z purged ({stale})", P, MD, f"| {stale} |", expect_absent=True)
    # MAJOR 4: 9.2 must state its correlations rather than gesture at them.
    check("§9.2 EEC-depth correlation stated", P, MD, "\\rho = -0.752$, $p = 0.0005$")
    check("§9.2 partial correlation reported", P, MD, "\\rho = -0.540$, $p = 0.031$")
    check("§9.2 rotation-rate sign noted", P, MD, "its sign is **positive**")
    check("§9.2 robustness premise scoped to GEM level", P, MD,
          "robust to span at the GEM level, not at the concept level")
    check("§9.2 unsupported no-association framing purged", P, MD,
          "make no claim about their association", expect_absent=True)
    # 5.6 is a six-model subset, not the 29-model corpus.
    check("§5.6 scoped to six models in §5 preamble", P, MD,
          "§5.6's random-window null runs on a **six-model subset**")
    check("§5.6 miscounted into the 29-model corpus (preamble)", P, MD,
          "the §5.3/§5.5/§5.6 controls run on the **29-model, 493-pair handoff corpus**",
          expect_absent=True)
    check("§5.6 miscounted into the 29-model corpus (Appendix A)", P, MD,
          "the §5.3/§5.5/§5.6 controls use the full 29-model corpus", expect_absent=True)

    # --- BLOCKER 2: the "depth-matched" control is matched to the SHALLOWEST of
    # ~2.57 ablation sites. Recomputed on the 380 retained pairs: control mean
    # relative depth 0.363, treatment deepest site mean 0.958, all-site mean
    # 0.639. So the arms are not depth-equal and +41.6pp cannot isolate depth.
    check("§5.5 names the shallowest-GEM convention", P, MD,
          "`ablation_targets[0]` — the **shallowest** GEM's handoff layer")
    check("§5.5 states the depth asymmetry", P, MD,
          "The arms therefore differed in how many layers were ablated *and* in how deep the ablation reached")
    check("§5.5 gives the treatment-arm depth", P, MD, "**0.958** at the deepest")
    check("§9.3 no longer calls the control a cleaner isolation", P, MD,
          "provides a cleaner isolation", expect_absent=True)
    check("'comparable depth' claim purged", P, MD, "comparable depth", expect_absent=True)

    # --- The site-matched control (GPU run 2026-07-30, flows abfac284 + 2fb00186).
    # This is the experiment §5.5 named as missing; it collapses the single-site
    # +41.9pp to +3.6pp. Values recomputed from the per-model artifacts, not from
    # the run's own aggregate.json.
    sm = _p2_site_matched()
    if sm is None:
        FAILURES.append(
            "[gem] §5.5 site-matched control: store tree missing at "
            f"{P2_SITE_MATCHED_DIR} — restore with: hf download "
            "james-ra-henry/Rosetta-Activations --repo-type dataset --local-dir "
            "~/rosetta_data/ --include 'paper_n250/_gem_depth_matched/*'"
        )
    else:
        allw = sum(1 for r in sm if r["win"])
        check("§5.5 site-matched corpus rate", P, MD,
              f"{allw}/{len(sm)} pairs ({100*allw/len(sm):.1f}%",
              value=f"{allw}/{len(sm)} pairs ({100*allw/len(sm):.1f}%")
        EXCL = {"gpt2", "gpt2-medium", "gemma-2-2b", "gemma-2-9b"}
        sub = [r for r in sm if r["model"] not in EXCL]
        subw = sum(1 for r in sub if r["win"])
        check("§5.5 site-matched interpretable-subset rate", P, MD,
              f"**{subw}/{len(sub)} ({100*subw/len(sub):.1f}%)",
              value=f"**{subw}/{len(sub)} ({100*subw/len(sub):.1f}%)")
        check("§5.5 legacy single-site rate resynced", P, MD, "361/382 (94.5%)")
        check("§5.5 legacy mean resynced", P, MD, "+41.9pp")
    check("§5.5 withdraws the large-advantage claim", P, MD,
          "**withdraw the claim that ablating at the handoff layer suppresses separation "
          "substantially more than ablating at a comparable-depth alternative.**")
    check("contribution 7 states the smaller effect", P, MD,
          "**A much smaller intervention effect than previously reported**")
    check("abstract carries the site-matched result", P, MD,
          "cuts the intervention advantage by an order of magnitude")
    check("D.5 rewritten to the site-matched control", P, MD,
          "under the **site-matched** control (one depth-matched control layer per GEM")
    # The superseded single-site framing must not return anywhere.
    check("interim single-site values survive only as §5.5 provenance", P, MD,
          "the 360/382 (94.2%) / +41.6pp reported in the interim moves by one cell")
    check("stale D.5 'strong performers' framing purged", P, MD,
          "Strong performers with 100% GEM wins include", expect_absent=True)
    check("stale §5.5 4/12 gemma cross-ref purged", P, MD, "§5.5's 4/12", expect_absent=True)

    # --- Mistral SWA error, routed in by the P3 reviewer 2026-07-30 and verified
    # against the HF configs: Mistral-7B-v0.3 has sliding_window=None (removed at
    # v0.2); v0.1 had 4096; gemma-2-2b has 4096. v1's "SWA exceptions" grouping
    # was therefore half false, and it was the premise of §5.5's explanation.
    check("Mistral SWA premise corrected", P, MD, "v0.3's configuration sets `sliding_window: null`")
    check("Mistral exception reported as unexplained", P, MD, "We have no replacement account")
    check("SWA grouping withdrawn in errata", P, MD,
          "**Mistral-7B-v0.3 does not use sliding window attention**")
    check("false uniform-SWA claim purged", P, MD,
          "Mistral-7B-v0.3 applies uniform sliding window attention", expect_absent=True)
    check("SWA-exceptions grouping purged from body", P, MD,
          "Both use non-standard sliding window attention", expect_absent=True)


# ---------------------------------------------------------------------------
# P1 (caz-framework)
# ---------------------------------------------------------------------------
def manifest_p1():
    P, MD = "caz-framework", ["preprint.md"]
    check("quotes P2's frozen headline", P, MD, "341/493")
    check("stale GEM headline purged", P, MD, "350/493", expect_absent=True)
    check("stale GEM W purged", P, MD, "W=79", expect_absent=True)
    check("withdrawn scale-floor framing purged", P, MD, "then plateaus around", expect_absent=True)
    check("P3 gentle-CAZ claim phrased as P3 states it", P, MD,
          "versus 28% of non-CAZ layers")
    check("fused gentle-CAZ phrasing purged", P, MD,
          "exceed the non-CAZ ablation baseline", expect_absent=True)
    check("figure caption smoothing corrected", P, MD, "$w=1$ velocity smoothing", expect_absent=True)
    # Companion-sourced numbers re-verified against P2/P3 current text (2026-07-29 P1-reviewer
    # handoff sweep). Presence guards against silent drift; correctness is anchored by the
    # companion's own manifest (P3 checks 0.404 against C5 JSON; P2 checks its headline).
    check("§4.4 P2 Gemma-excluded rate (disambiguates 71.0% from headline)", P, MD, "326/459")
    check("§5.2 companion τ (P3, supersedes at C=17)", P, MD, "0.404")
    check("§5.8 companion width ρ (P3)", P, MD, "0.307")
    check("§7 companion depth-shift |Δ| synced to P3", P, MD, "9.4 pp")
    check("§7 stale pre-sync |Δ| purged (M31)", P, MD, "10.6 pp", expect_absent=True)
    for v in ["0.9750", "Δ=+0.195", "1,666/1,666", "[Henry, 2026d]"]:
        check(f"P4 purge: {v}", P, MD, v, expect_absent=True)

    # --- P1-AUTHORED numbers (round 2). Until now every p1 check was a
    # companion-sourced or absence guard, and that gap is exactly where round 2
    # found the Figure 1 misattribution and the 3.0-vs-3.14 drift.
    # §6 worked example, GPT-2-XL credibility, verified vs paper_n250.
    check("§6 GPT-2-XL credibility peak S", P, MD, "S = 1.17")
    check("§6 GPT-2-XL layer 45 (94% depth)", P, MD, "layer 45 (94% depth)")
    check("§6 GPT-2-XL dominant region score", P, MD, "score=0.84")
    check("§6 Pythia-160M mean CAZes", P, MD, "mean of 3.1 CAZes")
    check("§6 stale Pythia-160M mean purged", P, MD, "mean of 3.0 CAZes", expect_absent=True)
    # Figure 1: per-CAZ suppression, replacing the two-site final-layer number
    # that §7(6) classifies as having no non-tautological content.
    check("Fig 1 per-CAZ suppression (shallow)", P, MD, "46.5% of separation at the shallow CAZ")
    check("Fig 1 per-CAZ suppression (deep)", P, MD, "77.5% at the deep one")
    check("Fig 1 tautological 74.5% purged", P, MD, "reduces separation by 74.5%", expect_absent=True)
    # Claims a companion formally withdrew must not be asserted here.
    check("no Delta/Windowed PCA superiority claim", P, MD,
          "give no further gain over this single-layer choice", expect_absent=True)
    check("no 'provide no further gain' PCA claim", P, MD,
          "but provide no further gain on downstream ablation tasks", expect_absent=True)
    # Gemma-2: P3 §6.9 retracts the causal-inertness reading.
    check("no Gemma causal-inertness reading", P, MD, "its CAZes are not demonstrably causal", expect_absent=True)
    check("no 'inert peaks' caption", P, MD, "high-score-but-inert peaks", expect_absent=True)
    # External-literature corrections (round 2, verified against primary sources).
    check("Engels dark matter stated correctly", P, MD, "more than 90% of its norm, is *linearly predictable*")
    check("Engels 50%-resists misreading purged", P, MD,
          "that resists linear decomposition — accounts for roughly 50%", expect_absent=True)
    check("Arditi capability cost stated correctly", P, MD, "minimal effect on other capabilities")
    check("Arditi collateral-effect misattribution purged", P, MD,
          "not without collateral effect on general capabilities", expect_absent=True)
    check("Gurnee manifolds in early layers", P, MD, "curved manifolds in early layers")
    check("Gurnee 'middle layers' purged", P, MD, "curved manifolds in middle layers", expect_absent=True)
    # Tenney et al. 2019 runs no interventions and makes no salience-vs-causality
    # claim (verified against the source, 2026-07-30 P1 reference audit); the
    # syntactic-to-semantic gradient it does show is retained as Prediction 2's basis.
    check("Tenney salience-vs-causality overclaim purged", P, MD,
          "anticipate the salience-vs-causality distinction", expect_absent=True)
    # Corpus attribution for the tiling verification (493 is P2's corpus).
    check("tiling verification attributed to GEM corpus", P, MD,
          "verified across the companion GEM corpus")
    check("no bare 'in this corpus' for 493", P, MD,
          "493 concept × model pairs in this corpus", expect_absent=True)


MANIFESTS = {"p1": manifest_p1, "p2": manifest_p2, "p3": manifest_p3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=sorted(MANIFESTS), action="append")
    a = ap.parse_args()
    for k in a.paper or sorted(MANIFESTS):
        MANIFESTS[k]()
    print(f"number-provenance verifier: {PASSES} passed, {len(FAILURES)} failed\n")
    for f in FAILURES:
        print("FAIL " + f)
    if FAILURES:
        print("\nA failure means manuscript prose and its artifact disagree, or a "
              "superseded value survived. Fix the prose (or, if the artifact is "
              "the stale side, say so explicitly in the paper's ledger).")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
