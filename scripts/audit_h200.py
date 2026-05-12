#!/usr/bin/env python3
"""
audit_h200.py — Pre-sync data completeness audit.

Checks that every expected file is present and at the correct N on the
current machine (run on H200 before syncing, or dev machine after).

Usage:
    cd ~/rosetta_analysis
    uv run python scripts/audit_h200.py
    uv run python scripts/audit_h200.py --verbose   # show per-model detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: resolve rosetta_tools regardless of machine
# ---------------------------------------------------------------------------
import importlib
if importlib.util.find_spec("rosetta_tools") is None:
    import sys as _sys
    for _p in [Path.home() / "rosetta_tools",
               Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"]:
        if _p.exists():
            _sys.path.insert(0, str(_p))
            break

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.extract import (
    PRH_PROXY_MODELS, P3_MODELS, P3_INSTRUCT_MODELS, PRH_CLUSTER_H,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_17_CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# P3 ablation uses 7 core concepts
P3_ABL_CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

N_EXPECTED = 250
PAPER_OUT  = ROSETTA_RESULTS / "PRH"

PASS = "✓"
WARN = "⚠"
FAIL = "✗"

issues: list[tuple[str, str]] = []   # (severity, message)


def slug(mid: str) -> str:
    return mid.replace("/", "_").replace("-", "_").replace(".", "_")


def check_model_caz(mid: str, concepts: list[str], n_expected: int,
                    verbose: bool) -> list[str]:
    """Return list of failure strings for this model's CAZ files."""
    d = ROSETTA_MODELS / slug(mid)
    failures = []
    for c in concepts:
        f = d / f"caz_{c}.json"
        if not f.exists():
            failures.append(f"{c}: MISSING")
            continue
        try:
            n = json.loads(f.read_text()).get("n_pairs", 0)
        except Exception:
            failures.append(f"{c}: UNREADABLE")
            continue
        if n < n_expected:
            failures.append(f"{c}: n={n} < {n_expected}")
    return failures


def check_model_ablation(mid: str, concepts: list[str],
                          verbose: bool) -> list[str]:
    """Return list of failure strings for this model's ablation files."""
    d = ROSETTA_MODELS / slug(mid)
    failures = []
    for c in concepts:
        f = d / f"ablation_{c}.json"
        if not f.exists():
            failures.append(f"{c}: MISSING")
    return failures


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(1, 60 - len(title))}")


def row(status: str, label: str, detail: str = "") -> None:
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if status == FAIL:
        issues.append(("FAIL", f"{label} {detail}"))
    elif status == WARN:
        issues.append(("WARN", f"{label} {detail}"))


def audit_model_set(title: str, models: list[str], concepts: list[str],
                    n_exp: int, check_ablation: bool,
                    abl_concepts: list[str], verbose: bool) -> None:
    section(title)
    total_ok = total_fail = 0
    for mid in sorted(models):
        caz_fails = check_model_caz(mid, concepts, n_exp, verbose)
        abl_fails = check_model_ablation(mid, abl_concepts, verbose) if check_ablation else []
        short = mid.split("/")[-1]
        if not caz_fails and not abl_fails:
            total_ok += 1
            if verbose:
                row(PASS, short)
        else:
            total_fail += 1
            detail = "; ".join(caz_fails + [f"abl:{a}" for a in abl_fails])
            row(FAIL, short, detail)
    row(PASS if total_fail == 0 else FAIL,
        f"{total_ok}/{total_ok + total_fail} models complete")


def audit_results_file(label: str, path: Path, warn_only: bool = False) -> None:
    if path.exists():
        row(PASS, label)
    else:
        row(WARN if warn_only else FAIL, label, f"missing: {path}")


# ===========================================================================
# Main audit
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\nRosetta data audit — {Path.home() / 'rosetta_data'}")
    print(f"Checking models in: {ROSETTA_MODELS}")

    # ── 1. PRH proxy corpus (19 models, 17 concepts, N=250) ─────────────────
    audit_model_set(
        f"1. PRH proxy corpus ({len(PRH_PROXY_MODELS)} models × 17 concepts, N={N_EXPECTED})",
        PRH_PROXY_MODELS,
        ALL_17_CONCEPTS,
        n_exp=N_EXPECTED,
        check_ablation=False,
        abl_concepts=[],
        verbose=args.verbose,
    )

    # ── 2. P3 base corpus (26 models, 17 concepts + ablation) ───────────────
    audit_model_set(
        f"2. P3 base corpus ({len(P3_MODELS)} models × 17 concepts + ablation, N={N_EXPECTED})",
        P3_MODELS,
        ALL_17_CONCEPTS,
        n_exp=N_EXPECTED,
        check_ablation=True,
        abl_concepts=P3_ABL_CONCEPTS,
        verbose=args.verbose,
    )

    # ── 3. P3 instruct corpus (9 models, 17 concepts, N=250) ────────────────
    audit_model_set(
        f"3. P3 instruct corpus ({len(P3_INSTRUCT_MODELS)} models × 17 concepts, N={N_EXPECTED})",
        P3_INSTRUCT_MODELS,
        ALL_17_CONCEPTS,
        n_exp=N_EXPECTED,
        check_ablation=False,
        abl_concepts=[],
        verbose=args.verbose,
    )

    # ── 4. Cluster H — Gemma 4 MoE (new) ────────────────────────────────────
    audit_model_set(
        f"4. Cluster H — Gemma 4 MoE ({len(PRH_CLUSTER_H)} models × 17 concepts, N={N_EXPECTED})",
        PRH_CLUSTER_H,
        ALL_17_CONCEPTS,
        n_exp=N_EXPECTED,
        check_ablation=False,
        abl_concepts=[],
        verbose=args.verbose,
    )

    # ── 5. PRH results files ─────────────────────────────────────────────────
    section("5. PRH results files")
    audit_results_file("prh_main.csv or alignment N=250",
                       PAPER_OUT / "prh_main.csv", warn_only=True)
    audit_results_file("alignment_results_samedim_n250.csv",
                       PAPER_OUT / "alignment_results_samedim_n250.csv", warn_only=True)
    audit_results_file("prh_random_calib_null.json",
                       PAPER_OUT / "prh_random_calib_null.json")
    audit_results_file("provenance.json",
                       PAPER_OUT / "provenance.json")

    # ── 6. P5 results files ──────────────────────────────────────────────────
    section("6. P5 results files")
    p5 = PAPER_OUT / "p5"
    audit_results_file("p5_validation_battery.json",      p5 / "p5_validation_battery.json")
    audit_results_file("p5_propdepth_samedim_results.json", p5 / "p5_propdepth_samedim_results.json",
                       warn_only=True)
    audit_results_file("p5_cka_real_results.json",        p5 / "p5_cka_real_results.json",
                       warn_only=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    fails = [m for s, m in issues if s == "FAIL"]
    warns = [m for s, m in issues if s == "WARN"]
    if not issues:
        print("ALL CHECKS PASSED — safe to release H200.")
    else:
        if fails:
            print(f"FAILURES ({len(fails)}) — resolve before releasing H200:")
            for m in fails:
                print(f"  {FAIL}  {m}")
        if warns:
            print(f"\nWARNINGS ({len(warns)}) — review:")
            for m in warns:
                print(f"  {WARN}  {m}")
    print()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
