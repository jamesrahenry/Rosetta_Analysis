#!/usr/bin/env python3
"""Reconstruct the corrected 249-pair exfiltration set (EXFILTRATION_RERUN_SPEC §1/§1a).

Recovers the exact original 250-pair recorded draw (the one every other
concept's paper_n250 extraction is dated/documented against) from a stored
`calibration_exfiltration_meta.json` manifest, takes those pairs' TEXTS from
the pre-correction RCP revision (b5fa231), overlays the corrected pos/neg
labels from the current RCP revision (a088c29) by (composite pair id, exact
text) match, and drops the pair a088c29 deleted. Emits:

  exfiltration_consensus_pairs.jsonl   -- the 498-record (249 pos + 249 neg)
      REPLACEMENT pool for the runner's RCP checkout. Pool == draw size, so
      `load_concept_pairs(n=249)` takes everything and the seed is irrelevant.
  exfiltration_flip_list.json          -- sidecar: per-pair flipped bool +
      the full audit trail (draw manifest source, RCP revisions, counts).

Expected accounting against the spec (verified twice independently before
this script existed -- t50c6362 and P3's own check, same numbers): of the
250 drawn pairs, 179 flipped, 70 unchanged, 1 deleted => N=249. The script
REFUSES to write output if the observed breakdown differs.

Run on the dev machine (needs the RCP git checkout + HF access, no GPU):

    python build_corrected_exfiltration_pairs.py \
        --rcp ~/Games2/Eigan/Rosetta_Concept_Pairs --out <dir>

Written: 2026-07-17 02:10 UTC by claude:exfil-rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OLD_REV = "b5fa231"   # pre-correction: texts + the recorded draw's era
NEW_REV = "a088c29"   # corrected labels + 7-pair deletion ("clean up data")
RCP_RELPATH = "pairs/raw/v1/exfiltration_consensus_pairs.jsonl"

# The recorded draw lives in every clusters-A-E model's meta, all identical
# (verified across pythia-70m/gpt2/Qwen2.5-14B/gemma-2-9b/opt-350m
# 2026-07-17; only Cluster F differs -- re-extracted 2026-07-16 with the new
# sampler, superseded by this rerun anyway). pythia-70m is the reference.
MANIFEST_REPO = "james-ra-henry/Rosetta-Activations"
MANIFEST_FILE = "paper_n250/EleutherAI_pythia_70m/calibration_exfiltration_meta.json"

EXPECTED = {"flipped": 179, "unchanged": 70, "deleted": 1}


def composite(rec: dict) -> str:
    return f"{rec['pair_id']}__{rec['model_name']}"


def load_rev(rcp: Path, rev: str) -> list[dict]:
    raw = subprocess.run(
        ["git", "-C", str(rcp), "show", f"{rev}:{RCP_RELPATH}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rcp", type=Path, required=True,
                    help="path to a Rosetta_Concept_Pairs git checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-json", type=Path, default=None,
                    help="local meta JSON to use instead of downloading")
    args = ap.parse_args()

    # 1. Recorded draw (250 composite ids, order preserved for the record)
    if args.manifest_json:
        meta = json.loads(args.manifest_json.read_text())
    else:
        from huggingface_hub import hf_hub_download
        meta = json.loads(Path(hf_hub_download(
            MANIFEST_REPO, MANIFEST_FILE, repo_type="dataset")).read_text())
    draw = meta["corpus"]["pair_ids"]
    assert len(draw) == 250 and len(set(draw)) == 250, "draw manifest malformed"

    # 2. Old revision: texts (and old labels, for flip detection)
    old_by_comp: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in load_rev(args.rcp, OLD_REV):
        old_by_comp[composite(rec)][int(rec["label"])] = rec

    # 3. New revision: corrected label per (composite, exact text)
    new_label: dict[tuple[str, str], int] = {}
    new_comps: set[str] = set()
    for rec in load_rev(args.rcp, NEW_REV):
        new_label[(composite(rec), rec["text"])] = int(rec["label"])
        new_comps.add(composite(rec))

    out_records: list[dict] = []
    flip_list: dict[str, bool] = {}
    counts = {"flipped": 0, "unchanged": 0, "deleted": 0}
    problems: list[str] = []

    for comp in draw:
        pair = old_by_comp.get(comp)
        if not pair or set(pair) != {0, 1}:
            problems.append(f"{comp}: not a clean pos/neg pair at {OLD_REV}")
            continue
        if comp not in new_comps:
            counts["deleted"] += 1
            flip_list[comp] = None  # deleted, not flipped
            continue
        corrected = {}
        for old_lab, rec in sorted(pair.items(), reverse=True):
            key = (comp, rec["text"])
            if key not in new_label:
                problems.append(
                    f"{comp}: text (old label {old_lab}) has no exact match at "
                    f"{NEW_REV} -- text was edited, not just relabeled; refusing "
                    "to guess (spec §1a: flag back, don't work around)")
                break
            corrected[old_lab] = new_label[key]
        else:
            if set(corrected.values()) != {0, 1}:
                problems.append(f"{comp}: corrected labels are {corrected} -- "
                                "pair no longer has one pos + one neg")
                continue
            flipped = corrected[1] == 0
            flip_list[comp] = flipped
            counts["flipped" if flipped else "unchanged"] += 1
            for old_lab in (1, 0):
                rec = dict(pair[old_lab])
                rec["label"] = corrected[old_lab]
                out_records.append(rec)

    print(f"breakdown: {counts} (expected {EXPECTED})")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S) -- refusing to write output:")
        for p in problems[:20]:
            print(f"  - {p}")
        sys.exit(1)
    if counts != EXPECTED:
        print("\nbreakdown does NOT match the twice-verified expected "
              "179/70/1 -- refusing to write output. Something changed "
              "upstream; re-derive by hand before trusting this script.")
        sys.exit(1)

    n_pairs = counts["flipped"] + counts["unchanged"]
    assert len(out_records) == 2 * n_pairs == 498

    # Emit pos record before neg record per pair (matches RCP file layout),
    # pair order = recorded draw order (deterministic, auditable).
    ordered: list[dict] = []
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for rec in out_records:
        by_comp[composite(rec)].append(rec)
    for comp in draw:
        recs = by_comp.get(comp)
        if recs:
            ordered.extend(sorted(recs, key=lambda r: -r["label"]))

    args.out.mkdir(parents=True, exist_ok=True)
    pool = args.out / "exfiltration_consensus_pairs.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in ordered))
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()

    sidecar = args.out / "exfiltration_flip_list.json"
    sidecar.write_text(json.dumps({
        "spec": "EXFILTRATION_RERUN_SPEC.md §1/§1a",
        "old_rev": OLD_REV, "new_rev": NEW_REV,
        "draw_manifest": MANIFEST_FILE, "n_pairs": n_pairs,
        "counts": counts, "pool_sha256": digest,
        "flips": flip_list,
    }, indent=1))

    print(f"wrote {pool} ({len(ordered)} records, {n_pairs} pairs)")
    print(f"wrote {sidecar}")
    print(f"pool sha256: {digest}")


if __name__ == "__main__":
    main()
