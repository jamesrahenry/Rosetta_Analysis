#!/usr/bin/env bash
# Full-coverage P4 Phase-A null sweep — FROZEN_REVIEW_ROUND1 P4-B1/B3.
#
# Every cluster, every one of the 17 concepts, every cross-family same-dim pair.
# No --max-pairs, no concept subsetting. Supersedes the earlier bounded runs,
# which showed ~0.08 of concept-sampling error on cluster F (5-concept floor
# 0.539 vs 17-concept 0.4635) and so cannot be quoted.
#
# Sequential by design: the box is 4 cores and BLAS already saturates them.
# ~16 h total. CPU only — no GPU anywhere in this path.
set -u
cd "$(dirname "$0")"

export P4_REGEN_STAGE=/home/jhenry/Games2/_p4_stage
export HF_HOME=/home/jhenry/Games2/.hf_cache
export P4_PEAK_CACHE=/home/jhenry/Games2/_p4_peakcache

LOG=full_sweep_chain.log

# Wait out the in-flight F scramble (17 concepts x all 6 pairs) before starting.
until grep -q "SCRAMBLE exit" nullfloor_F_chain.log 2>/dev/null; do sleep 30; done

run () {   # run <mode> <cluster> <K> <outdir> [extra args...]
  local mode=$1 cl=$2 K=$3 out=$4; shift 4
  echo "START $mode $cl $(date -u +%FT%TZ)" >> "$LOG"
  python nullfloor_analysis.py --mode "$mode" --clusters "$cl" --K "$K" \
      --out "$out" "$@" > "${out}.log" 2>&1
  echo "DONE  $mode $cl exit=$? $(date -u +%FT%TZ)" >> "$LOG"
}

# Floors. Cheap -> expensive so coverage lands early. G/H/D have never had a
# measured floor at all; A and B were only ever reviewer/n-sweep estimates.
# B and C are split off to tesseract (FX-8350) — see run_full_nullfloor_tess.sh.
for cl in G H A E D; do
  run spectrum "$cl" 8 "nullfloor_full_${cl}_floor"
done

# F floor is already complete for 12 concepts; re-run the original 5 so that all
# 17 pool exactly (those predate raw-sample retention).
run spectrum F 8 nullfloor_full_F_floor5raw \
    --concepts exfiltration,agency,authorization,causation,certainty

# Within-class scrambles. The existing A-E scramble covered 6 of 17 concepts.
for cl in G H A E D; do
  run scramble "$cl" 8 "nullfloor_full_${cl}_scr"
done

echo "ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
