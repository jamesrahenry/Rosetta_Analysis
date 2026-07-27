#!/usr/bin/env bash
# Full-coverage P4 Phase-A null sweep — tesseract (FX-8350) half.
# Takes clusters B (d=2048, 36 pairs) and C (d=4096, 26 pairs): the two most
# expensive, ~10.6 of the ~15.4 remaining single-host hours. The dev laptop
# (i7-4600U) runs G/H/A/E/D + F in parallel via run_full_nullfloor.sh.
#
# All 17 concepts, ALL cross-family same-dim pairs, no subsampling.
set -u
cd "$(dirname "$0")"

ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache
export P4_REGEN_STAGE=$ROOT/stage
export P4_PEAK_CACHE=$ROOT/peakcache
PY=$ROOT/venv/bin/python

LOG=$ROOT/tess_sweep_chain.log

run () {   # run <mode> <cluster> <K> <outdir>
  local mode=$1 cl=$2 K=$3 out=$4
  echo "START $mode $cl $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode "$mode" --clusters "$cl" --K "$K" \
      --out "$ROOT/$out" > "$ROOT/${out}.log" 2>&1
  echo "DONE  $mode $cl exit=$? $(date -u +%FT%TZ)" >> "$LOG"
}

run spectrum C 8 nullfloor_full_C_floor
run spectrum B 8 nullfloor_full_B_floor
run scramble C 8 nullfloor_full_C_scr
run scramble B 8 nullfloor_full_B_scr

echo "ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
