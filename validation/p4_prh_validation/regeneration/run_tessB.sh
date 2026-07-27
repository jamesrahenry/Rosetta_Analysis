#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache
export P4_REGEN_STAGE=$ROOT/stage
export P4_PEAK_CACHE=$ROOT/peakcache
# BLAS thread cap: the FX-8350 has 8 integer cores but only 4 shared FPU
# modules. Unset (8 threads) an aligned_cosine at d=2048 takes 16.6s; at 4
# threads it takes 3.3s -- a 5x difference. Two 4-thread streams therefore use
# the box far better than one 8-thread job.
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
PY=$ROOT/venv/bin/python
LOG=$ROOT/tess_sweep_chain.log

run () {   # run <mode> <cluster> <K> <outdir> [extra args...]
  local mode=$1 cl=$2 K=$3 out=$4; shift 4
  if [ -f "$ROOT/$out/.done" ]; then
    echo "SKIP  $mode $cl ($out)" >> "$LOG"; return
  fi
  echo "START $mode $cl $out $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode "$mode" --clusters "$cl" --K "$K" \
      --out "$ROOT/$out" "$@" > "$ROOT/${out}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && mkdir -p "$ROOT/$out" && touch "$ROOT/$out/.done"
  echo "DONE  $mode $cl $out exit=$rc $(date -u +%FT%TZ)" >> "$LOG"
}

# Stream B -- the small clusters, plus F's 5 concepts that predate raw-sample
# retention (so all 17 F concepts pool exactly).
run spectrum G 8 nullfloor_full_G_floor
run spectrum A 8 nullfloor_full_A_floor
run spectrum H 8 nullfloor_full_H_floor
run spectrum E 8 nullfloor_full_E_floor
run spectrum D 8 nullfloor_full_D_floor
run spectrum F 8 nullfloor_full_F_floor5raw --concepts exfiltration,agency,authorization,causation,certainty
run scramble G 8 nullfloor_full_G_scr
run scramble A 8 nullfloor_full_A_scr
run scramble H 8 nullfloor_full_H_scr
run scramble E 8 nullfloor_full_E_scr
run scramble D 8 nullfloor_full_D_scr
echo "STREAM_B ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
