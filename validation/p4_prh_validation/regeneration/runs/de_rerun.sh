#!/usr/bin/env bash
set -u
ROOT=/storage/JamesData/p4_nullfloor
cd "$ROOT"
export HF_HOME=$ROOT/cache P4_REGEN_STAGE=$ROOT/stage P4_PEAK_CACHE=$ROOT/peakcache
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=$ROOT/venv/bin/python
CHAIN=$ROOT/de_rerun_chain.log
run(){ local mode=$1 cl=$2 out=$3
  echo "START $mode $cl $(date -u +%FT%TZ)" >> "$CHAIN"
  "$PY" nullfloor_analysis.py --mode "$mode" --clusters "$cl" --K 8 --out "$ROOT/$out" > "$ROOT/${out}.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] && touch "$ROOT/$out/.done"
  echo "DONE  $mode $cl exit=$rc $(date -u +%FT%TZ)" >> "$CHAIN"; }
echo "DE_RERUN START $(date -u +%FT%TZ) (guard per-pair fix)" >> "$CHAIN"
run spectrum D nullfloor_full_D_floor
run spectrum E nullfloor_full_E_floor
run scramble D nullfloor_full_D_scr
run scramble E nullfloor_full_E_scr
echo "DE_RERUN ALL DONE $(date -u +%FT%TZ)" >> "$CHAIN"
