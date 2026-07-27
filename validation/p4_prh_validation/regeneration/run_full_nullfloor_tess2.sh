#!/usr/bin/env bash
# Full-coverage P4 Phase-A null sweep — ALL remaining clusters, tesseract only.
# The dev laptop (i7-4600U, 2 physical cores) was released for other work.
#
# Every cluster, all 17 concepts, ALL cross-family same-dim pairs. K=8 to match
# the completed F runs — cross-cluster comparability is the whole point, so the
# surrogate-draw count is held fixed rather than tuned per cluster.
#
# Already complete (do not repeat):
#   F floor    17 concepts (5 + 12 pooled) -> 0.4635, real 0.9784, margin 0.5150
#   F scramble 17 concepts x 6 pairs       -> true 0.9784 / scramble 0.9892
#
# Waits for the in-flight C floor to finish, then runs cheap -> expensive so
# results land early.
set -u
cd "$(dirname "$0")"

ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache
export P4_REGEN_STAGE=$ROOT/stage
export P4_PEAK_CACHE=$ROOT/peakcache
PY=$ROOT/venv/bin/python
LOG=$ROOT/tess_sweep_chain.log

# Do not start until the already-running C floor process exits.
while pgrep -f "out /storage/JamesData/p4_nullfloor/nullfloor_full_C_floor" >/dev/null 2>&1; do
  sleep 30
done
echo "TESS2 START $(date -u +%FT%TZ)" >> "$LOG"

run () {   # run <mode> <cluster> <K> <outdir> [extra args...]
  local mode=$1 cl=$2 K=$3 out=$4; shift 4
  if [ -f "$ROOT/$out/.done" ]; then
    echo "SKIP  $mode $cl (already done)" >> "$LOG"; return
  fi
  echo "START $mode $cl $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode "$mode" --clusters "$cl" --K "$K" \
      --out "$ROOT/$out" "$@" > "$ROOT/${out}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && touch "$ROOT/$out/.done"
  echo "DONE  $mode $cl exit=$rc $(date -u +%FT%TZ)" >> "$LOG"
}

# --- floors, cheap -> expensive -------------------------------------------
run spectrum G 8 nullfloor_full_G_floor
run spectrum A 8 nullfloor_full_A_floor
run spectrum H 8 nullfloor_full_H_floor
run spectrum E 8 nullfloor_full_E_floor
run spectrum D 8 nullfloor_full_D_floor
# F: the 5 concepts that predate raw-sample retention, so all 17 pool exactly.
run spectrum F 8 nullfloor_full_F_floor5raw \
    --concepts exfiltration,agency,authorization,causation,certainty
run spectrum B 8 nullfloor_full_B_floor

# --- within-class scrambles (existing A-E scramble covered only 6 of 17) ---
run scramble G 8 nullfloor_full_G_scr
run scramble A 8 nullfloor_full_A_scr
run scramble H 8 nullfloor_full_H_scr
run scramble E 8 nullfloor_full_E_scr
run scramble D 8 nullfloor_full_D_scr
run scramble C 8 nullfloor_full_C_scr
run scramble B 8 nullfloor_full_B_scr

echo "TESS2 ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
