#!/bin/bash
# Cluster-A full-coverage rerun: original run silently dropped 10/12 plurality
# cells to transient HF load failures (rate-limit era, cache now primed).
set -u
cd /storage/JamesData/p4_nullfloor
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache
export P4_REGEN_STAGE=$ROOT/stage
export P4_PEAK_CACHE=$ROOT/peakcache
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=$ROOT/venv/bin/python
LOG=$ROOT/a_plurality_rerun.log
echo "A_RERUN START $(date -u +%FT%TZ)" >> "$LOG"
for mode in spectrum scramble; do
  echo "START $mode A $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode $mode --clusters A --K 8 \
      --out $ROOT/nullfloor_full_A_${mode}_r2 >> "$LOG" 2>&1
  echo "DONE  $mode A exit=$? $(date -u +%FT%TZ)" >> "$LOG"
done
echo "A_RERUN ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
touch a_plurality_rerun.done
