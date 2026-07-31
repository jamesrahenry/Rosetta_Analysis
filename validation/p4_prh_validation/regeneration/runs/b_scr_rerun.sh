#!/bin/bash
cd /storage/JamesData/p4_nullfloor
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
PY=/storage/JamesData/p4_nullfloor/venv/bin/python
LOG=b_scr_rerun.log
echo "B_SCR_RERUN START $(date -u +%FT%TZ) (solo, 8 threads)" >> "$LOG"
$PY nullfloor_analysis.py --mode scramble --clusters B --K 8 --out nullfloor_full_B_scr >> "$LOG" 2>&1
rc=$?
echo "B_SCR_RERUN DONE exit=$rc $(date -u +%FT%TZ)" >> "$LOG"
touch b_scr_rerun.done
