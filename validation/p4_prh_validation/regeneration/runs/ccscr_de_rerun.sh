#!/bin/bash
cd /storage/JamesData/p4_nullfloor
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=/storage/JamesData/p4_nullfloor/venv/bin/python
LOG=ccscr_de_rerun.log
echo "CCSCR_DE_RERUN START $(date -u +%FT%TZ) : D E (guarded 0fd1efe)" >> "$LOG"
for c in D E; do
  echo "START $c $(date -u +%FT%TZ)" >> "$LOG"
  $PY crossconcept_scramble.py --cluster $c --K 8 --out ccscr_$c >> "$LOG" 2>&1
  echo "DONE  $c exit=$? $(date -u +%FT%TZ)" >> "$LOG"
done
echo "CCSCR_DE_RERUN ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
touch ccscr_de_rerun.done
