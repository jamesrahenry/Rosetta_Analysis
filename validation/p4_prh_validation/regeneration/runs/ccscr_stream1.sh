#!/bin/bash
cd /storage/JamesData/p4_nullfloor
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=/storage/JamesData/p4_nullfloor/venv/bin/python
LOG=ccscr_stream1.log
echo "CCSCR_STREAM1 START $(date -u +%FT%TZ) : C E F" >> "$LOG"
for c in C E F; do
  echo "START $c $(date -u +%FT%TZ)" >> "$LOG"
  $PY crossconcept_scramble.py --cluster $c --K 8 --out ccscr_$c >> "$LOG" 2>&1
  echo "DONE  $c exit=$? $(date -u +%FT%TZ)" >> "$LOG"
done
echo "CCSCR_STREAM1 ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
touch ccscr_stream1.done
