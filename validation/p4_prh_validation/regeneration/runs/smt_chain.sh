#!/bin/bash
set -u
cd /storage/JamesData/p4_nullfloor
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache P4_REGEN_STAGE=$ROOT/stage P4_PEAK_CACHE=$ROOT/peakcache
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
LOG=$ROOT/smt_chain.log
echo "SMT waiting for drt_pilot.done $(date -u +%FT%TZ)" >> "$LOG"
while [ ! -f drt_pilot.done ]; do sleep 300; done
echo "SMT START $(date -u +%FT%TZ)" >> "$LOG"
$ROOT/venv/bin/python stage_matched_transfer.py --cluster A --K 2 --out smt_A >> $ROOT/smt_A.log 2>&1
echo "SMT DONE exit=$? $(date -u +%FT%TZ)" >> "$LOG"
touch smt_A.done
