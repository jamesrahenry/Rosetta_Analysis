#!/bin/bash
# Cluster-A depth-resolved transfer pilot — chains after the D/E ccscr rerun.
set -u
cd /storage/JamesData/p4_nullfloor
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache P4_REGEN_STAGE=$ROOT/stage P4_PEAK_CACHE=$ROOT/peakcache
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=$ROOT/venv/bin/python
LOG=$ROOT/drt_pilot.log
echo "DRT_PILOT waiting for ccscr_de_rerun.done $(date -u +%FT%TZ)" >> "$LOG"
while [ ! -f ccscr_de_rerun.done ]; do sleep 300; done
echo "DRT_PILOT START $(date -u +%FT%TZ)" >> "$LOG"
$PY depth_resolved_transfer.py --cluster A --depths 0.1,0.2,0.3,0.4,0.5 \
    --K 3 --floorK 3 --out drt_A_s1 >> $ROOT/drt_A_s1.log 2>&1 &
P1=$!
$PY depth_resolved_transfer.py --cluster A --depths 0.6,0.7,0.8,0.9,1.0 \
    --K 3 --floorK 3 --out drt_A_s2 >> $ROOT/drt_A_s2.log 2>&1 &
P2=$!
wait $P1; E1=$?
wait $P2; E2=$?
echo "DRT_PILOT DONE s1=$E1 s2=$E2 $(date -u +%FT%TZ)" >> "$LOG"
touch drt_pilot.done
