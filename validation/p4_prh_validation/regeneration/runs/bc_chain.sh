#!/bin/bash
# B/C depth + stage-matched replication (trimmed: 7 depths, strided half-pairs).
set -u
cd /storage/JamesData/p4_nullfloor
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache P4_REGEN_STAGE=$ROOT/stage P4_PEAK_CACHE=$ROOT/peakcache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=$ROOT/venv/bin/python
DEPTHS=0.1,0.2,0.4,0.5,0.6,0.8,0.9
LOG=$ROOT/bc_chain.log
run_stream () {  # run_stream <cluster> <maxpairs>
  local c=$1 mp=$2
  echo "BC START depth $c $(date -u +%FT%TZ)" >> "$LOG"
  $PY depth_resolved_transfer.py --cluster $c --depths $DEPTHS --K 3 --floorK 3 \
      --max-pairs $mp --out drt_$c >> $ROOT/drt_$c.log 2>&1
  echo "BC DONE depth $c exit=$? $(date -u +%FT%TZ)" >> "$LOG"
  echo "BC START stage $c $(date -u +%FT%TZ)" >> "$LOG"
  $PY stage_matched_transfer.py --cluster $c --K 2 --max-pairs $mp \
      --out smt_$c >> $ROOT/smt_$c.log 2>&1
  echo "BC DONE stage $c exit=$? $(date -u +%FT%TZ)" >> "$LOG"
}
run_stream B 18 & P1=$!
run_stream C 13 & P2=$!
wait $P1; wait $P2
# cell-count reconciliation: expected = pairs x 17 x depths (minus logged skips)
$PY - << "PYR" >> "$LOG" 2>&1
import json
for c, mp in (("B", 18), ("C", 13)):
    try:
        d = json.load(open(f"drt_{c}/depth_transfer.json"))
        exp = mp * 17 * len(d["meta"]["depths"])
        print(f"RECON {c}: depth rows {len(d[rows])}/{exp}")
        s = json.load(open(f"smt_{c}/stage_matched.json"))
        print(f"RECON {c}: stage rows {len(s[rows])}")
    except Exception as e:
        print(f"RECON {c}: FAILED {type(e).__name__}: {e}")
PYR
echo "BC_CHAIN ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
touch bc_chain.done
