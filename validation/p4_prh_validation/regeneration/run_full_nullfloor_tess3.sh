#!/usr/bin/env bash
# Stage 3: cross-concept transport under a scrambled fit.
# Waits for the tess2 floor/scramble chain to drain, then runs the test that
# decides whether the cross-concept signal (0.302 vs a 0.0004 floor) is
# independent of cross-architecture text correspondence.
#
# Cluster B (d=2048) is the reference cluster for the cross-concept line -- the
# existing 0.302 / 0.0004 numbers are B, so this stays comparable. All 36 pairs,
# all 17 concepts as fit-concept (cyclic test-concept), K=8.
set -u
cd "$(dirname "$0")"

ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache
export P4_REGEN_STAGE=$ROOT/stage
export P4_PEAK_CACHE=$ROOT/peakcache
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=$ROOT/venv/bin/python
LOG=$ROOT/tess_sweep_chain.log

until grep -q "STREAM_A ALL DONE" "$LOG" 2>/dev/null && grep -q "STREAM_B ALL DONE" "$LOG" 2>/dev/null; do sleep 60; done

echo "TESS3 START $(date -u +%FT%TZ)" >> "$LOG"
$PY crossconcept_scramble.py --cluster B --K 8 \
    --out "$ROOT/ccscr_B" > "$ROOT/ccscr_B.log" 2>&1
echo "TESS3 DONE exit=$? $(date -u +%FT%TZ)" >> "$LOG"
