#!/bin/bash
# Phase-A audit-gap closure: d=8192 validate, fixed n-sweep, oob floors,
# bootstrap CIs, HF upload of all Phase-A/depth artifacts.
set -u
cd /storage/JamesData/p4_nullfloor
ROOT=/storage/JamesData/p4_nullfloor
export HF_HOME=$ROOT/cache P4_REGEN_STAGE=$ROOT/stage P4_PEAK_CACHE=$ROOT/peakcache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=$ROOT/venv/bin/python
LOG=$ROOT/gaps_chain.log
s1 () {
  echo "GAPS START validate $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode validate --K 8 --out nullfloor_out_v2 >> $ROOT/gaps_validate.log 2>&1
  echo "GAPS DONE validate exit=$? $(date -u +%FT%TZ)" >> "$LOG"
  echo "GAPS START nsweep $(date -u +%FT%TZ)" >> "$LOG"
  $PY nullfloor_analysis.py --mode nsweep --nsweep-cluster B --K 4 --max-pairs 12 \
      --out nsweep_v2 >> $ROOT/gaps_nsweep.log 2>&1
  echo "GAPS DONE nsweep exit=$? $(date -u +%FT%TZ)" >> "$LOG"
}
s2 () {
  echo "GAPS START oob A $(date -u +%FT%TZ)" >> "$LOG"
  $PY oob_floor.py --cluster A --K 3 --out oob_A >> $ROOT/gaps_oob_A.log 2>&1
  echo "GAPS DONE oob A exit=$? $(date -u +%FT%TZ)" >> "$LOG"
  echo "GAPS START oob B $(date -u +%FT%TZ)" >> "$LOG"
  $PY oob_floor.py --cluster B --K 3 --max-pairs 18 --out oob_B >> $ROOT/gaps_oob_B.log 2>&1
  echo "GAPS DONE oob B exit=$? $(date -u +%FT%TZ)" >> "$LOG"
}
s1 & P1=$!
s2 & P2=$!
wait $P1; wait $P2
echo "GAPS START bootstrap $(date -u +%FT%TZ)" >> "$LOG"
$PY phase_a_bootstrap.py --root $ROOT --B 2000 --out $ROOT/phase_a_ci.json >> $ROOT/gaps_boot.log 2>&1
echo "GAPS DONE bootstrap exit=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "GAPS START hf-upload $(date -u +%FT%TZ)" >> "$LOG"
$PY - << "PYU" >> $ROOT/gaps_upload.log 2>&1
import glob, os
from huggingface_hub import HfApi
api = HfApi()
REPO = "james-ra-henry/Rosetta-Activations"
ops = []
for pat in ("ccscr_*/crossconcept_scramble.json", "drt_*/depth_transfer.json",
            "smt_*/stage_matched.json", "oob_*/oob_floor.json",
            "nullfloor_full_*/spectrum_floor.json", "nullfloor_full_*/scramble.json",
            "nsweep_v2/nsweep_B.json", "nullfloor_out_v2/noise_floor.json",
            "phase_a_ci.json"):
    for f in glob.glob(pat):
        dest = "paper_n250/_phase_a/" + f
        api.upload_file(path_or_fileobj=f, path_in_repo=dest,
                        repo_id=REPO, repo_type="dataset")
        print("uploaded", dest, flush=True)
files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
         if f.startswith("paper_n250/_phase_a/")]
print("VERIFY: %d files under paper_n250/_phase_a/" % len(files))
PYU
echo "GAPS DONE hf-upload exit=$? $(date -u +%FT%TZ)" >> "$LOG"
echo "GAPS_CHAIN ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
touch gaps_chain.done
