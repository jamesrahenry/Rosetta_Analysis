# Round-3 GPU battery — surfaced from the papers workspace 2026-07-31

*Per policy (James, 2026-07-31): all paper-generating code lives in Rosetta_Analysis.*

These are the P3/P4 round-3 GPU session scripts (exfiltration correction, G2/G3/G5/G6
batteries, Gemma mechanism tests, cluster-F extraction, handoff nulls) copied verbatim
from `Rosetta_Program/papers/shared/round3_gpu/` — previously the only copy. The local
`common.py`/`forward_utils.py` here are the variants these scripts import (they differ
from `validation/p4_prh_validation/regeneration/{,gpu/}` copies; imports resolve within
this directory). `g5b_random_text_null_original_corpus.py` also exists at
`validation/p4_prh_validation/regeneration/gpu/` — identical at surfacing time.
Run-context notes remain in the papers workspace (`BRINGUP_NOTES.md`, `EXFILTRATION_RERUN_SPEC.md`).
