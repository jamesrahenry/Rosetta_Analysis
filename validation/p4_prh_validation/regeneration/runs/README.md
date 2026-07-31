# Tesseract orchestration wrappers — Phase-A / depth-program runs

*Surfaced 2026-07-31 per James's directive: all paper-generating code lives in
Rosetta_Analysis; host-only copies are a gap.*

Shell wrappers exactly as executed on tesseract (`/storage/JamesData/p4_nullfloor`)
for the 2026-07-27..31 Phase-A and depth-program runs, plus `ccagg.py` (ad hoc
cross-concept aggregator). Paths are host-specific by design — these are run
records, not portable tools. The science scripts they invoke are the canonical
ones in the parent directory.

| wrapper | run |
|---|---|
| ccscr_stream{1,2}.sh | original cross-concept full-gamut streams (D/E crashed; see audit) |
| de_rerun.sh, b_scr_rerun.sh | D/E floor+scramble reruns (97bf090 guard) |
| ccscr_de_rerun.sh | guarded D/E cross-concept rerun (0fd1efe) |
| a_plurality_rerun.sh | cluster-A 204-cell floor/scramble rerun (cache-blob fix) |
| drt_pilot_chain.sh, smt_chain.sh | cluster-A depth pilot + stage-matched pass |
| bc_chain.sh | B/C depth+stage replication (strided pairs, reconciliation) |
| gaps_chain.sh | audit-gap closure: d=8192, n-sweep v2, oob floors, bootstrap, HF upload |
