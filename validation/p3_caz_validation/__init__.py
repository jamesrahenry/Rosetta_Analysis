"""P3 (CAZ Validation, caz-validation/preprint.md) regeneration + claim scripts.

`regeneration/` holds the paper-specific recompute and figure scripts that
produce every number, table, and figure in the P3 manuscript from the
frozen paper_n250 corpus. See regeneration/README.md for the
number -> script map. The upstream data pipeline (extraction, per-model
ablation/global-sweep/patching/gem/caz artifacts) lives in the repo's
extraction/, gem/, and caz/ modules; these scripts consume those outputs.
"""
