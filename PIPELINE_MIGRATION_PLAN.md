# Pipeline migration plan — extraction core → rosetta_tools

*Written: 2026-07-20 12:34 UTC*
*Updated: 2026-07-22 06:19 UTC — corrected the version-reconciliation section to
match actual repo state (local/remote have diverged; the earlier "local behind at
1.3.1" description was wrong).*

**Follow-up, do not execute yet.** Scheduled for the pre-publication coordinated
version bump (P3/P4 publish + P1/P2 republish). Scopes moving the CAZ/GEM
*pipeline* logic out of `Rosetta_Analysis/extraction/extract.py` and into
`rosetta_tools`, per the scope rule: **rosetta_tools = reusable CAZ/GEM
pipeline; Rosetta_Analysis = paper-writing support** (rosters, regeneration,
HF upload, manuscript glue).

## Why

`concept_quality_report` (the new inline QA) is already correctly placed in
`rosetta_tools.caz`. But it is *called from* `extract_layer_wise_metrics`, which
is itself pipeline logic living on the paper-support side — a function that is
part of the CAZ/GEM pipeline reaching back into rosetta_tools is the tell that
the boundary is drawn one layer too shallow. This migration redraws it.

## What MOVES  (Rosetta_Analysis/extraction/extract.py → rosetta_tools)

- **`extract_layer_wise_metrics()`** — per-layer S/C/velocity/DOM computation,
  peak detection, the QA embed, and the `layer_data` record assembly. This is
  the extraction core. Target: a new `rosetta_tools.extraction` surface (or
  extend `rosetta_tools.caz`), sitting alongside the primitives it already uses
  (`compute_separation`, `compute_coherence`, `compute_velocity`,
  `find_caz_regions`, `extract_layer_activations`, `concept_quality_report`).
- **The caz-record schema builder** — the `results` dict in `extract_concept`
  (model_id, concept, hidden_dim, n_layers, token_pos, layer_data, …). Building
  the record is pipeline; *writing the file* and *uploading to HF* stay.
- **The cross-model consistency checker** (still to be built — the corpus-level
  pass that catches a clean systematic label inversion, the split-half blind
  spot). Reusable ⇒ belongs in rosetta_tools from the start, not Rosetta_Analysis.

## What STAYS  (Rosetta_Analysis — paper-support driver)

- Model roster selection (`--p1-corpus`, `--frontier`, `--prh-cluster`, …),
  model loading / device / dtype / quantization orchestration, CLI argparse.
- Concept-pair / corpus loading (paper-specific dataset glue), `run_summary` /
  manifest assembly, HF upload, cache cleaning, `--no-quality` and friends.
- After migration, `extract.py` is a thin driver: pick roster → load model →
  call the rosetta_tools extractor → write/upload. Any consumer calling the
  rosetta_tools extractor then gets the QA block for free.

## Version implication

Likely a **major** bump (rosetta_tools 2.0.0), not a minor, because:
- New public API surface (`rosetta_tools.extraction`), and
- signatures/contracts that Rosetta_Analysis and other consumers import change,
- and this is the natural moment to clean up other rosetta_tools APIs at once.

The caz JSON gains `layer_data.quality` — additive, backward-compatible for
readers (tolerate the new key), so the *data* schema change is non-breaking even
if the *code* API is.

## Version-reconciliation reality  (verified 2026-07-22 06:19 UTC)

**Ignore the earlier "local is at 1.3.1, behind the pinned v1.4.0" framing — it was
wrong.** The true state of `rosetta_tools`:

- **Local HEAD `8eb5d83`** ("feat(caz): inline extraction-time QA/stability report",
  pyproject `version = "1.5.0"`) — **untagged and unpushed**. It *does* already
  contain the `v1.4.0` tag in its history.
- **Remote `origin/main` HEAD `e7d08a6`** — pyproject still `version = "1.3.1"`. Does
  **not** contain the QA commit.
- Local and remote **have diverged** at merge-base `b567388` (one commit past
  `v1.4.0`). This is a real merge, *not* a fast-forward / rebase-and-tag:
  - **Local-only (2):** `9f14302` (gpu_utils: caller-supplied `max_memory`) and
    `8eb5d83` (the QA report + the 1.5.0 pyproject bump).
  - **Remote-only (~10):** the staging-flatten commits (`ada9a01`, `831b90d`) and a
    pile of `gpu_daemon`/`gpu_utils` infra (poll interval, `GPU_Runs` rename, MPS
    support, `HOPPER_TASK_ID` export, macOS `df`…), plus `e7d08a6` which is a
    **second copy of the same `max_memory` change** as local `9f14302` (identical
    title) — **de-duplicate at merge; keep one.**
- **Tag/pyproject skew:** the `v1.4.0` tag (`be923f5`) was cut with pyproject still
  at `1.3.1`; remote `main` is likewise `1.3.1` in-code. So the in-code version was
  never bumped to match the `v1.4.0` tag. Only the local branch set pyproject (→1.5.0).
- **Parent repo (`Rosetta_Program`) submodule pointer** is still `09b933e` in its
  committed tree; the working tree has moved it to `8eb5d83` (shows as uncommitted
  `M rosetta_tools`). Commit the pointer once the merge below lands.

**Corrected reconciliation steps** for the coordinated 2.0.0 bump:

1. **Merge the two divergent lines** — bring `origin/main`'s infra (staging-flatten +
   gpu_daemon) together with the local QA line. **De-duplicate** the `max_memory`
   commit (`9f14302` local vs `e7d08a6` remote — keep one).
2. Land the migration (the `## What MOVES` section) on the merged line.
3. **Set pyproject `version = "2.0.0"`** on `rosetta_tools` (there is no live v1.5.0 —
   drop that number) and the matching `Rosetta_Analysis` version. Cut & push `v2.0.0`.
4. Commit the parent-repo submodule pointer to the reconciled `rosetta_tools` HEAD.
5. Re-pin every consumer (`Rosetta_Analysis`, CIA, omnibus notebooks) to `@v2.0.0`,
   and **re-mint the Zenodo DOI** (currently `10.5281/zenodo.20361433` @ v1.3.1).

(RA_merge → `Rosetta_Analysis` dir rename already done, 2026-07-20.)

## Sequencing

1. Build + run the retrospective QA sweep (calibrates `QUALITY_THRESHOLDS`; also
   a pre-submission corpus audit). The cross-model checker is built here.
2. Execute this migration + coordinated version bump.
3. Code cleanup + documentation pass (separate follow-up).
4. Publish.
