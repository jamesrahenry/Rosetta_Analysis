# GPU run: site-matched depth control — the P2 §5.5 / P3 F8 joint experiment

*Written: 2026-07-30 14:55 UTC by claude:p4-phaseA-dive. Approved by James 2026-07-30
("just the one"). The single GPU run currently authorized for P1–P3; no other runs ride along.*

## What this run settles

P2 §5.5's handoff-vs-depth-matched comparison and P3's F8 review item are the same experiment:
ablate at the handoff layer(s) vs. an equally deep non-handoff control **at equal site count**,
over the same GEMs. The stored M6-era artifact on HF predates the site-matched implementation
(no `site_matched` key), and until `Rosetta_Analysis@12c1a0f` the script's legacy early-return
skipped the 111 single-GEM (terminal) atlases before the site-matched block ran — the exact
class carrying the crux comparison (single-site handoff arm vs single-site control), leaving
n=2. This run produces the first real site-matched numbers, at full population.

One pass covers both papers: `BASE_28` (P3 §6.5 population, 28 models — verified) is a strict
subset of `ablate_gem.P2_MODELS` (29; the extra is `meta-llama/Llama-3.1-8B`).

## The run

- **Script:** `gem/ablate_gem_depth_matched_control.py` at `Rosetta_Analysis@12c1a0f` or later.
  Do not run any earlier revision — the early-return defect reintroduces the skip, and the
  pre-12c1a0f aggregate crashes on terminal-atlas rows.
- **Invocation:**
  ```bash
  cd ~/rosetta_analysis   # GPU-host path convention (never ~/Source/...)
  python gem/ablate_gem_depth_matched_control.py --all --overwrite
  ```
  `--overwrite` is REQUIRED: existing per-model outputs predate the site-matched field and must
  be recomputed, not skipped.
- **Host:** one 40–48 GB card (A6000 / A100-40) is comfortable for the full roster in bf16
  (largest members: pythia-12b, Qwen2.5-14B). A 24 GB A10 works only if the two >9B models run
  8-bit — prefer the 48 GB class to keep all rows same-precision. The four >14B models
  (Qwen-32B/72B, Llama-70B, falcon-40b) are outside `P2_MODELS` and are NOT part of this run.
- **Orchestration:** Eigan stack per forward policy (Prefect flow + MLflow tracking; deployment
  target named by James at provision time). `HF_TOKEN` must be set (james-ra-henry personal
  token) — unauthenticated fetches have already cost this program one silent data loss.
- **Estimated wall/cost:** ~10–16 h, ≈ $20–30 at $1.3–1.8/hr. Models load once each; per-model
  work is 17 concepts × (activation extraction + legacy-arm ablation where defined + two
  site-matched-arm ablations).

## Outputs & upload (HF is the system of record)

- Per-model JSON: `<extraction_dir>_depth_matched_control.json` (now with `site_matched_control`
  per concept: layers, modes, both arms' retained %, delta) + `aggregate.json` (legacy and
  site-matched reported separately; site-matched split into `site_matched_post_caz` vs
  `site_matched_with_fallback` — the within-segment fallback is a weaker comparator and is
  never silently pooled).
- **Upload before teardown:** `james-ra-henry/Rosetta-Activations`, tree
  `paper_n250/_gem_depth_matched/` (supersedes the M6-era artifacts there). Verify the upload
  by listing the repo, not by trusting the log.

## Success criteria (check before teardown)

1. 29 model JSONs present; **cell-count reconciliation**: total non-skipped concept rows ≈ 493
   pairs (the P2 population), of which ~111 terminal-atlas rows must carry
   `site_matched_control.site_matched: true` with `control_modes` containing `within_segment` —
   if those rows are absent, the defect class was skipped again; stop and investigate.
2. `aggregate.json` has non-null `site_matched_post_caz` and `site_matched_with_fallback`
   blocks.
3. Rows with `delta_pp: null` are exactly the terminal atlases (legacy arm undefined) — not
   errors.

## Consumers

- **P2 §5.5** (owner: td198454 board thread): replaces the n=2 site-matched sentence; the
  site-matched delta and win-rate, both modes reported separately.
- **P3 F8** (owner: t7dce6a8): same numbers restated for the §6.5 population (BASE_28 subset of
  the output).
- Route results via Hopper notes on both tasks; neither paper's owner should re-run anything —
  this artifact is the record.
