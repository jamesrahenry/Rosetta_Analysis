"""
build_gems.py — Batch Geometric Evolution Map (GEM) builder for all models × concepts.

Reads existing caz_*.json and ablation_multimodal_*.json files to construct
ConceptGEM objects, then writes gem_*.json files alongside the existing
extraction data.

Schema v2 (2026-10-15) is ON by default: every GEM also gets a LayerProfile
(per-layer records at ALL layers, Fisher + projected separation with
permutation nulls, attention/MLP attribution per layer, post-final-norm
stream) and a gem_<concept>.sidecar.npz. The attention/MLP split needs a
forward pass, so by default this script LOADS THE MODEL (GPU if available)
and re-runs the same concept pairs the caz_*.json was built from. Switches:

    --legacy               pre-v2 output: caz_*.json only, no model, no GPU
    --no-sublayer-capture  stream-only profile from calibration_alllayer_*.npy
                           (no model needed; no attn/MLP split)
    see --help for the rest (GEMExtractionConfig fields)

Usage
-----
    python gem/build_gems.py --model EleutherAI/pythia-1.4b
    python gem/build_gems.py --all --legacy
    python gem/build_gems.py --model Qwen/Qwen2.5-0.5B --concepts negation --out-dir /tmp/x

Outputs: ~/rosetta_data/models/<model_slug>/gem_<concept>.json (+ .sidecar.npz)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import rosetta_tools  # noqa: F401  (import-path probe)
except ImportError:  # GPU-host layout first, then dev machine (see Rosetta_Program/CLAUDE.md)
    for _p in (Path.home() / "rosetta_tools", Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"):
        if _p.exists():
            sys.path.insert(0, str(_p))
            break

from rosetta_tools.gem import (
    GEMExtractionConfig, discover_concepts, discover_all_models, find_extraction_dir,
)

# ---------------------------------------------------------------------------
# Model / extraction discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _load_pos_neg_by_layer(extraction_dir: Path, concept: str, n_pairs: int) -> tuple[dict, dict] | None:
    """Load calibration_alllayer_{concept}.npy (raw per-example activations,
    saved by extraction but never previously read by the GEM-building step)
    and split it into h_pos_by_layer / h_neg_by_layer dicts, per the layout
    documented at extraction time: shape [n_layers, n_pos+n_neg, hidden_dim],
    rows 0..n_pairs-1 = positive, rest = negative. Returns None if the file
    isn't present (e.g. not saved for this model/concept) rather than
    raising -- callers should treat that as "Phase 2 unavailable here."
    """
    import numpy as np

    path = extraction_dir / f"calibration_alllayer_{concept}.npy"
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    n_layers = arr.shape[0]
    h_pos_by_layer = {l: np.array(arr[l, :n_pairs, :]) for l in range(n_layers)}
    h_neg_by_layer = {l: np.array(arr[l, n_pairs:, :]) for l in range(n_layers)}
    return h_pos_by_layer, h_neg_by_layer


class _ModelHandle:
    """Loads the model/tokenizer lazily (only if some concept needs a
    sublayer capture) and once per model. Same loading as extraction/
    extract.py for small models: AutoModel, get_dtype(device) (bf16 on GPU)."""

    def __init__(self, model_id: str, device: str | None, dtype: str | None):
        self.model_id, self._device, self._dtype = model_id, device, dtype
        self.model = self.tokenizer = None

    def get(self):
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from rosetta_tools.gpu_utils import get_device, get_dtype, log_vram
            self.device = self._device or get_device()
            dtype = (getattr(torch, self._dtype) if self._dtype and self._dtype != "auto"
                     else get_dtype(self.device))
            log.info("  loading %s (%s, %s) for sublayer capture", self.model_id, self.device, dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
            log_vram("after model load")
        return self.model, self.tokenizer, self.device

    def release(self):
        if self.model is not None:
            from rosetta_tools.gpu_utils import release_model
            release_model(self.model)
            self.model = None


def _build_layer_profile_for(
    concept: str, caz_data: dict, extraction_dir: Path, cfg: GEMExtractionConfig,
    handle: _ModelHandle,
):
    """Return (profile, arrays, docs) or None per cfg. Never raises for a
    missing optional input; logs and returns None instead (the k=1 GEM is
    still written)."""
    from rosetta_tools.gem_sublayer import build_layer_profile, extract_concept_layer_profile
    n_pairs = caz_data.get("n_pairs", 0)
    split = caz_data.get("split", "train")
    if cfg.sublayer_capture:
        from rosetta_tools.dataset import load_concept_pairs, texts_by_label
        pairs = load_concept_pairs(concept, n=n_pairs, split=split)
        if len(pairs) != n_pairs:
            log.warning("  %s: loaded %d pairs, caz_*.json used %d -- profile not comparable, skipping",
                        concept, len(pairs), n_pairs)
            return None
        pos, neg = texts_by_label(pairs)
        model, tok, device = handle.get()
        return extract_concept_layer_profile(
            model, tok, pos, neg, model_id=handle.model_id, device=device, config=cfg,
            dataset=f"Rosetta_Concept_Pairs v1 load_concept_pairs({concept!r}, n={n_pairs}, split={split!r})",
        )
    loaded = _load_pos_neg_by_layer(extraction_dir, concept, n_pairs) if n_pairs else None
    if loaded is None:
        log.warning("  %s: no calibration_alllayer_%s.npy -- no stream-only layer profile", concept, concept)
        return None
    hp, hn = loaded
    import numpy as np
    L = len(hp)
    from rosetta_tools.gem import Provenance
    prov = Provenance(
        method="build_gems stream-only: build_layer_profile on calibration_alllayer (hidden_states[1:]); "
               "last layer is post-final-norm, no sublayer split",
        model_id=caz_data.get("model_id", ""), capture_dtype="float32",
        dataset=f"calibration_alllayer_{concept}.npy", pooling="last",
    )
    return build_layer_profile(
        np.stack([hp[l] for l in range(L)]), np.stack([hn[l] for l in range(L)]),
        n_perm=cfg.n_permutations, seed=cfg.permutation_seed, provenance=prov,
        stream_source="hidden_states",
    )


def build_model_gems(
    model_id: str,
    concepts: list[str] | None = None,
    k: int = 1,
    force: bool = False,
    compute_settled_subspace: bool = False,
    settled_subspace_kwargs: dict | None = None,
    config: GEMExtractionConfig | None = None,
    out_dir: Path | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> dict:
    """Build GEMs for one model across all concepts.

    config : GEMExtractionConfig (default: GEMExtractionConfig(), schema v2
        with everything on). GEMExtractionConfig.legacy() reproduces the
        pre-2026-10-15 output (v1 fields identical; v2 keys present as null).
    out_dir : write here instead of the extraction dir (for smoke tests).

    k > 1 or compute_settled_subspace=True both require raw per-example
    activations (calibration_alllayer_{concept}.npy, saved at extraction
    time but unused by this script before 2026-09-25). If that file is
    missing for a given concept, Phase 2 / settled-subspace computation is
    skipped for that concept only (falls back to k=1, logged as a warning)
    rather than failing the whole run -- most of the corpus may not have
    this file saved, and that shouldn't block building the k=1 GEMs that
    only need caz_*.json.

    compute_settled_subspace is expensive (INLP with permutation-test
    fallback; see attach_settled_subspace's docstring for measured
    runtimes) -- do not enable across a full --all run without first
    estimating cost on a couple of models.

    Returns dict of {concept: diagnostics_dict}.
    """
    from rosetta_tools.models import attention_paradigm_of

    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s", model_id)
        return None

    if concepts is None:
        concepts = discover_concepts(extraction_dir, source="caz")

    paradigm = attention_paradigm_of(model_id)
    cfg = config or GEMExtractionConfig()
    target_dir = Path(out_dir) if out_dir else extraction_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    handle = _ModelHandle(model_id, device, dtype)
    log.info("=== Building GEMs: %s [%s] layer_profile=%s sublayer=%s ===",
             model_id, paradigm, cfg.layer_profile, cfg.sublayer_capture)

    results = {}
    try:
        results = _build_concepts(model_id, concepts, extraction_dir, target_dir, paradigm, cfg,
                                  handle, k, force, compute_settled_subspace, settled_subspace_kwargs)
    finally:
        handle.release()
    return results


def _build_concepts(model_id, concepts, extraction_dir, target_dir, paradigm, cfg, handle,
                    k, force, compute_settled_subspace, settled_subspace_kwargs) -> dict:
    from dataclasses import replace
    from rosetta_tools.gem import (
        build_concept_gem, save_gem, validate_gem_node, gem_diagnostics,
    )
    results = {}
    for concept in concepts:
        caz_path = extraction_dir / f"caz_{concept}.json"
        if not caz_path.exists():
            log.warning("  No caz_%s.json, skipping", concept)
            continue

        out_path = target_dir / f"gem_{concept}.json"
        if out_path.exists() and not force:
            # Check freshness: GEM should be newer than caz data AND match n_pairs
            if out_path.stat().st_mtime > caz_path.stat().st_mtime:
                try:
                    caz_n = json.loads(caz_path.read_text()).get("n_pairs", 0)
                    existing = json.loads(out_path.read_text())
                    gem_n = existing.get("n_pairs", 0)
                    # A v1 GEM is stale when the config asks for a layer profile.
                    has_profile = existing.get("layer_profile") is not None
                    if gem_n >= caz_n > 0 and (has_profile or not cfg.layer_profile):
                        log.info("  %s: GEM up-to-date, skipping (use --force to rebuild)",
                                 concept)
                        continue
                    log.info("  %s: GEM n_pairs mismatch (gem=%d caz=%d) — rebuilding",
                             concept, gem_n, caz_n)
                except Exception:
                    pass  # fall through to rebuild

        caz_data = json.loads(caz_path.read_text())

        # Load multimodal interaction data if available
        mm_path = extraction_dir / f"ablation_multimodal_{concept}.json"
        multimodal_data = None
        if mm_path.exists():
            multimodal_data = json.loads(mm_path.read_text())

        # k>1 (Phase 2) and/or compute_settled_subspace both need raw
        # per-example activations, which caz_*.json doesn't have.
        h_pos_by_layer = h_neg_by_layer = None
        this_k = k
        this_compute_settled_subspace = compute_settled_subspace
        if k > 1 or compute_settled_subspace:
            n_pairs = caz_data.get("n_pairs", 0)
            loaded = _load_pos_neg_by_layer(extraction_dir, concept, n_pairs) if n_pairs else None
            if loaded is None:
                log.warning(
                    "  %s: calibration_alllayer_%s.npy not found — falling back to k=1, "
                    "no settled_subspace (raw activations required for Phase 2)",
                    concept, concept,
                )
                this_k = 1
                this_compute_settled_subspace = False
            else:
                h_pos_by_layer, h_neg_by_layer = loaded

        layer_profile = arrays = docs = None
        if cfg.layer_profile:
            try:
                built = _build_layer_profile_for(concept, caz_data, extraction_dir, cfg, handle)
            except (ValueError, RuntimeError, NotImplementedError) as exc:
                # e.g. additivity check failed / unsupported block: do not
                # write wrong numbers; keep the v1 GEM.
                log.error("  %s: layer profile FAILED (%s) -- writing GEM without it", concept, exc)
                built = None
            if built is not None:
                layer_profile, arrays, docs = built

        try:
            # Build the GEM
            gem = build_concept_gem(
                caz_data,
                multimodal_data=multimodal_data,
                attention_paradigm=paradigm,
                k=this_k,
                h_pos_by_layer=h_pos_by_layer,
                h_neg_by_layer=h_neg_by_layer,
                compute_settled_subspace=this_compute_settled_subspace,
                settled_subspace_kwargs=settled_subspace_kwargs,
                layer_profile=layer_profile,
                detection_score=cfg.detection_score if layer_profile is not None else "fisher",
            )
        except (ValueError, KeyError, IndexError) as exc:
            log.error("  %s: FAILED — %s", concept, exc)
            continue

        # Validate every node
        all_warnings = []
        for node in gem.nodes:
            warnings = validate_gem_node(node)
            if warnings:
                for w in warnings:
                    log.warning("  %s node %d: %s", concept, node.caz_index, w)
                all_warnings.extend(warnings)

        # Save — carry n_pairs from caz_data (top-level key, via `extra`) so
        # the skip logic can validate it; large arrays go to the sidecar.
        gem = replace(gem, extra={**(gem.extra or {}), "n_pairs": caz_data.get("n_pairs", 0)})
        save_gem(gem, out_path, arrays=arrays, array_docs=docs,
                 inline_directions=cfg.inline_directions and arrays is not None)

        # Diagnostics
        diag = gem_diagnostics(gem)
        results[concept] = diag

        # Summary line
        n_targets = len(gem.ablation_targets or [])
        types_str = ""
        if gem.node_types:
            from collections import Counter
            counts = Counter(gem.node_types)
            types_str = " | ".join(f"{t}={c}" for t, c in sorted(counts.items()))

        log.info(
            "  %s: %d nodes, %d targets | handoff_cos=%.3f | entry_exit=%.3f | %s%s",
            concept,
            gem.n_nodes,
            n_targets,
            diag.get("handoff_cosine_mean", 0),
            diag.get("entry_exit_cosine_mean", 0),
            types_str,
            f" | {len(all_warnings)} warnings" if all_warnings else "",
        )

    return results


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(all_results: dict[str, dict[str, dict]]) -> None:
    """Print aggregate summary across all models."""
    import numpy as np

    print("\n" + "=" * 80)
    print("GEM BUILD SUMMARY")
    print("=" * 80)

    total_nodes = 0
    total_targets = 0
    all_handoff = []
    all_entry_exit = []
    all_rotation = []

    for model_id, concept_diags in sorted(all_results.items()):
        model_nodes = 0
        model_targets = 0
        for concept, diag in concept_diags.items():
            n = diag.get("n_nodes", 0)
            t = diag.get("n_ablation_targets", 0)
            model_nodes += n
            model_targets += t
            total_nodes += n
            total_targets += t
            if n > 0:
                all_handoff.append(diag["handoff_cosine_mean"])
                all_entry_exit.append(diag["entry_exit_cosine_mean"])
                all_rotation.append(diag["max_rotation_mean"])

        short = model_id.split("/")[-1]
        print(f"  {short:<30} {model_nodes:>3} nodes, {model_targets:>3} targets "
              f"({len(concept_diags)} concepts)")

    print("-" * 80)
    print(f"  {'TOTAL':<30} {total_nodes:>3} nodes, {total_targets:>3} targets")

    if all_handoff:
        print(f"\n  Handoff cosine:    mean={np.mean(all_handoff):.3f}  "
              f"std={np.std(all_handoff):.3f}  "
              f"range=[{np.min(all_handoff):.3f}, {np.max(all_handoff):.3f}]")
        print(f"  Entry-exit cosine: mean={np.mean(all_entry_exit):.3f}  "
              f"std={np.std(all_entry_exit):.3f}  "
              f"range=[{np.min(all_entry_exit):.3f}, {np.max(all_entry_exit):.3f}]")
        print(f"  Max rotation/layer: mean={np.mean(all_rotation):.3f}  "
              f"range=[{np.min(all_rotation):.3f}, {np.max(all_rotation):.3f}]")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build CAZ GEMs (Geometric Evolution Maps). Schema v2 layer profile on by "
                    "default (loads the model for the attn/MLP split); --legacy needs no GPU.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true",
                       help="Build for all models with extraction results")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts (default: all 7)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1 (k=1, dom_vector) or Phase 2 (k>1, deep dive)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of eigenvector threads to track (Phase 2)")
    parser.add_argument("--settled-subspace", action="store_true",
                        help="Also compute a multi-direction SettledSubspace at each node's "
                             "handoff layer via INLP (2026-09-25). Requires "
                             "calibration_alllayer_{concept}.npy; falls back to skipping per "
                             "concept if missing. EXPENSIVE (permutation-test fallback can take "
                             "30-90+ min per layer on a small model) -- do not enable across "
                             "--all without first estimating cost on a couple of models.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if GEM files are up-to-date")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID", help="Skip this model (may be repeated)")
    # Schema v2 switches (GEMExtractionConfig). Defaults: everything on.
    v2 = parser.add_argument_group("schema v2 (layer profile)")
    v2.add_argument("--legacy", action="store_true",
                    help="Pre-v2 output: no layer profile, no model load (overrides the switches below)")
    v2.add_argument("--no-layer-profile", action="store_true", help="Skip the LayerProfile")
    v2.add_argument("--no-sublayer-capture", action="store_true",
                    help="Stream-only profile from calibration_alllayer_*.npy (no model load)")
    v2.add_argument("--no-full-sublayer-activations", action="store_true",
                    help="Do not store raw per-example attn/MLP writes in the sidecar")
    v2.add_argument("--no-post-final-norm", action="store_true",
                    help="Do not record the post-final-norm stream at the last layer")
    v2.add_argument("--inline-directions", action="store_true",
                    help="Also write per-layer directions inline in the JSON")
    v2.add_argument("--n-permutations", type=int, default=200)
    v2.add_argument("--detection-score", choices=["fisher", "projected"], default="fisher",
                    help="CAZ detector input; 'projected' is the parallel research path")
    v2.add_argument("--allow-inexact", action="store_true",
                    help="Record (and flag) instead of failing when Delta != attn + mlp")
    v2.add_argument("--additivity-tol", type=float, default=None)
    v2.add_argument("--batch-size", type=int, default=16)
    v2.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    v2.add_argument("--dtype", default=None, choices=["auto", "bfloat16", "float16", "float32"],
                    help="Forward-pass dtype (default: auto = bf16 on GPU, as extraction)")
    v2.add_argument("--out-dir", type=Path, default=None,
                    help="Write GEMs here instead of the extraction dir (single --model only)")
    args = parser.parse_args()

    if args.legacy:
        cfg = GEMExtractionConfig.legacy()
    else:
        cfg = GEMExtractionConfig(
            layer_profile=not args.no_layer_profile,
            sublayer_capture=not args.no_sublayer_capture,
            store_full_sublayer_activations=not args.no_full_sublayer_activations,
            store_post_final_norm=not args.no_post_final_norm,
            inline_directions=args.inline_directions,
            n_permutations=args.n_permutations,
            detection_score=args.detection_score,
            allow_inexact=args.allow_inexact,
            additivity_tol=args.additivity_tol,
            batch_size=args.batch_size,
        )
    if args.out_dir and args.all:
        parser.error("--out-dir only with --model")

    concepts = args.concepts or None  # None → auto-discover per model from caz_*.json

    if args.phase == 2 and args.k == 1:
        args.k = 3
        log.info("Phase 2 requested; defaulting to k=3")

    if args.all:
        models = discover_all_models()
        log.info("Found %d models with extraction results", len(models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]

    all_results = {}
    any_extraction_found = False
    for model_id in models:
        results = build_model_gems(
            model_id, concepts, k=args.k, force=args.force,
            compute_settled_subspace=args.settled_subspace,
            config=cfg, out_dir=args.out_dir, device=args.device, dtype=args.dtype,
        )
        if results is None:
            continue  # no extraction dir — already logged as error
        any_extraction_found = True
        if results:
            all_results[model_id] = results

    if all_results:
        print_summary(all_results)
    elif any_extraction_found:
        log.info("All GEMs already up-to-date.")
    else:
        log.warning("No GEMs built. Check that extraction data exists.")


if __name__ == "__main__":
    main()
