"""
ablate_behavioral_pilot.py — Behavioral validation of CAZ peak causal claims.

Asks: does ablating the concept direction at the CAZ peak layer suppress
concept-diagnostic *next-token predictions* more than ablating the same
direction at a matched non-CAZ control layer?

For each (model, concept) pair:
  1. Load the concept direction (dom_vector at CAZ peak) and list of all
     detected CAZ layers from existing extraction results.
  2. Select a control layer: non-CAZ layer closest to model midpoint.
  3. For each probe sentence (21 total, 3 per concept), measure:
       logit_diff = log_p(pos_token) - log_p(neg_token)
     under four conditions:
       - baseline  : no ablation
       - peak      : concept direction ablated at CAZ peak
       - control   : concept direction ablated at non-CAZ midpoint layer
       - random    : random unit vector ablated at CAZ peak (direction null)
  4. Suppression = logit_diff(baseline) - logit_diff(condition)
     Prediction: suppression(peak) >> suppression(control) >> suppression(random≈0)

Output written to results/behavioral_pilot/<model_slug>/
  behavioral_pilot_<concept>.json   — per-concept detail
  behavioral_pilot_summary.json     — cross-concept aggregation
  behavioral_pilot_table.txt        — paper-ready table

Usage
-----
    python src/ablate_behavioral_pilot.py --model EleutherAI/pythia-1.4b
    python src/ablate_behavioral_pilot.py --all

Written: 2026-04-19 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, vram_stats,
)
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")

# ---------------------------------------------------------------------------
# Probe sentences: 3 per concept × 7 concepts = 21 probes
# pos_token: token that should be MORE likely given the context (concept-aligned)
# neg_token: token that should be LESS likely (concept-misaligned)
# After ablation of concept direction at CAZ peak, logit_diff should shrink.
# ---------------------------------------------------------------------------
BEHAVIORAL_PROBES = [
    # --- sentiment ---
    {
        "concept": "sentiment",
        "prefix": "The film deeply moved audiences and earned universal praise. Critics called it",
        "pos_token": " masterful",
        "neg_token": " disappointing",
    },
    {
        "concept": "sentiment",
        "prefix": "The experience was deeply unpleasant and left everyone feeling",
        "pos_token": " terrible",
        "neg_token": " wonderful",
    },
    {
        "concept": "sentiment",
        "prefix": "The warm sunny day lifted everyone's spirits and they all felt",
        "pos_token": " joyful",
        "neg_token": " miserable",
    },
    # --- credibility ---
    {
        "concept": "credibility",
        "prefix": "The peer-reviewed meta-analysis replicated across twelve countries found the treatment",
        "pos_token": " effective",
        "neg_token": " useless",
    },
    {
        "concept": "credibility",
        "prefix": "The anonymous forum post claiming vaccines cause harm was found to be",
        "pos_token": " false",
        "neg_token": " credible",
    },
    {
        "concept": "credibility",
        "prefix": "The Nobel laureate's findings, confirmed by independent laboratories, are widely considered",
        "pos_token": " reliable",
        "neg_token": " dubious",
    },
    # --- negation ---
    {
        "concept": "negation",
        "prefix": "It was not a good outcome at all. The entire team felt",
        "pos_token": " disappointed",
        "neg_token": " satisfied",
    },
    {
        "concept": "negation",
        "prefix": "The drug did not reduce symptoms in any patient. Patients therefore",
        "pos_token": " remained",
        "neg_token": " recovered",
    },
    {
        "concept": "negation",
        "prefix": "He could not understand the instructions at all, which made him feel",
        "pos_token": " confused",
        "neg_token": " confident",
    },
    # --- causation ---
    {
        "concept": "causation",
        "prefix": "Heavy rainfall flooded the roads, directly",
        "pos_token": " causing",
        "neg_token": " preventing",
    },
    {
        "concept": "causation",
        "prefix": "The new policy reduced unemployment rates, thereby",
        "pos_token": " improving",
        "neg_token": " worsening",
    },
    {
        "concept": "causation",
        "prefix": "Long-term smoking damages lung tissue and",
        "pos_token": " causes",
        "neg_token": " prevents",
    },
    # --- certainty ---
    {
        "concept": "certainty",
        "prefix": "Based on overwhelming evidence from hundreds of replicated studies, scientists are",
        "pos_token": " certain",
        "neg_token": " uncertain",
    },
    {
        "concept": "certainty",
        "prefix": "Without any supporting data or independent replication, the finding remains",
        "pos_token": " uncertain",
        "neg_token": " established",
    },
    {
        "concept": "certainty",
        "prefix": "All repeated experiments confirmed the same result, so researchers feel",
        "pos_token": " confident",
        "neg_token": " doubtful",
    },
    # --- moral_valence ---
    {
        "concept": "moral_valence",
        "prefix": "Torturing innocent animals purely for entertainment is morally",
        "pos_token": " wrong",
        "neg_token": " acceptable",
    },
    {
        "concept": "moral_valence",
        "prefix": "Donating generously to help disaster victims is considered morally",
        "pos_token": " good",
        "neg_token": " bad",
    },
    {
        "concept": "moral_valence",
        "prefix": "Stealing from the poor to enrich the already wealthy is deeply",
        "pos_token": " unjust",
        "neg_token": " justified",
    },
    # --- temporal_order ---
    {
        "concept": "temporal_order",
        "prefix": "First the proposal was submitted for review, and",
        "pos_token": " then",
        "neg_token": " before",
    },
    {
        "concept": "temporal_order",
        "prefix": "The rain began falling heavily, after which the flooding",
        "pos_token": " started",
        "neg_token": " ended",
    },
    {
        "concept": "temporal_order",
        "prefix": "Before the ceremony began, the invited guests had already",
        "pos_token": " arrived",
        "neg_token": " departed",
    },
]


# ---------------------------------------------------------------------------
# Results discovery
# ---------------------------------------------------------------------------

def find_extraction_dir(model_id: str) -> Path | None:
    candidates = []
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        s = d / "run_summary.json"
        if d.is_dir() and s.exists():
            try:
                if json.loads(s.read_text()).get("model_id") == model_id:
                    candidates.append(d)
            except Exception:
                continue
    for c in candidates:
        if any(c.glob("ablation_global_sweep_*.json")):
            return c
    return candidates[0] if candidates else None


def discover_models() -> list[str]:
    models = set()
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if s.exists():
            try:
                models.add(json.loads(s.read_text()).get("model_id", ""))
            except Exception:
                pass
    return sorted(m for m in models if m)


def load_concept_data(extraction_dir: Path, concept: str) -> dict | None:
    """Load direction, CAZ peak, and all detected CAZ layers for a concept."""
    caz_path = extraction_dir / f"caz_{concept}.json"
    sweep_path = extraction_dir / f"ablation_global_sweep_{concept}.json"
    if not caz_path.exists() or not sweep_path.exists():
        return None

    caz_data   = json.loads(caz_path.read_text())
    sweep_data = json.loads(sweep_path.read_text())

    caz_peak   = sweep_data["caz_peak"]
    n_layers   = sweep_data["n_layers"]
    metrics    = caz_data["layer_data"]["metrics"]
    direction  = np.array(metrics[caz_peak]["dom_vector"], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    # All detected CAZ layer indices (peaks within any detected zone)
    caz_layers: set[int] = set()
    for zone in caz_data.get("zones", []):
        for li in range(zone.get("start", 0), zone.get("end", 0) + 1):
            caz_layers.add(li)
    # Also mark ±2 around the peak itself as off-limits for control
    for delta in range(-2, 3):
        caz_layers.add(caz_peak + delta)

    return {
        "caz_peak":   caz_peak,
        "n_layers":   n_layers,
        "direction":  direction,
        "caz_layers": caz_layers,
    }


def select_control_layer(caz_layers: set[int], n_layers: int) -> int:
    """Non-CAZ layer closest to the model midpoint."""
    midpoint = n_layers // 2
    for delta in range(0, n_layers):
        for sign in ([0] if delta == 0 else [1, -1]):
            candidate = midpoint + sign * delta
            if 0 <= candidate < n_layers and candidate not in caz_layers:
                return candidate
    return midpoint  # fallback


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def get_token_id(tokenizer, token_str: str) -> int | None:
    """Return the first token ID for token_str, or None if it maps to unk."""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if not ids:
        return None
    tid = ids[0]
    if tid == tokenizer.unk_token_id:
        return None
    return tid


# ---------------------------------------------------------------------------
# Logit diff measurement
# ---------------------------------------------------------------------------

def measure_logit_diff(
    model,
    tokenizer,
    prefix: str,
    pos_id: int,
    neg_id: int,
    device: str,
    layers: list | None = None,
    ablate_layer: int | None = None,
    direction_t: "torch.Tensor | None" = None,
) -> float:
    """log_p(pos_id) - log_p(neg_id) at the final token position of prefix."""
    inputs = tokenizer(prefix, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    ctx = DirectionalAblator(layers[ablate_layer], direction_t) \
        if (ablate_layer is not None and layers is not None and direction_t is not None) \
        else _null_ctx()

    with ctx:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    return (log_probs[pos_id] - log_probs[neg_id]).item()


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Per-concept runner
# ---------------------------------------------------------------------------

def run_concept(
    model,
    tokenizer,
    concept: str,
    concept_data: dict,
    device: str,
    args,
) -> dict:
    layers = get_transformer_layers(model)
    caz_peak      = concept_data["caz_peak"]
    n_layers      = concept_data["n_layers"]
    direction     = concept_data["direction"]
    caz_layers    = concept_data["caz_layers"]
    ctrl_layer    = select_control_layer(caz_layers, n_layers)

    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32
    direction_t = torch.tensor(direction, dtype=dtype, device=device)
    direction_t = direction_t / direction_t.norm()

    rng = np.random.default_rng(seed=99)
    random_dir = rng.standard_normal(len(direction)).astype(np.float64)
    random_dir /= np.linalg.norm(random_dir)
    random_t = torch.tensor(random_dir, dtype=dtype, device=device)
    random_t = random_t / random_t.norm()

    log.info("  concept=%s  caz_peak=L%d  ctrl_layer=L%d  dim=%d",
             concept, caz_peak, ctrl_layer, len(direction))

    probes = [p for p in BEHAVIORAL_PROBES if p["concept"] == concept]
    probe_results = []

    for probe in probes:
        pos_id = get_token_id(tokenizer, probe["pos_token"])
        neg_id = get_token_id(tokenizer, probe["neg_token"])
        if pos_id is None or neg_id is None:
            log.warning("    Skipping probe (unk token): %s / %s",
                        probe["pos_token"], probe["neg_token"])
            continue

        prefix = probe["prefix"]
        t0 = time.time()

        ld_base  = measure_logit_diff(model, tokenizer, prefix, pos_id, neg_id, device)
        ld_peak  = measure_logit_diff(model, tokenizer, prefix, pos_id, neg_id, device,
                                      layers, caz_peak, direction_t)
        ld_ctrl  = measure_logit_diff(model, tokenizer, prefix, pos_id, neg_id, device,
                                      layers, ctrl_layer, direction_t)
        ld_rand  = measure_logit_diff(model, tokenizer, prefix, pos_id, neg_id, device,
                                      layers, caz_peak, random_t)

        supp_peak = ld_base - ld_peak
        supp_ctrl = ld_base - ld_ctrl
        supp_rand = ld_base - ld_rand

        log.info("    probe='%s...'  base=%.3f  peak_supp=%.3f  ctrl_supp=%.3f  rand_supp=%.3f  (%.1fs)",
                 prefix[:40], ld_base, supp_peak, supp_ctrl, supp_rand, time.time() - t0)

        probe_results.append({
            "prefix":         prefix,
            "pos_token":      probe["pos_token"],
            "neg_token":      probe["neg_token"],
            "pos_id":         int(pos_id),
            "neg_id":         int(neg_id),
            "logit_diff_baseline":   round(ld_base, 4),
            "logit_diff_peak":       round(ld_peak, 4),
            "logit_diff_ctrl":       round(ld_ctrl, 4),
            "logit_diff_random":     round(ld_rand, 4),
            "suppression_peak":      round(supp_peak, 4),
            "suppression_ctrl":      round(supp_ctrl, 4),
            "suppression_random":    round(supp_rand, 4),
        })

    if not probe_results:
        return {}

    mean_supp_peak = float(np.mean([r["suppression_peak"] for r in probe_results]))
    mean_supp_ctrl = float(np.mean([r["suppression_ctrl"] for r in probe_results]))
    mean_supp_rand = float(np.mean([r["suppression_random"] for r in probe_results]))
    peak_vs_ctrl_ratio = mean_supp_peak / max(abs(mean_supp_ctrl), 1e-4)
    peak_vs_rand_ratio = mean_supp_peak / max(abs(mean_supp_rand), 1e-4)
    peak_wins_ctrl = int(sum(r["suppression_peak"] > r["suppression_ctrl"]
                             for r in probe_results))

    log.info("  concept=%s  mean_supp: peak=%.3f  ctrl=%.3f  rand=%.3f  "
             "peak/ctrl_ratio=%.2f  peak_wins=%d/%d",
             concept, mean_supp_peak, mean_supp_ctrl, mean_supp_rand,
             peak_vs_ctrl_ratio, peak_wins_ctrl, len(probe_results))

    return {
        "concept":           concept,
        "caz_peak":          caz_peak,
        "ctrl_layer":        ctrl_layer,
        "n_probes":          len(probe_results),
        "mean_supp_peak":    round(mean_supp_peak, 4),
        "mean_supp_ctrl":    round(mean_supp_ctrl, 4),
        "mean_supp_random":  round(mean_supp_rand, 4),
        "peak_vs_ctrl_ratio": round(peak_vs_ctrl_ratio, 2),
        "peak_vs_rand_ratio": round(peak_vs_rand_ratio, 2),
        "peak_wins_ctrl":    peak_wins_ctrl,
        "n_valid_probes":    len(probe_results),
        "probes":            probe_results,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s — skipping", model_id)
        return

    model_slug = model_id.replace("/", "__")
    out_dir    = RESULTS_ROOT / "behavioral_pilot" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = [p["concept"] for p in BEHAVIORAL_PROBES]
    concepts = sorted(set(concepts))  # unique, alphabetical

    pending = [c for c in concepts
               if not (out_dir / f"behavioral_pilot_{c}.json").exists()
               or args.overwrite]
    if not pending:
        log.info("Already done: %s", model_id)
        return

    log.info("=== Behavioral pilot: %s ===", model_id)
    device = get_device(args.device)
    dtype  = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if vram_stats(device):
        log_vram(device)

    t_model_start = time.time()

    for concept in pending:
        out_path = out_dir / f"behavioral_pilot_{concept}.json"
        if out_path.exists() and not args.overwrite:
            continue

        concept_data = load_concept_data(extraction_dir, concept)
        if concept_data is None:
            log.warning("  No data for %s %s — skipping", model_id, concept)
            continue

        try:
            result = run_concept(model, tokenizer, concept, concept_data, device, args)
        except Exception as e:
            log.error("  %s %s failed: %s", model_id, concept, e)
            continue

        if not result:
            continue
        result["model_id"] = model_id
        out_path.write_text(json.dumps(result, indent=2))
        log.info("  Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    log.info("Done: %s (%.1fs)", model_id, time.time() - t_model_start)

    # Aggregate this model's results immediately
    aggregate_model(out_dir, model_id)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_model(out_dir: Path, model_id: str) -> dict:
    files = list(out_dir.glob("behavioral_pilot_*.json"))
    records = []
    for f in sorted(files):
        try:
            d = json.loads(f.read_text())
            if d:
                records.append(d)
        except Exception:
            continue

    if not records:
        return {}

    all_supp_peak = [r["mean_supp_peak"] for r in records]
    all_supp_ctrl = [r["mean_supp_ctrl"] for r in records]
    all_supp_rand = [r["mean_supp_random"] for r in records]
    all_wins      = [r["peak_wins_ctrl"] for r in records]
    all_n         = [r["n_valid_probes"] for r in records]
    total_wins    = sum(all_wins)
    total_probes  = sum(all_n)

    summary = {
        "model_id":              model_id,
        "n_concepts":            len(records),
        "mean_supp_peak":        round(float(np.mean(all_supp_peak)), 4),
        "mean_supp_ctrl":        round(float(np.mean(all_supp_ctrl)), 4),
        "mean_supp_random":      round(float(np.mean(all_supp_rand)), 4),
        "peak_vs_ctrl_ratio":    round(float(np.mean(all_supp_peak)) /
                                       max(abs(float(np.mean(all_supp_ctrl))), 1e-4), 2),
        "pct_peak_wins_ctrl":    round(100 * total_wins / max(total_probes, 1), 1),
        "total_probes":          total_probes,
        "by_concept":            {r["concept"]: {
            "supp_peak": r["mean_supp_peak"],
            "supp_ctrl": r["mean_supp_ctrl"],
            "supp_rand": r["mean_supp_random"],
            "wins":      r["peak_wins_ctrl"],
            "n":         r["n_valid_probes"],
        } for r in records},
    }

    summary_path = out_dir / "behavioral_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    _write_model_table(summary, out_dir / "behavioral_pilot_table.txt")
    log.info("Saved %s", summary_path)
    return summary


def _write_model_table(s: dict, out_path: Path) -> None:
    model_id = s["model_id"]
    lines = [
        f"Behavioral pilot — {model_id}",
        f"Peak-ablation vs non-CAZ control: mean suppression of concept-diagnostic logit difference",
        "",
        f"{'Concept':<16}  {'Supp(peak)':>11}  {'Supp(ctrl)':>11}  {'Supp(rand)':>11}  "
        f"{'Peak>Ctrl':>10}",
        "-" * 68,
    ]
    for concept, v in sorted(s["by_concept"].items()):
        lines.append(
            f"{concept:<16}  {v['supp_peak']:>11.4f}  {v['supp_ctrl']:>11.4f}  "
            f"{v['supp_rand']:>11.4f}  {v['wins']}/{v['n']:>2}"
        )
    lines += [
        "-" * 68,
        f"{'Overall':<16}  {s['mean_supp_peak']:>11.4f}  {s['mean_supp_ctrl']:>11.4f}  "
        f"{s['mean_supp_random']:>11.4f}  {s['pct_peak_wins_ctrl']:.0f}%",
        "",
        f"Peak/ctrl ratio: {s['peak_vs_ctrl_ratio']:.2f}×  "
        f"(N = {s['total_probes']} probes across {s['n_concepts']} concepts)",
        "Suppression = logit_diff(baseline) − logit_diff(ablated); higher = more suppression.",
        "Control: non-CAZ layer closest to model midpoint. Random: random unit vector at CAZ peak.",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral pilot: does CAZ peak ablation suppress concept token predictions?"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all",   action="store_true")

    parser.add_argument("--device",    choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype",     choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")

    args = parser.parse_args()

    if args.all:
        models = discover_models()
        log.info("Found %d models", len(models))
    else:
        models = [args.model]

    for model_id in models:
        run_model(model_id, args)


if __name__ == "__main__":
    main()
