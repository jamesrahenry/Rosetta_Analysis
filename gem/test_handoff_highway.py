#!/usr/bin/env python3
"""Handoff Highway Tests — Short-circuit and injection experiments on UF001/F003.

Two experiments on Pythia-6.9b F003 (the causation→certainty→credibility highway):

Test 1: Short-Circuit
  Feed a sentence with strong causation but deliberately subverted certainty.
  Does F003 light up normally in the causation phase, then crash at the
  certainty handoff layer because the expected payload wasn't there?

Test 2: Injection
  Take a neutral sentence with no causal or epistemic content.
  Manually add the F003 direction to the residual stream at the post-handoff
  layer (~L20). Does the model's output become more certain/credible-sounding
  even though the input had no such content?

Usage:
    python src/test_handoff_highway.py \
        --model EleutherAI/pythia-6.9b \
        --library feature_library/ \
        --results results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.gpu_utils import get_device, get_dtype, release_model, log_vram

log = logging.getLogger(__name__)

# Test sentences for short-circuit experiment
SHORT_CIRCUIT_SENTENCES = [
    # Strong causation, subverted certainty
    "Because the engine failed, the crash was possibly, though not definitely, inevitable.",
    "The drought caused crop failures, but whether this will lead to famine remains deeply uncertain.",
    "Since the policy changed, unemployment rose — or so it seems, though the data is inconclusive.",
    # Strong causation, strong certainty (control)
    "Because the bridge collapsed, the road is absolutely impassable.",
    "The vaccine caused immunity — this is definitively proven by the clinical data.",
    # Neutral (baseline)
    "The cat sat on the mat near the window.",
    "She walked slowly through the park in the afternoon.",
]

INJECTION_SENTENCES = [
    "The cat sat on the mat.",
    "She walked slowly through the park.",
    "The weather today is mild.",
    "A book was left on the table.",
]


def load_f003_direction(library_dir: Path, model_id: str, layer: int) -> np.ndarray | None:
    """Load F003's direction vector at a specific layer from the feature library."""
    slug = model_id.split("/")[-1]
    directions_dir = library_dir / "models" / slug / "directions"
    npy_path = directions_dir / f"directions_L{layer:03d}.npy"
    if not npy_path.exists():
        log.warning("No directions file for layer %d", layer)
        return None
    dirs = np.load(npy_path)
    # F003 uses PC index 0 at all layers (from feature_map.json pc_indices)
    return dirs[0].astype(np.float64)


def get_residual_stream(model, tokenizer, text: str, device: str,
                        layers_of_interest: list[int]) -> dict[int, np.ndarray]:
    """Forward pass collecting residual stream at specified layers."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    hidden_states = outputs.hidden_states
    # Pool: last non-padding token
    length = attention_mask.sum(dim=1) - 1
    result = {}
    for layer in layers_of_interest:
        if layer < len(hidden_states):
            hs = hidden_states[layer]  # [1, seq, hidden]
            pooled = hs[0, length[0]].cpu().float().numpy()
            result[layer] = pooled

    return result


def project_onto_direction(activation: np.ndarray, direction: np.ndarray) -> float:
    """Cosine similarity between activation and feature direction."""
    a_norm = np.linalg.norm(activation)
    d_norm = np.linalg.norm(direction)
    if a_norm < 1e-12 or d_norm < 1e-12:
        return 0.0
    return float(np.dot(activation / a_norm, direction / d_norm))


def run_short_circuit_test(model, tokenizer, f003_directions: dict[int, np.ndarray],
                           device: str, handoff_layer: int):
    """Test 1: Does F003 crash at the handoff layer when causation is present but certainty is subverted?"""
    log.info("\n=== SHORT-CIRCUIT TEST ===")
    log.info("Handoff layer: L%d", handoff_layer)
    log.info("Hypothesis: F003 crashes post-handoff when certainty is subverted\n")

    layers = sorted(f003_directions.keys())

    print(f"{'Sentence':>60s}  ", end="")
    for l in [5, 10, handoff_layer, 25, 30]:
        if l in f003_directions:
            print(f" L{l:02d}", end="")
    print()
    print("-" * 100)

    results = {}
    for sentence in SHORT_CIRCUIT_SENTENCES:
        acts = get_residual_stream(model, tokenizer, sentence, device, list(f003_directions.keys()))
        projections = {}
        for layer, direction in f003_directions.items():
            if layer in acts:
                projections[layer] = project_onto_direction(acts[layer], direction)
        results[sentence] = projections

        label = sentence[:60]
        print(f"{label:>60s}  ", end="")
        for l in [5, 10, handoff_layer, 25, 30]:
            if l in projections:
                val = projections[l]
                bar = "+" if val > 0.1 else ("-" if val < -0.1 else " ")
                print(f" {val:+.3f}", end="")
        print()

    # Analysis: do subverted-certainty sentences crash post-handoff?
    print("\n--- Post-handoff mean F003 activation ---")
    post_layers = [l for l in layers if l >= handoff_layer + 2]
    for sentence in SHORT_CIRCUIT_SENTENCES:
        post_mean = np.mean([results[sentence][l] for l in post_layers if l in results[sentence]])
        label = sentence[:60]
        print(f"  {label:>60s}  post-handoff mean: {post_mean:+.4f}")

    return results


def run_injection_test(model, tokenizer, f003_directions: dict[int, np.ndarray],
                       device: str, injection_layer: int, injection_scale: float = 2.0):
    """Test 2: Does injecting F003 into a neutral sentence make output more certain/credible?"""
    log.info("\n=== INJECTION TEST ===")
    log.info("Injecting F003 direction at L%d (scale=%.1f)", injection_layer, injection_scale)
    log.info("Hypothesis: neutral sentences sound more certain/credible after injection\n")

    if injection_layer not in f003_directions:
        log.warning("No F003 direction at L%d", injection_layer)
        return

    direction = f003_directions[injection_layer]
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    results = {}
    for sentence in INJECTION_SENTENCES:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        baseline_logits = None
        injected_logits = None

        # Hook to inject at the target layer
        injection_done = [False]

        def inject_hook(module, input, output):
            if not injection_done[0]:
                injection_done[0] = True
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                # Add F003 direction scaled to all token positions
                d = direction_tensor.to(hs.device).to(hs.dtype)
                d = d / (d.norm() + 1e-12)
                hs = hs + injection_scale * d.unsqueeze(0).unsqueeze(0)
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return output

        # Baseline
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            baseline_logits = out.logits[0, -1, :].cpu()

        # Injected
        injection_done[0] = False
        try:
            layers_attr = None
            for attr in ["gpt_neox", "transformer", "model"]:
                if hasattr(model, attr):
                    layers_attr = getattr(model, attr)
                    break
            if layers_attr and hasattr(layers_attr, "layers"):
                hook_target = layers_attr.layers[injection_layer]
            elif layers_attr and hasattr(layers_attr, "h"):
                hook_target = layers_attr.h[injection_layer]
            else:
                log.warning("Couldn't find layer %d for injection hook", injection_layer)
                continue

            handle = hook_target.register_forward_hook(inject_hook)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                injected_logits = out.logits[0, -1, :].cpu()
            handle.remove()
        except Exception as e:
            log.warning("Injection hook failed: %s", e)
            continue

        # Decode top-5 tokens from baseline vs injected
        baseline_top = torch.topk(baseline_logits, 5)
        injected_top = torch.topk(injected_logits, 5)

        baseline_tokens = [tokenizer.decode([t]) for t in baseline_top.indices]
        injected_tokens = [tokenizer.decode([t]) for t in injected_top.indices]

        print(f"\n  Input: '{sentence}'")
        print(f"  Baseline next tokens: {baseline_tokens}")
        print(f"  Injected next tokens: {injected_tokens}")

        # Check if injected output has more epistemic/certainty markers
        epistemic_markers = {"absolutely", "certainly", "definitely", "clearly", "obviously",
                            "proven", "undeniably", "confirmed", "established", "demonstrates"}
        base_ep = sum(1 for t in baseline_tokens if any(m in t.lower() for m in epistemic_markers))
        inj_ep = sum(1 for t in injected_tokens if any(m in t.lower() for m in epistemic_markers))
        print(f"  Epistemic markers: baseline={base_ep}, injected={inj_ep} {'← MORE CERTAIN' if inj_ep > base_ep else ''}")
        results[sentence] = {"baseline": baseline_tokens, "injected": injected_tokens}

    return results


def main():
    parser = argparse.ArgumentParser(description="Handoff Highway Tests — F003 short-circuit and injection")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-6.9b")
    parser.add_argument("--library", type=Path, required=True, help="Path to feature_library/")
    parser.add_argument("--results", type=Path, required=True, help="Path to results/ dir")
    parser.add_argument("--feature-id", type=int, default=3, help="Feature to test (default: F003)")
    parser.add_argument("--handoff-layer", type=int, default=9,
                        help="Layer where causation->certainty handoff occurs (default: 9)")
    parser.add_argument("--injection-layer", type=int, default=20,
                        help="Layer to inject F003 direction (default: 20, post-handoff)")
    parser.add_argument("--injection-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = get_device(args.device)
    dtype = get_dtype(device)

    # Load F003 directions at key layers
    slug = args.model.split("/")[-1]
    directions_dir = args.library / "models" / slug / "directions"

    # Load at all available layers for short-circuit test
    f003_directions = {}
    if directions_dir.exists():
        for npy_path in sorted(directions_dir.glob("directions_L*.npy")):
            layer = int(npy_path.stem.split("_L")[1])
            dirs = np.load(npy_path)
            if len(dirs) > 0:
                f003_directions[layer] = dirs[0].astype(np.float64)  # PC0 = dominant direction
    else:
        log.error("No directions found at %s", directions_dir)
        return

    log.info("Loaded F003 directions for %d layers", len(f003_directions))

    # Load model
    log.info("Loading %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModel.from_pretrained(args.model, dtype=dtype, device_map=device)
    except (ValueError, ImportError):
        model = AutoModel.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    log_vram("after model load")

    # Run tests
    run_short_circuit_test(model, tokenizer, f003_directions, device, args.handoff_layer)
    run_injection_test(model, tokenizer, f003_directions, device, args.injection_layer, args.injection_scale)

    release_model(model)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
