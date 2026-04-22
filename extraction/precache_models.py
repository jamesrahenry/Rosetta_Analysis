#!/usr/bin/env python3
"""Pre-download model weights to HF cache for offline GPU runs.

Downloads only safetensors + config + tokenizer (no ONNX/TFLite/Flax).
Run locally where network is reliable, then scp the cache to GPU box.

Usage:
    python src/precache_models.py                # all models with results
    python src/precache_models.py --model EleutherAI/pythia-1.4b
    python src/precache_models.py --cache-dir /tmp/hf_cache
    python src/precache_models.py --dry-run      # list models, don't download

After downloading:
    # On local machine — tar just the model cache dirs
    tar czf hf_models.tar.gz -C ~/.cache/huggingface/hub .

    # SCP to GPU box
    scp hf_models.tar.gz coder@gpu-box:~

    # On GPU box — unpack into HF cache
    mkdir -p ~/.cache/huggingface/hub
    tar xzf ~/hf_models.tar.gz -C ~/.cache/huggingface/hub
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.txt",         # tokenizer files (merges.txt, vocab.txt)
    "*.model",       # sentencepiece models
    "*.tiktoken",
    "tokenizer*",
]


def discover_models() -> list[str]:
    models = set()
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if s.exists():
            with open(s) as f:
                models.add(json.load(f)["model_id"])
    return sorted(models)


def precache_model(model_id: str, cache_dir: str | None = None) -> float:
    """Download model weights + tokenizer. Returns elapsed seconds."""
    t0 = time.time()
    kwargs = {"allow_patterns": ALLOW_PATTERNS}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    print(f"  Downloading weights...", end=" ", flush=True)
    snapshot_download(model_id, **kwargs)
    print("done.", flush=True)

    print(f"  Downloading tokenizer...", end=" ", flush=True)
    tok_kwargs = {}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    print("done.", flush=True)

    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Pre-download HF models for offline GPU runs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, help="Single model to download")
    group.add_argument("--all", action="store_true", default=True, help="All models with results (default)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom HF cache directory")
    parser.add_argument("--dry-run", action="store_true", help="List models without downloading")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = discover_models()

    print(f"Models to cache: {len(models)}")
    if args.dry_run:
        for m in models:
            print(f"  {m}")
        return

    total_t = 0
    failed = []
    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_id}")
        try:
            elapsed = precache_model(model_id, args.cache_dir)
            total_t += elapsed
            print(f"  ✓ {elapsed:.0f}s")
        except Exception as e:
            print(f"  ✗ {e}")
            failed.append(model_id)

    print(f"\nDone: {len(models) - len(failed)}/{len(models)} cached in {total_t:.0f}s")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
