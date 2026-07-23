#!/usr/bin/env python3
"""Recompute the §6.7 "coasting fraction" from the CKA artifacts. tc4fd04e item (5).

The published figure — "75.6% of model depth is classified as coasting" — is a
transcription of the pre-correction multimodal rate (360/476 = 75.6%), NOT a
CKA-derived quantity: no detrend threshold in 0.001–0.02 reproduces it (the real
coasting fraction is stable at 78–83%). This script recomputes it faithfully with
the exact analyze_coasting.py labeling.

Method (analyze_coasting.py): detrend adjacent-layer CKA (subtract a
uniform_filter1d(size=5) baseline); around each Fisher region peak, the CKA-CAZ
extent is the contiguous run where detrended CKA < -threshold (default 0.003);
every layer outside all extents is "coasting". Fraction = coasting layers / depth.

Data: paper_n250/<model>/cka_<concept>.json (cka_adjacent, caz_regions, n_layers)
on HF james-ra-henry/Rosetta-Activations — 26 models (CKA extraction predates the
pythia-12b / Qwen2.5-14B roster additions). Auto-downloads the JSONs if absent.
CPU-only. Written: 2026-07-23 UTC.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

DATA = Path.home() / "rosetta_data" / "paper_n250"


def ensure_cka():
    if glob.glob(str(DATA / "*" / "cka_*.json")):
        return
    from huggingface_hub import snapshot_download
    snapshot_download("james-ra-henry/Rosetta-Activations", repo_type="dataset",
                      allow_patterns=["paper_n250/*/cka_*.json"],
                      local_dir=str(Path.home() / "rosetta_data"))


def detrend(cka, w=5):
    a = np.array(cka, dtype=np.float64)
    return a - uniform_filter1d(a, size=w, mode="nearest")


def coasting_labels(cka_adj, fisher_peaks, n_layers, threshold=0.003):
    det = detrend(cka_adj)
    n = len(cka_adj)
    lab = ["coasting"] * n_layers
    for peak in fisher_peaks:
        p = max(0, min(peak - 1, n - 1))
        lo, hi = max(0, p - 3), min(n, p + 4)
        lmi = lo + int(np.argmin(det[lo:hi]))
        if float(det[lmi]) < -threshold:
            s = lmi
            while s > 0 and det[s - 1] < -threshold:
                s -= 1
            e = lmi
            while e < n - 1 and det[e + 1] < -threshold:
                e += 1
        else:
            s = e = lmi
        for l in range(s, e + 2):          # cka_layer_end = e+1, label_layers is inclusive
            if 0 <= l < n_layers:
                lab[l] = "caz"
    return lab


def fraction(threshold=0.003):
    files = [f for f in glob.glob(str(DATA / "*" / "cka_*.json")) if "cka_acts" not in f]
    tot_coast = tot_layers = 0
    cells = []
    for f in files:
        d = json.loads(Path(f).read_text())
        cka, regs = d.get("cka_adjacent", []), d.get("caz_regions", [])
        nL = d.get("n_layers", len(cka) + 1)
        if not cka or not regs:
            continue
        lab = coasting_labels(cka, [r["peak"] for r in regs], nL, threshold)
        c = lab.count("coasting")
        tot_coast += c
        tot_layers += nL
        cells.append(c / nL)
    return {"n_cells": len(cells), "n_models": len({Path(f).parent.name for f in files}),
            "pooled_pct": round(100 * tot_coast / tot_layers, 1),
            "per_cell_mean_pct": round(100 * float(np.mean(cells)), 1)}


def main():
    ensure_cka()
    base = fraction(0.003)
    sweep = {th: fraction(th)["per_cell_mean_pct"] for th in (0.001, 0.003, 0.01, 0.02)}
    out = {**base, "threshold_sweep_per_cell_mean": sweep,
           "published_erroneous": 75.6, "note": "75.6 = 360/476 old multimodal rate, not CKA-derived"}
    print(json.dumps(out, indent=2))
    Path(__file__).parent.joinpath("coasting_fraction_recompute_results.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
