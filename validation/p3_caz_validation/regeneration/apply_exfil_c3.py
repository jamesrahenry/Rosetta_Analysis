#!/usr/bin/env python3
"""Apply-pass: §6.4 / abstract 49.4% legibility-causality divergence + C3
within-cell permutation null, defective vs corrected exfiltration.

Per multimodal cell (>=2 CAZ regions), from ablation_multimodal_*.json:
  score_region  = argmax(caz_score)          -- what the detector calls dominant
  causal_region = argmin(self_retained_pct)  -- what ablation says matters most
  divergence    = (score_region != causal_region)
  causal_deeper = depth(causal) > depth(score)     [among divergent cells]
  gap_pp        = self_retained(score) - self_retained(causal)  [divergent cells]
Permutation null: within each cell shuffle self_retained across regions,
recompute divergence rate; 10000 draws. chance ~ 1 - mean(1/k).

Validation: defective state must reproduce p3_c3_results.json
(0.4944 / causal_deeper 0.9382 / gap 41.33 / null 0.577+-0.026 / z -3.22).
Corrected state drops the 2 not_applicable exfiltration cells (correction
pushed them below the >=2-region threshold): population 360 -> 358.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path.home() / "rosetta_data/paper_n250"
DEF = Path.home() / "rosetta_data/_defective_exfiltration"
CONCEPTS = ["credibility","negation","causation","temporal_order","sentiment","certainty",
"moral_valence","specificity","plurality","agency","formality","threat_severity",
"authorization","urgency","sarcasm","deception","exfiltration"]
# corrected: these two exfiltration cells fall below >=2 regions (manifest not_applicable)
CORRECTED_DROP = {("google_gemma_2_9b" if False else s) for s in []}  # placeholder
NOT_APPLICABLE_EXFIL = {"Qwen_Qwen2.5_14B", "facebook_opt_6.7b"}
MODELS = sorted(d.name for d in ROOT.iterdir() if d.is_dir()
                and not d.name.startswith("_")
                and not any(x in d.name for x in ["Instruct","gpt_neo","32B","72B","70B","falcon","Llama_3.1"]))


def load_cells(state):
    cells = []
    for m in MODELS:
        for c in CONCEPTS:
            defect = (c == "exfiltration")
            if state == "corrected" and c == "exfiltration" and m in NOT_APPLICABLE_EXFIL:
                continue  # cell no longer exists post-correction
            base = DEF / m if (state == "defective" and defect) else ROOT / m
            f = base / f"ablation_multimodal_{c}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            regs = d.get("cazs", [])
            if len(regs) < 2:
                continue
            scores = np.array([r["caz_score"] for r in regs])
            retained = np.array([r["self_retained_pct"] for r in regs])
            depths = np.array([r["depth_pct"] for r in regs])
            cells.append((scores, retained, depths))
    return cells


def analyze(state, seed=0):
    cells = load_cells(state)
    div, deeper, gaps, ks = [], [], [], []
    for scores, retained, depths in cells:
        si = int(np.argmax(scores))
        ci = int(np.argmin(retained))
        d = si != ci
        div.append(d)
        ks.append(len(scores))
        if d:
            deeper.append(depths[ci] > depths[si])
            gaps.append(retained[si] - retained[ci])
    div = np.array(div)
    rate = div.mean()
    # permutation null: shuffle retained within each cell
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(10000):
        cnt = 0
        for scores, retained, depths in cells:
            si = int(np.argmax(scores))
            # reassign self_retained values to region slots at random, then argmin
            shuffled = retained[rng.permutation(len(retained))]
            ci = int(np.argmin(shuffled))
            cnt += (si != ci)
        null.append(cnt / len(cells))
    null = np.array(null)
    chance = np.mean([1 - 1/k for k in ks])
    z = (rate - null.mean()) / null.std()
    res = dict(state=state, n_cells=len(cells), divergence_rate=float(rate),
               causal_deeper_rate=float(np.mean(deeper)), mean_gap_pp=float(np.mean(gaps)),
               theoretical_chance=float(chance), null_mean=float(null.mean()),
               null_std=float(null.std()), z=float(z),
               p_below=float((null <= rate).mean()))
    print(f"{state:10s} n={len(cells)} div={rate:.4f} deeper={np.mean(deeper):.4f} "
          f"gap={np.mean(gaps):.2f} chance={chance:.4f} null={null.mean():.4f}+-{null.std():.4f} "
          f"z={z:.2f} p={(null<=rate).mean():.4f}")
    return res


if __name__ == "__main__":
    out = {"job": "C3/§6.4 divergence + null, defective vs corrected"}
    out["defective"] = analyze("defective")
    out["corrected"] = analyze("corrected")
    import datetime
    out["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    Path(__file__).parent.joinpath("results/apply_exfil_c3.json").write_text(json.dumps(out, indent=1))
    print("saved results/apply_exfil_c3.json")
