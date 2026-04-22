"""
viz_style.py — Shared style constants and helpers for all Rosetta viz scripts.

Style guide and rationale: Rosetta_Program/VIZ_STYLE_GUIDE.md
That document is the canonical reference; this file is the implementation.

Import at the top of any viz script:

    from viz_style import (
        concept_color,
        CONCEPT_COLORS, CONCEPT_TYPE, CONCEPTS,
        CONCEPT_COLORS_ACCESSIBLE,
        FAMILY_COLORS, FAMILY_MAP, FAMILY_ORDER,
        CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat,
        THEME, apply_theme, layer_ticks, model_label, sort_models,
        add_outside_callouts,
    )

Design decisions (2026-04-11, updated 2026-04-11):
  - White background, print-ready (PDF-safe)
  - 18 named concept colors, deep/saturated, max-separation placement
  - Unlimited concept support via concept_color() hash fallback (SAE-scale)
  - Paul Tol accessible palette available as CONCEPT_COLORS_ACCESSIBLE
  - Max 8 concepts per single plot panel
  - Architecture family colors are distinct and legible at small sizes
  - X-axis: layer count (L0, L7...) with % in parentheses — both references
  - Callouts go OUTSIDE the plot area with straight vertical lines
  - Callout placement uses 4 slots (2 above × 2 below) to avoid overlap

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import hashlib
import colorsys
from pathlib import Path
from typing import Any

import numpy as np

# ── Concept palette ────────────────────────────────────────────────────────────
# 18 named concepts, deep/saturated (L≈0.34, S≈0.76), max-separation placement.
# First 7 are pinned from the original spec; extended 11 placed by greedy
# maximum-minimum-distance algorithm across the hue wheel.
#
# For any concept not in this dict, call concept_color(name) — it produces a
# stable deterministic color from the same aesthetic (SAE-scale safe).

CONCEPT_COLORS: dict[str, str] = {
    # ── Core 7 (pinned) ───────────────────────────────────────────
    "credibility":    "#7B1FA2",   # deep purple       H≈282°
    "certainty":      "#AD1457",   # deep pink         H≈334°
    "negation":       "#C62828",   # deep red          H≈0°
    "causation":      "#E65100",   # deep orange       H≈21°
    "temporal_order": "#827717",   # olive             H≈54°
    "sentiment":      "#2E7D32",   # forest green      H≈123°
    "moral_valence":  "#00695C",   # teal              H≈173°
    # ── Extended 11 (algorithm-placed) ────────────────────────────
    "sarcasm":        "#986714",   # bronze-brown      H≈38°
    "plurality":      "#809814",   # yellow-olive      H≈71°
    "exfiltration":   "#599814",   # lime-green        H≈89°
    "agency":         "#339814",   # medium green      H≈106°
    "threat_severity":"#149852",   # teal-green        H≈148°
    "formality":      "#148A98",   # cyan              H≈187°
    "obfuscation":    "#146C98",   # steel blue        H≈200°
    "specificity":    "#144F98",   # medium blue       H≈214°
    "authorization":  "#143098",   # strong blue       H≈228°
    "deception":      "#351498",   # deep indigo       H≈255°
    "urgency":        "#981487",   # deep magenta      H≈308°
}

CONCEPT_TYPE: dict[str, str] = {
    "credibility":    "epistemic",
    "certainty":      "epistemic",
    "deception":      "epistemic",
    "sarcasm":        "epistemic",
    "specificity":    "epistemic",
    "formality":      "epistemic",
    "negation":       "syntactic",
    "plurality":      "syntactic",
    "causation":      "relational",
    "temporal_order": "relational",
    "agency":         "relational",
    "sentiment":      "affective",
    "moral_valence":  "affective",
    "urgency":        "affective",
    "threat_severity":"affective",
    "authorization":  "security",
    "exfiltration":   "security",
    "obfuscation":    "security",
}

CONCEPTS = list(CONCEPT_COLORS.keys())

# ── Accessible palette (Paul Tol — colorblind-safe) ────────────────────────────
# Secondary palette. Recommended when colorblind accessibility is required,
# or when explicitly requested. Best for ≤ 8 concepts per plot.
# Source: https://personal.sron.nl/~pault/
# Bright (7) + Muted (10) combined; grey entries dropped as print-unsafe.

CONCEPT_COLORS_ACCESSIBLE: dict[str, str] = {
    "credibility":    "#AA3377",   # Tol Bright purple-pink
    "certainty":      "#882255",   # Tol Muted dark plum
    "negation":       "#CC6677",   # Tol Muted muted red
    "causation":      "#EE7733",   # Tol Bright orange
    "temporal_order": "#CCBB44",   # Tol Muted dark yellow
    "sentiment":      "#228833",   # Tol Bright green
    "moral_valence":  "#44AA99",   # Tol Muted teal
    "sarcasm":        "#EE6677",   # Tol Bright pink-red
    "plurality":      "#999933",   # Tol Muted olive
    "exfiltration":   "#117733",   # Tol Muted dark green
    "agency":         "#66CCEE",   # Tol Bright sky blue
    "threat_severity":"#CC3311",   # Tol Vibrant dark red
    "formality":      "#009988",   # Tol Vibrant deep teal
    "obfuscation":    "#0077BB",   # Tol Vibrant medium blue
    "specificity":    "#4477AA",   # Tol Bright blue
    "authorization":  "#332288",   # Tol Muted deep indigo
    "deception":      "#AA4499",   # Tol Muted medium purple
    "urgency":        "#DDCC77",   # Tol Muted light yellow
}


def concept_color(name: str, *, accessible: bool = False) -> str:
    """Return the canonical color for any concept name.

    For the 18 named concepts, returns the fixed palette entry.
    For any other name (SAE features, novel concepts, etc.), derives a
    stable color via MD5 hash — same name always produces the same color,
    in the same deep/saturated aesthetic (L=0.34, S=0.76).

    Parameters
    ----------
    name:
        Concept identifier string.
    accessible:
        If True, use the Paul Tol colorblind-safe palette for named concepts.
        Hash fallback aesthetic is unchanged regardless of this flag.
    """
    palette = CONCEPT_COLORS_ACCESSIBLE if accessible else CONCEPT_COLORS
    if name in palette:
        return palette[name]
    # Hash fallback: deterministic, same L/S as the named palette
    h_int = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue   = (h_int % 3600) / 3600.0   # 0.1° resolution
    r, g, b = colorsys.hls_to_rgb(hue, 0.34, 0.76)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

# ── Architecture family palette ────────────────────────────────────────────────

FAMILY_ORDER = ["Pythia", "OPT", "GPT-2", "Qwen", "Llama", "Mistral", "Gemma", "Phi", "Other"]

FAMILY_COLORS: dict[str, str] = {
    "Pythia":  "#1565C0",   # strong blue
    "OPT":     "#6A1B9A",   # purple
    "GPT-2":   "#558B2F",   # olive green
    "Qwen":    "#E65100",   # deep orange
    "Llama":   "#AD1457",   # deep pink
    "Mistral": "#0277BD",   # light blue
    "Gemma":   "#00695C",   # teal
    "Phi":     "#4E342E",   # brown
    "Other":   "#546E7A",   # blue-grey
}

# (family, param_count_M) — param count used for within-family sort
FAMILY_MAP: dict[str, tuple[str, int]] = {
    "EleutherAI/pythia-70m":           ("Pythia",  70),
    "EleutherAI/pythia-160m":          ("Pythia",  160),
    "EleutherAI/pythia-410m":          ("Pythia",  410),
    "EleutherAI/pythia-1b":            ("Pythia",  1000),
    "EleutherAI/pythia-1.4b":          ("Pythia",  1400),
    "EleutherAI/pythia-2.8b":          ("Pythia",  2800),
    "EleutherAI/pythia-6.9b":          ("Pythia",  6900),
    "facebook/opt-125m":               ("OPT",     125),
    "facebook/opt-350m":               ("OPT",     350),
    "facebook/opt-1.3b":               ("OPT",     1300),
    "facebook/opt-2.7b":               ("OPT",     2700),
    "facebook/opt-6.7b":               ("OPT",     6700),
    "openai-community/gpt2":           ("GPT-2",   117),
    "openai-community/gpt2-medium":    ("GPT-2",   345),
    "openai-community/gpt2-large":     ("GPT-2",   800),
    "openai-community/gpt2-xl":        ("GPT-2",   1500),
    "Qwen/Qwen2.5-0.5B":              ("Qwen",    500),
    "Qwen/Qwen2.5-1.5B":              ("Qwen",    1500),
    "Qwen/Qwen2.5-3B":                ("Qwen",    3000),
    "Qwen/Qwen2.5-7B":                ("Qwen",    7000),
    "meta-llama/Llama-3.2-1B":         ("Llama",   1000),
    "meta-llama/Llama-3.2-3B":         ("Llama",   3000),
    "mistralai/Mistral-7B-v0.3":       ("Mistral", 7000),
    "google/gemma-2-2b":               ("Gemma",   2000),
    "google/gemma-2-9b":               ("Gemma",   9000),
    "microsoft/phi-2":                 ("Phi",     2700),
}

# ── CAZ score category palette ────────────────────────────────────────────────
# Visual encoding for CAZ strength categories.
# Edge/marker colors are saturated; fill colors are very light for band backgrounds.
# Import via: from viz_style import CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat

CAZ_CAT_COLORS: dict[str, str] = {
    "black_hole": "#C62828",   # dark red     — score > 0.5
    "strong":     "#E65100",   # deep orange  — score > 0.2
    "moderate":   "#F9A825",   # amber        — score > 0.05
    "gentle":     "#1565C0",   # dark blue    — score ≤ 0.05
}

CAZ_CAT_FILL: dict[str, str] = {
    "black_hole": "#FFCDD2",   # very light red
    "strong":     "#FFE0B2",   # very light orange
    "moderate":   "#FFF9C4",   # very light amber
    "gentle":     "#BBDEFB",   # very light blue
}

CAZ_CAT_LABELS: dict[str, str] = {
    "black_hole": "Black hole",
    "strong":     "Strong",
    "moderate":   "Moderate",
    "gentle":     "Gentle",
}


def caz_score_cat(score: float) -> str:
    """Return the CAZ strength category string for a given score value.

    Thresholds:  > 0.5  → black_hole
                 > 0.2  → strong
                 > 0.05 → moderate
                 ≤ 0.05 → gentle
    """
    if score > 0.5:  return "black_hole"
    if score > 0.2:  return "strong"
    if score > 0.05: return "moderate"
    return "gentle"


# ── Theme constants ────────────────────────────────────────────────────────────

THEME = {
    "bg":        "white",
    "panel_bg":  "white",
    "grid":      "#e8e8e8",
    "spine":     "#cccccc",
    "text":      "#111111",
    "dim":       "#555555",
    "cka_line":  "#1565C0",   # CKA adjacent-layer curve
    "coh_line":  "#546E7A",   # Coherence C(l) secondary axis
    "annot":     "#222222",   # generic annotation / callout text
}


# ── Axis helpers ───────────────────────────────────────────────────────────────

def layer_ticks(n_layers: int, pcts: tuple[int, ...] = (0, 25, 50, 75, 100)):
    """
    Return (tick_positions, tick_labels) for an x-axis showing both layer
    count and depth percentage.

    Labels format: "L{n}\\n({pct}%)" so the layer number is primary and
    the percentage is secondary, readable at a glance without model metadata.
    """
    positions = [int(p / 100 * (n_layers - 1)) for p in pcts]
    labels    = [f"L{l}\n({p}%)" for l, p in zip(positions, pcts)]
    return positions, labels


def apply_theme(ax, ax_twin=None, *, grid: bool = True):
    """
    Apply standard white-background theme to a matplotlib Axes (and optional
    twin). Call once per panel after creating axes.
    """
    ax.set_facecolor(THEME["panel_bg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["dim"], labelsize=7.5, length=3, width=0.7)
    if grid:
        ax.grid(True, color=THEME["grid"], linewidth=0.6, alpha=1.0, zorder=0)

    if ax_twin is not None:
        ax_twin.set_facecolor(THEME["panel_bg"])
        for spine in ax_twin.spines.values():
            spine.set_edgecolor(THEME["spine"])
            spine.set_linewidth(0.8)
        ax_twin.tick_params(
            colors=THEME["cka_line"], labelsize=7, length=3, width=0.7,
        )


# ── Model helpers ──────────────────────────────────────────────────────────────

def model_label(model_id: str) -> str:
    """
    Short human-readable label, e.g. 'Pythia-1.4B', 'Qwen-3B'.
    Falls back to the bare HuggingFace slug if not in FAMILY_MAP.
    """
    family, _ = FAMILY_MAP.get(model_id, ("", 0))
    short     = model_id.split("/")[-1]
    for prefix in ("pythia-", "opt-", "gpt2-", "Qwen2.5-", "Llama-3.2-",
                   "Mistral-", "gemma-2-", "phi-"):
        if short.lower().startswith(prefix.lower()):
            size = short[len(prefix):]
            return f"{family}-{size.upper()}"
    return short


def sort_models(model_ids: list[str]) -> list[str]:
    """Sort model IDs by family order, then by parameter count."""
    def key(mid):
        fam, params = FAMILY_MAP.get(mid, ("Other", 0))
        fam_idx     = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        return (fam_idx, params)
    return sorted(model_ids, key=key)


# ── Outside-axes callout system ────────────────────────────────────────────────

def _assign_callout_slots(
    callouts: list[dict],
    n_layers: int,
) -> list[tuple[str, float]]:
    """
    Assign each callout to one of four vertical slots outside the axes area.

    Slots (tried in priority order):
      top-near   y =  1.09   (above, closest to axes)
      bot-near   y = -0.10   (below, closest to axes)
      top-far    y =  1.22   (above, second row)
      bot-far    y = -0.24   (below, second row)

    Greedy: iterate callouts in x order, place in first non-colliding slot.
    Returns list of (side, y_frac) in original callout order.
    """
    if not callouts:
        return []

    # Rough character-width in layer units (7px/char, ~787px wide axes at 13in)
    char_w = 7.0 * n_layers / 787.0

    SLOTS = [
        ("top",    1.09),
        ("bottom", -0.10),
        ("top",    1.22),
        ("bottom", -0.24),
    ]

    sorted_idx = sorted(range(len(callouts)), key=lambda i: callouts[i]["x"])
    slot_right = [-999.0] * len(SLOTS)
    result     = [None]   * len(callouts)

    for orig_i in sorted_idx:
        c        = callouts[orig_i]
        x        = c["x"]
        max_line = max(len(ln) for ln in c["label"].split("\n"))
        half_w   = max_line * char_w / 2.0 + 1.0

        placed = False
        for s_i, (side, y_frac) in enumerate(SLOTS):
            if (x - half_w) > slot_right[s_i]:
                result[orig_i]  = (side, y_frac)
                slot_right[s_i] = x + half_w
                placed          = True
                break

        if not placed:
            best             = min(range(len(SLOTS)), key=lambda i: slot_right[i])
            result[orig_i]   = SLOTS[best]
            slot_right[best] = x + half_w

    return result


def add_outside_callouts(
    ax,
    callouts: list[dict],
    n_layers: int,
):
    """
    Draw outside-the-axes callouts: straight vertical lines from a label box
    placed above or below the axes down to the data point.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to annotate. For CKA-axis callouts, pass ax_cka instead.
    callouts : list of dicts, each with keys:
        x       : float  — data x position (layer number)
        y       : float  — data y position (in ax's coordinate system)
        label   : str    — text (may include \\n for multi-line)
        color   : str    — hex color for line and label border
        bold    : bool   — optional; make text bold (default True)
    n_layers : int
        Total number of layers (used to estimate label widths).

    Example
    -------
    add_outside_callouts(ax, [
        {"x": 4,  "y": 0.8, "label": "Fisher CAZ extent\\n(velocity threshold)",
         "color": "#7B1FA2"},
        {"x": 14, "y": 0.95, "label": "Velocity peak\\n(max assembly rate)",
         "color": "#222222"},
    ], n_layers=24)
    """
    slots = _assign_callout_slots(callouts, n_layers)

    for pt, (side, y_label) in zip(callouts, slots):
        va   = "bottom" if side == "top" else "top"
        bold = pt.get("bold", True)

        ax.annotate(
            pt["label"],
            xy=(pt["x"], pt["y"]),
            xycoords="data",
            xytext=(pt["x"], y_label),
            textcoords=("data", "axes fraction"),
            ha="center",
            va=va,
            color=pt["color"],
            fontsize=7.5,
            fontweight="bold" if bold else "normal",
            clip_on=False,
            arrowprops=dict(
                arrowstyle="-",
                color=pt["color"],
                lw=0.9,
                shrinkA=4,
                shrinkB=3,
            ),
            bbox=dict(
                boxstyle="round,pad=0.28",
                fc="white",
                ec=pt["color"],
                lw=0.8,
                alpha=0.97,
            ),
            zorder=20,
        )
