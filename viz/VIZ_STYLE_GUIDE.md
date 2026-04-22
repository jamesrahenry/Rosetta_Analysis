# Visualization Style Guide

*Written: 2026-04-11 UTC — Updated: 2026-04-11 UTC*

Program-wide conventions for all Rosetta visualization scripts.

**Reference implementation**: [`caz_scaling/src/viz_style.py`](caz_scaling/src/viz_style.py)
— this document is the spec; that file is the code. If they ever disagree, fix
the code to match the guide, then note the update here.

---

All visualization scripts in `caz_scaling/src/` share a common style imported
from `viz_style.py`. This document explains the conventions and how to use the
shared module when writing new figures.

---

## Quick Start

```python
from viz_style import (
    concept_color,
    CONCEPT_COLORS, CONCEPT_TYPE, CONCEPTS,
    CONCEPT_COLORS_ACCESSIBLE,
    FAMILY_COLORS, FAMILY_MAP, FAMILY_ORDER,
    CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat,
    THEME, apply_theme, layer_ticks, model_label, sort_models,
    add_outside_callouts,
)
```

---

## Core Decisions

### Background: White

All figures use a white background (`facecolor="white"`). Dark backgrounds
look good on screens but wash out in PDFs and break in printed papers.

```python
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
```

### Concept Colors

18 named concepts, deep/saturated (L≈0.34, S≈0.76), placed for maximum hue
separation. First 7 are pinned from the original spec.

| Concept | Hex | Type |
|---|---|---|
| credibility | `#7B1FA2` | epistemic |
| certainty | `#AD1457` | epistemic |
| negation | `#C62828` | syntactic |
| causation | `#E65100` | relational |
| temporal_order | `#827717` | relational |
| sentiment | `#2E7D32` | affective |
| moral_valence | `#00695C` | affective |
| sarcasm | `#986714` | epistemic |
| plurality | `#809814` | syntactic |
| exfiltration | `#599814` | security |
| agency | `#339814` | relational |
| threat_severity | `#149852` | affective |
| formality | `#148A98` | epistemic |
| obfuscation | `#146C98` | security |
| specificity | `#144F98` | epistemic |
| authorization | `#143098` | security |
| deception | `#351498` | epistemic |
| urgency | `#981487` | affective |

**Always use `concept_color(name)` — never index `CONCEPT_COLORS` directly.**
This function handles the 18 named concepts and falls back to a deterministic
hash-derived color for anything else (SAE features, novel concepts, etc.):

```python
color = concept_color("credibility")          # → "#7B1FA2"
color = concept_color("residual_stream_42")   # → stable hash color
color = concept_color("negation", accessible=True)  # → Tol colorblind palette
```

The hash fallback uses the same L=0.34, S=0.76 aesthetic — unknown concepts
blend visually with known ones, they just aren't pinned.

### Maximum Concepts Per Plot

**Hard limit: 8 concepts in any single panel.**

Beyond 8, hue separation degrades to the point where lines or bands become
ambiguous even with a legend. If you have more than 8 concepts to show:
- Split into multiple panels (one per semantic type, or one per model family)
- Use a sequential/diverging colormap instead of categorical colors
- Show only the most relevant subset and note the omission in the caption

### Accessible (Colorblind-Safe) Palette

Primary palette is **not** colorblind-safe by design — depth and saturation
take priority for print. When accessibility is required:

```python
color = concept_color(name, accessible=True)
```

This uses Paul Tol's Bright + Muted palettes. Best for ≤ 8 concepts.
Full 18-concept colorblind safety is not achievable; the accessible flag is
most reliable for single-type subsets (e.g. the 3 security concepts together).

### Architecture Family Colors

Models are grouped and color-coded by family for cross-model comparisons:

| Family | Hex |
|---|---|
| Pythia | `#1565C0` (blue) |
| OPT | `#6A1B9A` (purple) |
| GPT-2 | `#558B2F` (green) |
| Qwen | `#E65100` (orange) |
| Llama | `#AD1457` (pink) |
| Gemma | `#00695C` (teal) |
| Phi | `#4E342E` (brown) |

Use `FAMILY_COLORS[family]` and `FAMILY_MAP[model_id]` → `(family, param_count)`.
Use `sort_models(model_ids)` to order models by family then by scale.

### CKA Line Color

Adjacent-layer CKA profile always uses `#1565C0` (dark blue), drawn dashed at
`linewidth=1.1`, `alpha=0.65`. It shares color with the Pythia family label —
that's intentional and acceptable since they appear in different contexts.

```python
ax_cka.plot(cka_x, cka_arr, color=THEME["cka_line"],
            linewidth=1.1, alpha=0.65, linestyle="--")
```

### CAZ Bands

Two overlapping spans for each detected CAZ region:

```python
# Wide Fisher-derived extent — translucent
ax.axvspan(start, end, alpha=0.12, color=concept_color, linewidth=0)

# Narrow CKA-refined extent — more opaque
ax.axvspan(cka_start, cka_end, alpha=0.45, color=concept_color, linewidth=0)
```

The contrast between 0.12 and 0.45 alpha reads clearly in both color and
greyscale print.

### CAZ Score Categories

Four named strength levels, each with a saturated marker/edge color and a very
light fill for band shading. Definitions live in `viz_style.py`; import them
rather than defining locally.

| Category | Score threshold | Edge color | Fill color |
|---|---|---|---|
| `black_hole` | > 0.5 | `#C62828` dark red | `#FFCDD2` light red |
| `strong` | > 0.2 | `#E65100` deep orange | `#FFE0B2` light orange |
| `moderate` | > 0.05 | `#F9A825` amber | `#FFF9C4` light amber |
| `gentle` | ≤ 0.05 | `#1565C0` dark blue | `#BBDEFB` light blue |

```python
from viz_style import CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat

cat   = caz_score_cat(region.caz_score)          # "black_hole" | "strong" | "moderate" | "gentle"
color = CAZ_CAT_COLORS[cat]                       # saturated edge / marker color
fill  = CAZ_CAT_FILL[cat]                         # light fill for axvspan background
label = CAZ_CAT_LABELS[cat]                       # "Black hole" | "Strong" | "Moderate" | "Gentle"

# Canonical peak marker: saturated color, white edge, score printed inside
ax.plot(peak_x, peak_y, "o",
        color=color, markersize=10, zorder=5,
        markeredgecolor="white", markeredgewidth=1.5)
ax.text(peak_x, peak_y, f"{score:.2f}",
        ha="center", va="center", fontsize=5.5, color="white",
        fontweight="bold", zorder=6)

# Canonical band shading: light fill under the curve
ax.axvspan(region.start, region.end,
           facecolor=fill, edgecolor=color, linewidth=0.6,
           alpha=0.7, zorder=1)
```

The color ramp — red → orange → amber → blue — encodes intensity monotonically
and reads correctly in greyscale (red/orange are darker than amber/blue at
equal saturation).

### Coherence C(l) Secondary Axis

When showing coherence alongside separation on the same panel, use a right twin
axis. The convention is a dashed blue-grey line, subdued relative to the primary
S(l) curve so it reads as supplementary information.

```python
from viz_style import THEME

_coh = THEME["coh_line"]   # "#546E7A" — blue-grey, distinct from CKA blue
ax_c = ax.twinx()
ax_c.plot(layers, coherence, color=_coh, linewidth=0.8, linestyle="--",
          alpha=0.45, zorder=2, label="Coherence C(ℓ)")
ax_c.set_ylabel("Coherence  $C(\\ell)$", fontsize=8, color=_coh, labelpad=4)
ax_c.tick_params(axis="y", labelsize=7, colors=_coh)
ax_c.set_ylim(0, 0.7)          # coherence rarely exceeds 0.7; fixed range aids comparison
ax_c.yaxis.set_tick_params(width=0.5)
for sp in ax_c.spines.values():
    sp.set_linewidth(0.5)
```

Two distinct twin-axis colors are in play — don't mix them up:

| Axis | Color | Key | Use |
|---|---|---|---|
| CKA similarity | `#1565C0` | `THEME["cka_line"]` | Adjacent-layer CKA profile |
| Coherence C(l) | `#546E7A` | `THEME["coh_line"]` | Coherence secondary axis |

Both axes follow the same label/tick color convention: the right-axis label and
ticks are colored to match their curve, providing a visual link with no legend
needed.

### Velocity Peak Dot

```python
ax.plot(layer, sep_value, "o", color=concept_color, markersize=6.5,
        zorder=6, markeredgecolor="white", markeredgewidth=1.2)
```

White edge prevents the dot from disappearing into dark band regions.

---

## X-Axis: Layer Count + Percentage

Always show both the absolute layer number and the relative depth:

```python
positions, labels = layer_ticks(n_layers)   # default: 0%, 25%, 50%, 75%, 100%
ax.set_xticks(positions)
ax.set_xticklabels(labels, color=THEME["dim"], fontsize=7.5)
```

Output: `L0\n(0%)`, `L6\n(25%)`, etc. The layer count is the primary label;
the percentage sits below it in smaller text (handled by the `\n`).

For compact panels (cross-model grids) use just 0% / 50% / 100%:

```python
positions, labels = layer_ticks(n_layers, pcts=(0, 50, 100))
```

---

## Axes Setup

Call `apply_theme(ax, ax_twin)` once per panel after creation:

```python
ax_cka = ax.twinx()
apply_theme(ax, ax_cka)
```

This sets white background, light grey spines and grid, and dim tick colors
on the primary axis and CKA blue on the twin.

---

## Outside-Axes Callouts

Never place annotation text inside the data area — it always occludes data.
Use `add_outside_callouts()` to place labels above or below the axes with
straight vertical pointer lines.

```python
add_outside_callouts(ax, [
    {
        "x":     float(fisher_peak_layer),
        "y":     separation_at_peak,
        "label": "Velocity peak\n(max assembly rate)",
        "color": THEME["annot"],
    },
    {
        "x":     float(fisher_band_center),
        "y":     separation_at_center * 0.5,
        "label": "Fisher CAZ extent\n(velocity threshold)",
        "color": concept_color,
    },
    {
        "x":     float(cka_band_center),
        "y":     separation_at_center,
        "label": "CKA-refined extent\n(active-transformation\nwindow only)",
        "color": concept_color,
    },
], n_layers=n_layers)
```

The function auto-assigns each label to one of four slots (2 rows above, 2
rows below) to avoid overlap. Slots are tried in order: top-near → bot-near →
top-far → bot-far.

**Only annotate the first/key panel** in a multi-panel figure. Other panels
should be clean.

---

## Panel Titles

```python
# Bold concept name, left-aligned, in concept color
ax.set_title(concept.replace("_", " ").title(),
             color=concept_color, fontsize=10, fontweight="bold",
             loc="left", pad=4)

# Type + depth, dimmer, just to the right
ax.text(0.19, 1.01, f"{ctype}  ·  {n_layers} layers",
        transform=ax.transAxes, color=THEME["dim"], fontsize=8, va="bottom")
```

---

## Legends

Put the legend at the top center of the figure, not inside any panel:

```python
fig.legend(
    handles=[...],
    loc="upper center",
    bbox_to_anchor=(0.50, 0.995),
    ncol=3,            # adjust to number of items
    fontsize=8,
    facecolor="white",
    edgecolor=THEME["spine"],
    labelcolor=THEME["text"],
    handlelength=2.0,
    framealpha=1.0,
)
```

---

## Figure Titles

```python
fig.suptitle(
    f"Title — {model_short}",
    color=THEME["text"], fontsize=13, fontweight="bold",
    y=1.00, va="bottom",
)
```

---

## Output

```python
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
plt.close("all")
```

`dpi=150` balances file size with enough resolution for both screen and print.
Use `dpi=200` for figures intended as paper panels.

---

## Grid Lines

Horizontal grid only, light grey, no axes grid by default:

```python
ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
ax.grid(axis="x", visible=False)
```

`apply_theme()` sets this automatically. Override only when x-grid adds genuine
value (e.g. heatmaps, timeline figures).

---

## Interactive / Dark-Background Figures

The white-background rules apply to **static paper figures only**. Interactive
HTML outputs (Plotly 3D, animated viz, exploration dashboards) may use a dark
background. Standard dark bg: `rgb(5,5,15)` for figure, `rgb(8,8,18)` for
panels. Use `concept_color()` for all concept colors regardless of background —
the same hex values are readable on dark.

Dark-bg outputs live in `visualizations/` or `results/`, never in `papers/`.

---

## Saving Figures

Always pass `facecolor="white"` explicitly — matplotlib's default can produce
transparent backgrounds that render as black in some PDF viewers:

```python
fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
```

Use `dpi=200` for figures going directly into a paper panel. Use `dpi=150` for
all other outputs.

---

## What NOT to Do

- **No dark backgrounds in paper figures** — use dark bg only for interactive HTML in `visualizations/`
- **No in-plot annotation text** — always use `add_outside_callouts()`
- **No percentage-only x-axis** — always include layer count alongside
- **No per-script color definitions** — always call `concept_color()` or import from `viz_style.py`
- **No `plt.show()`** — scripts run headless on GPU machines; always save to file
- **No missing `facecolor="white"` in `savefig`** — pass it explicitly every time
- **No more than 8 concepts in one panel** — split or subset instead
- **No indexing `CONCEPT_COLORS` directly for unknown concepts** — use `concept_color()` for hash fallback
