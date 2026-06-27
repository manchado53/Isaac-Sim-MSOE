#!/usr/bin/env python3
"""Fig 1 — overview of the integrated BIM-to-Robot pipeline.

Systems-paper style: white cards with thin borders and a small colour accent,
real artifact thumbnails embedded in each stage, a dashed group box around the
three per-twin streams, and a bus-style connector (drops to a horizontal rail,
then into each box) instead of symmetric fanned arrows.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.transforms import Bbox

FIGDIR = Path(__file__).resolve().parent.parent / "latex" / "figures"
TH = FIGDIR / "thumbs"

FIGW, FIGH = 3.4, 4.5
plt.rcParams.update({"font.family": "serif"})

INK = "#1f2933"      # text
SUB = "#52606d"      # subtitle
LINE = "#7b8794"     # borders / connectors
ACC = {              # muted stage accents
    "bim": "#6b7280", "twin": "#2c5f8a", "route": "#6d5aa6",
    "nav": "#2f7d77", "perc": "#a8632b", "out": "#3f7d4e",
}

fig = plt.figure(figsize=(FIGW, FIGH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def card(x0, y0, w, h, accent, lw=0.9, fc="white", ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.010",
        linewidth=lw, edgecolor=LINE, facecolor=fc, linestyle=ls, zorder=2))
    if accent:
        ax.add_patch(Rectangle((x0, y0), 0.008, h, facecolor=accent,
                               edgecolor="none", zorder=3))


def thumb(cx, top, wf, img):
    """place an image (no distortion) with its top at `top`, centred on cx."""
    a = img.shape[1] / img.shape[0]
    hf = wf * FIGW / (a * FIGH)
    l, b = cx - wf / 2, top - hf
    axi = fig.add_axes([l, b, wf, hf], zorder=4)
    axi.imshow(img, aspect="auto"); axi.axis("off")
    ax.add_patch(Rectangle((l, b), wf, hf, fill=False, edgecolor=LINE,
                           linewidth=0.6, zorder=5))
    return b  # bottom y of the thumbnail


def thumb_h(cx, top, hf, img):
    """place an image at a FIXED height hf (width follows aspect); top-aligned."""
    a = img.shape[1] / img.shape[0]
    wf = hf * a * FIGH / FIGW
    l, b = cx - wf / 2, top - hf
    axi = fig.add_axes([l, b, wf, hf], zorder=4)
    axi.imshow(img, aspect="auto"); axi.axis("off")
    ax.add_patch(Rectangle((l, b), wf, hf, fill=False, edgecolor=LINE,
                           linewidth=0.6, zorder=5))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=8, lw=1.0, color=LINE, zorder=1,
                 shrinkA=0, shrinkB=0))


def rail(x0, x1, y):
    ax.plot([x0, x1], [y, y], color=LINE, lw=1.0, zorder=1)


def text(x, y, s, size, color=INK, weight="normal", style="normal", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=6, linespacing=1.05)


im = {k: mpimg.imread(TH / f"t_{k}.jpg") for k in ("twin", "route", "nav", "perc")}

# ---- Stage A: BIM ----------------------------------------------------------
card(0.27, 0.915, 0.46, 0.066, ACC["bim"])
text(0.515, 0.948, "Architectural BIM model", 7.6, weight="bold")
text(0.515, 0.927, "Autodesk Revit", 6.2, color=SUB)

# ---- Stage B: digital twin (text left, thumbnail right) --------------------
card(0.12, 0.715, 0.76, 0.165, ACC["twin"])
text(0.30, 0.838, "Physics-accurate", 7.4, weight="bold")
text(0.30, 0.818, "digital twin", 7.4, weight="bold")
text(0.30, 0.792, "Isaac Sim / USD", 6.2, color=SUB)
text(0.30, 0.775, "colliders, materials,", 6.0, color=SUB)
text(0.30, 0.759, "articulated doors", 6.0, color=SUB)
thumb(0.655, 0.862, 0.39, im["twin"])

# ---- group box around the three streams ------------------------------------
gx0, gy0, gw, gh = 0.035, 0.475, 0.93, 0.205
ax.add_patch(FancyBboxPatch((gx0, gy0), gw, gh,
             boxstyle="round,pad=0,rounding_size=0.012", linewidth=0.8,
             edgecolor=LINE, facecolor="#f6f7f9", linestyle=(0, (4, 3)),
             zorder=0))
# label sits in the clear gap above the dashed box (left of the connectors),
# with a white background so neither the border nor the cards cut through it
ax.text(0.035, 0.700, "per-twin streams", ha="left", va="center", fontsize=6.3,
        color=SUB, fontstyle="italic", zorder=7,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"))

# ---- three stream cards ----------------------------------------------------
streams = [
    (0.175, ACC["route"], "route", "Routing",
     ["occupancy map +", "nav graph + TSP route"]),
    (0.500, ACC["nav"], "nav", "Local navigation",
     ["ROS 2 Nav2 · LiDAR", "costmap obstacle avoid."]),
    (0.825, ACC["perc"], "perc", "Perception",
     ["Qwen3-VL-2B (LoRA)", "synthetic-label anomaly"]),
]
CW = 0.295
CARD_TOP, CARD_BOT = 0.66, 0.49
THUMB_TOP, THUMB_H = 0.642, 0.072
for cx, accent, key, title, subs in streams:
    card(cx - CW / 2, CARD_BOT, CW, CARD_TOP - CARD_BOT, accent)
    thumb_h(cx, THUMB_TOP, THUMB_H, im[key])
    text(cx, 0.553, title, 7.0, weight="bold")
    yy = 0.532
    for s in subs:
        text(cx, yy, s, 5.6, color=SUB); yy -= 0.0155

# ---- Stage D: deployment ---------------------------------------------------
card(0.18, 0.345, 0.64, 0.075, ACC["out"])
text(0.50, 0.393, "Autonomous building patrol", 7.4, weight="bold")
text(0.50, 0.369, "Nova Carter · open doors + cracked glass", 6.1, color=SUB)

# ---- connectors ------------------------------------------------------------
arrow(0.515, 0.915, 0.515, 0.882)               # A -> B
text(0.585, 0.898, "USD export", 5.6, color=SUB, style="italic", ha="left")

# B -> bus -> 3 cards
ax.plot([0.5, 0.5], [0.715, 0.692], color=LINE, lw=1.0, zorder=1)
rail(0.175, 0.825, 0.692)
for cx in (0.175, 0.5, 0.825):
    arrow(cx, 0.692, cx, 0.661)

# 3 cards -> bus -> D
for cx in (0.175, 0.5, 0.825):
    ax.plot([cx, cx], [0.49, 0.465], color=LINE, lw=1.0, zorder=1)
rail(0.175, 0.825, 0.465)
arrow(0.5, 0.465, 0.5, 0.421)

# crop to the content band (figure-fraction -> inches), trimming empty canvas
crop = Bbox.from_extents(0.02 * FIGW, 0.325 * FIGH, 0.98 * FIGW, 0.99 * FIGH)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_overview.{ext}", dpi=300, bbox_inches=crop)
print("saved", FIGDIR / "fig_overview.pdf")
