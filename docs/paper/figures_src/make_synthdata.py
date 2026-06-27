#!/usr/bin/env python3
"""Figure — synthetic anomaly data generated from the digital twin.

Same curtain-wall panel, cropped tight to the glazing so the breaks fill the
frame: intact vs. the paper's three crack styles. Tempered (Voronoi) and impact
(spider-web) are the prominent fracture types; edge (a single random-walk
fracture from a panel edge) is intentionally subtle.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "paper_figures" / "anomaly_comparison"
OUT = Path(__file__).resolve().parent.parent / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9})

# crop to the cracked pane (fraction of the 1280x720 render) so cracks dominate
CY0, CY1, CX0, CX1 = 0.04, 0.86, 0.355, 0.66

PANELS = [
    ("panel00_intact.png",   "Intact"),
    ("panel00_tempered.png", "Tempered (Voronoi)"),
    ("panel00_impact.png",   "Impact (spider-web)"),
    ("panel00_edge.png",     "Edge (random walk)"),
]

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.75))
for ax, (fname, label) in zip(axes, PANELS):
    img = mpimg.imread(SRC / fname)
    h, w = img.shape[:2]
    crop = img[int(h * CY0):int(h * CY1), int(w * CX0):int(w * CX1)]
    ax.imshow(crop)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=8.8, pad=3)
    for s in ax.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.6)

fig.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.01, wspace=0.045)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_synthdata.{ext}", bbox_inches="tight", dpi=300)
print("saved", OUT / "fig_synthdata.pdf")
