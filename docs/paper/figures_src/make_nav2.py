#!/usr/bin/env python3
"""Fig 4 — the Nav2-driven robot reaching a patrol waypoint in the twin.

Real frames captured during a floor-4 door-inspection patrol run (the robot is
driven autonomously between waypoints by ROS 2 Nav2; cache run 20260327_185946):
  (a) third-person follow camera — Nova Carter stopped at a glass-door waypoint;
  (b) the robot's forward-facing first-person camera (the VLM input) viewing the
      closed glass door it has reached.
No synthetic/illustrated render is used — these are rendered sensor frames from
the deployed stack. Frames are pre-extracted to figures/nav2_{follow,onboard}.png
by regenerate() so this script does not depend on the 40 MB pose cache.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / "latex" / "figures"
ROOT = Path(__file__).resolve().parents[3]

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def regenerate():
    """(Re)extract the two real frames from the tracked demo cache if missing."""
    import numpy as np
    from PIL import Image
    cache = ROOT / "door_demo_cache_238942.npz"
    if not cache.exists():
        return
    d = np.load(cache, allow_pickle=True)
    Image.fromarray(d["follow_rgba"][18][..., :3]).save(FIG / "nav2_follow.png")
    Image.fromarray(d["all_rgba"][40][..., :3]).save(FIG / "nav2_onboard.png")


PANELS = [
    ("nav2_follow.png",  "(a) Follow camera: Nova Carter at a patrol waypoint"),
    ("nav2_onboard.png", "(b) Onboard first-person camera (VLM input)"),
]

if not (FIG / "nav2_follow.png").exists():
    regenerate()

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
for ax, (fname, label) in zip(axes, PANELS):
    ax.imshow(mpimg.imread(FIG / fname))
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.045, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.2)
    for s in ax.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.6)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.085, wspace=0.025)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_nav2.{ext}", bbox_inches="tight", dpi=300)
print("saved", FIG / "fig_nav2.pdf")
