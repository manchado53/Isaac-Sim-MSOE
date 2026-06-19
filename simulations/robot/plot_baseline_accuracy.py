#!/usr/bin/env python3
"""
Generates a slide-ready bar chart showing zero-shot baseline accuracy:
  - Overall: 50% (coin-flip — all predictions OPEN)
  - CLOSED accuracy: 0%   (missed every closed door)
  - OPEN accuracy:   100% (trivially correct — always predicted OPEN)

Output: data/baseline_accuracy.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
out_path     = os.path.join(_project_dir, "data", "baseline_accuracy.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────
categories = ["CLOSED Accuracy", "Overall Accuracy", "OPEN Accuracy"]
values     = [0.0, 50.0, 100.0]
colors     = ["#c0392b", "#e67e22", "#27ae60"]

# ── Figure — standard matplotlib style ────────────────────────────────────
plt.style.use("default")
fig, ax = plt.subplots(figsize=(7, 5))

x    = np.arange(len(categories))
bars = ax.bar(x, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

# Value labels on bars
for bar, val in zip(bars, values):
    y_pos = val + 2.0 if val > 0 else 3.5
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:.0f}%",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold")

# Axes
ax.set_ylim(0, 115)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=10)
ax.set_title("Zero-Shot Baseline — Qwen2-VL-2B (No Fine-Tuning)",
             fontsize=13, fontweight="bold", pad=14)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

# Subtitle note
fig.text(0.5, 0.01,
         "900 CLOSED frames predicted OPEN (0% accuracy)  ·  900 OPEN frames predicted OPEN (100% accuracy)",
         ha="center", fontsize=8.5, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
