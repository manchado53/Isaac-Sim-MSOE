#!/usr/bin/env python3
"""
Bar chart of fine-tuned model accuracy results.
Output: data/finetuned_accuracy.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
out_path     = os.path.join(_project_dir, "data", "finetuned_accuracy.png")

categories = ["CLOSED Accuracy", "Overall Accuracy", "OPEN Accuracy"]
values     = [92.6, 96.3, 100.0]
colors     = ["#c0392b", "#e67e22", "#27ae60"]

plt.style.use("default")
fig, ax = plt.subplots(figsize=(7, 5))

x    = np.arange(len(categories))
bars = ax.bar(x, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold")

ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=10)
ax.set_title("Fine-Tuned Model Accuracy — Qwen2-VL-2B + LoRA",
             fontsize=13, fontweight="bold", pad=14)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

fig.text(0.5, 0.01,
         "270 held-out test frames  ·  135 CLOSED / 135 OPEN  ·  125 CLOSED correct  ·  135 OPEN correct",
         ha="center", fontsize=8.5, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
