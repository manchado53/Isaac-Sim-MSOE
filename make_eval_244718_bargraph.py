#!/usr/bin/env python3
"""Bar graph of fine-tuned Qwen3-VL-2B evaluation results (job 244718)."""

import matplotlib.pyplot as plt
import numpy as np

categories = ["Overall", "Door Open", "Door Closed"]
accuracies = [97.36, 97.37, 97.36]
counts = ["4354 / 4472", "2182 / 2241", "2172 / 2231"]

colors = ["#2E86AB", "#4FB477", "#E07A5F"]

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")

x = np.arange(len(categories))
bars = ax.bar(x, accuracies, color=colors, width=0.6, edgecolor="white", linewidth=2)

for bar, acc, n in zip(bars, accuracies, counts):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=16, fontweight="bold", color="#222")
    ax.text(bar.get_x() + bar.get_width() / 2, h / 2, n,
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")

ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
ax.set_title("Fine-Tuned Qwen3-VL-2B — Door State Classification",
             fontsize=16, fontweight="bold", pad=18)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13)
ax.tick_params(axis="y", labelsize=11)

ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["bottom"].set_color("#888")

plt.tight_layout()
out_path = "eval_244718_accuracy.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
