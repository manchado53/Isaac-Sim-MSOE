#!/usr/bin/env python3
"""Grouped bar graph: Qwen3-VL-2B baseline vs fine-tuned on combined 200-sample eval."""

import matplotlib.pyplot as plt
import numpy as np

categories = ["Overall", "Door Open", "Door Closed", "Glass Cracked", "Glass Intact"]
baseline   = [75.00, 100.00,  97.50, 32.00,  70.00]
finetuned  = [87.00,  88.33, 100.00, 62.00, 100.00]
base_n     = ["150 / 200", "60 / 60", "39 / 40", "16 / 50", "35 / 50"]
ft_n       = ["174 / 200", "53 / 60", "40 / 40", "31 / 50", "50 / 50"]

BASE_COLOR = "#2E86AB"
FT_COLOR   = "#4FB477"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

x = np.arange(len(categories))
w = 0.38

bars_b = ax.bar(x - w/2, baseline,  w, color=BASE_COLOR, edgecolor="white",
                linewidth=2, label="Baseline Qwen3-VL-2B")
bars_f = ax.bar(x + w/2, finetuned, w, color=FT_COLOR,   edgecolor="white",
                linewidth=2, label="Fine-tuned (unified_lora_v4)")

for bars, values, counts in [(bars_b, baseline, base_n), (bars_f, finetuned, ft_n)]:
    for bar, acc, n in zip(bars, values, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{acc:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222")
        ax.text(bar.get_x() + bar.get_width() / 2, h / 2, n,
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.set_ylim(0, 115)
ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
ax.set_title("Qwen3-VL-2B Baseline vs. Fine-Tuned — Combined Doors + Glass Evaluation (n=200)",
             fontsize=15, fontweight="bold", pad=18)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.tick_params(axis="y", labelsize=11)

ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["bottom"].set_color("#888")

ax.legend(loc="upper right", fontsize=11, frameon=False)

plt.tight_layout()
out_path = "eval_combined_baseline_vs_finetuned.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
