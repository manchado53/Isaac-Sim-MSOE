#!/usr/bin/env python3
"""
Bar graphs for FT Qwen3-VL-2B (unified_lora_v4) across the 3 eval datasets.

Each subplot shows Overall accuracy, Door Open (OPEN), Door Closed (CLOSED),
and F1 score for one dataset. F1 is plotted as F1 x 100 so all four bars share
the 0-100 scale.

Numbers sourced from:
  logs/eval_qwen3vl_unified_v4_holdout_245911.out   (V4 holdout, 7776)
  logs/eval_qwen3vl_unified_v4_dh_doors_245918.out  (dh_doors,   100)
  logs/eval_unified_v4_real_doors_245913.out        (real_doors, 400)
"""

import matplotlib.pyplot as plt
import numpy as np

datasets = [
    {
        "title":  "V4 Holdout (7,776 frames)\nunseen simulation doors",
        "overall": 90.2,
        "open":    85.5,
        "closed":  94.9,
        "f1":      0.897,
        "counts": {
            "Overall":    "7012 / 7776",
            "Door Open":  "3324 / 3888",
            "Door Closed": "3688 / 3888",
            "F1":         "",
        },
    },
    {
        "title":  "DH Doors (100 frames)\nDiercks Hall real photos",
        "overall": 93.0,
        "open":    88.3,
        "closed":  100.0,
        "f1":      0.938,
        "counts": {
            "Overall":    "93 / 100",
            "Door Open":  "53 / 60",
            "Door Closed": "40 / 40",
            "F1":         "",
        },
    },
    {
        "title":  "Real Doors (400 frames)\npublic real-door set",
        "overall": 93.2,
        "open":    94.0,
        "closed":  92.5,
        "f1":      0.935,
        "counts": {
            "Overall":    "373 / 400",
            "Door Open":  "188 / 200",
            "Door Closed": "185 / 200",
            "F1":         "",
        },
    },
]

categories = ["Overall", "Door Open", "Door Closed", "F1"]
colors     = ["#2E86AB", "#4FB477", "#E07A5F", "#8E44AD"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
fig.patch.set_facecolor("white")
fig.suptitle(
    "Fine-Tuned Qwen3-VL-2B (unified_lora_v4) — Per-Dataset Metrics",
    fontsize=18, fontweight="bold", y=1.02,
)

for ax, ds in zip(axes, datasets):
    values = [ds["overall"], ds["open"], ds["closed"], ds["f1"] * 100]
    label_values = [
        f'{ds["overall"]:.1f}%',
        f'{ds["open"]:.1f}%',
        f'{ds["closed"]:.1f}%',
        f'{ds["f1"]:.3f}',
    ]
    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, width=0.65,
                  edgecolor="white", linewidth=2)

    for bar, label, cat in zip(bars, label_values, categories):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.9, label,
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#222")
        count = ds["counts"].get(cat, "")
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2, count,
                    ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    ax.set_title(ds["title"], fontsize=13, fontweight="bold", pad=12)
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

axes[0].set_ylabel("Accuracy (%)  /  F1 × 100", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, 110)

plt.tight_layout()
out_path = "ft_qwen3vl_unified_v4_per_dataset.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
