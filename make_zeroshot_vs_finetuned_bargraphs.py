#!/usr/bin/env python3
"""Zero-shot vs Fine-tuned comparison — split into door and glass graphs."""

import matplotlib.pyplot as plt
import numpy as np


def make_graph(title, categories, zeroshot, finetuned, zs_overall, ft_overall, out_path):
    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    bars_ft = ax.bar(x - width/2, finetuned, width, label="Fine-Tuned",
                     color="#2E86AB", edgecolor="white", linewidth=2)
    bars_zs = ax.bar(x + width/2, zeroshot, width, label="Zero-Shot",
                     color="#E07A5F", edgecolor="white", linewidth=2)

    for bar, acc in zip(bars_zs, zeroshot):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="#222")

    for bar, acc in zip(bars_ft, finetuned):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="#222")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    subtitle = f"Overall Accuracy — Fine-Tuned: {ft_overall:.1f}%   |   Zero-Shot: {zs_overall:.1f}%"
    ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight="bold", pad=18)
    ax.title.set_position([0.5, 1.02])
    # Make subtitle smaller/lighter by setting it via suptitle-like approach
    ax.set_title(title, fontsize=16, fontweight="bold", pad=36)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, color="#555", style="italic")
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

    ax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_graph(
    title="Door State Classification — Zero-Shot vs Fine-Tuned",
    categories=["Door Open", "Door Closed"],
    zeroshot=[77.4, 90.7],
    finetuned=[85.5, 94.9],
    zs_overall=84.1,
    ft_overall=90.2,
    out_path="eval_zeroshot_vs_finetuned_doors.png",
)

make_graph(
    title="Glass State Classification — Zero-Shot vs Fine-Tuned",
    categories=["Glass Cracked", "Glass Intact"],
    zeroshot=[30.9, 44.2],
    finetuned=[98.8, 95.6],
    zs_overall=37.6,
    ft_overall=97.2,
    out_path="eval_zeroshot_vs_finetuned_glass.png",
)
