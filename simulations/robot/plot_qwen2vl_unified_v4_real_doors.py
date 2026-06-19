#!/usr/bin/env python3
"""
plot_qwen2vl_unified_v4_real_doors.py — bar chart comparing the unified V4
fine-tuned Qwen2-VL adapter vs. the off-the-shelf baseline on the 400-frame
real-door eval set (data/val_real_doors.jsonl).

Reads:
  data/real_doors_eval_qwen2vl_2b_unified_v4_finetuned.json
  data/real_doors_eval_qwen2vl_2b_unified_v4_zeroshot.json

Writes:
  data/plot_qwen2vl_unified_v4_real_doors.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FT_JSON = os.path.join(PROJECT_DIR, "data",
                       "real_doors_eval_qwen2vl_2b_unified_v4_finetuned.json")
ZS_JSON = os.path.join(PROJECT_DIR, "data",
                       "real_doors_eval_qwen2vl_2b_unified_v4_zeroshot.json")
OUT_PNG = os.path.join(PROJECT_DIR, "data",
                       "plot_qwen2vl_unified_v4_real_doors.png")


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "Overall": d["overall_accuracy"] * 100,
        "Closed":  d["closed_accuracy"]  * 100,
        "Open":    d["open_accuracy"]    * 100,
        "F1":      d["f1"]               * 100,
    }


def main():
    ft = load(FT_JSON)
    zs = load(ZS_JSON)

    metrics  = ["Overall", "Closed", "Open", "F1"]
    ft_vals  = [ft[m] for m in metrics]
    zs_vals  = [zs[m] for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    ft_color = "#1e3a8a"   # navy
    zs_color = "#0d9488"   # teal

    b1 = ax.bar(x - width/2, ft_vals, width, label="Fine-tuned (unified V4)",
                color=ft_color, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + width/2, zs_vals, width, label="Zero-shot Qwen2-VL-2B",
                color=zs_color, edgecolor="white", linewidth=1.2)

    ax.axhline(50, linestyle="--", linewidth=0.9, color="crimson", alpha=0.6,
               label="Chance (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Qwen2-VL real-door eval (400 frames) — fine-tuned vs. baseline",
                 fontsize=13, fontweight="bold", pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color="#1f2937")

    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Wrote {OUT_PNG}")
    print(f"  Fine-tuned: {ft}")
    print(f"  Zero-shot:  {zs}")


if __name__ == "__main__":
    main()
