#!/usr/bin/env python3
"""
plot_qwen3vl_holdout.py — visualize the held-out-door generalization eval.

Reads:
  data/holdout_eval_qwen3vl_2b_finetuned.json

Writes to data/qwen3vl_holdout_plots/:
  1. accuracy_bars.png      — Overall / CLOSED / OPEN / F1
  2. confusion_matrix.png   — 2x2 heatmap
  3. per_door.png           — per-door breakdown of the 2 holdout doors
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOLDOUT     = os.path.join(PROJECT_DIR, "data", "holdout_eval_qwen3vl_2b_finetuned.json")
OUT_DIR     = os.path.join(PROJECT_DIR, "data", "qwen3vl_holdout_plots")
os.makedirs(OUT_DIR, exist_ok=True)

_DOOR_RE = re.compile(r"(Doors_\d+)")


def plot_accuracy_bars(holdout, out_path):
    metrics = ["Overall", "CLOSED", "OPEN", "F1"]
    values  = [holdout["overall_accuracy"]*100, holdout["closed_accuracy"]*100,
               holdout["open_accuracy"]*100,    holdout["f1"]*100]

    # Curated palette — deep slate anchors the headline metric, cool teal/indigo
    # accents for the per-class metrics, muted plum for the summary metric.
    colors = ["#1e3a8a", "#0d9488", "#7c3aed", "#475569"]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor="white")
    ax.set_facecolor("#fafafa")

    bars = ax.bar(x, values, 0.6, color=colors,
                  edgecolor="white", linewidth=1.5, zorder=3)

    ax.axhline(50, color="#dc2626", linestyle=(0, (4, 4)), linewidth=1.2,
               alpha=0.7, zorder=2, label="Chance (50%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#1f2937")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, color="#1f2937")
    ax.tick_params(axis="y", labelsize=10, colors="#475569")
    ax.set_ylabel("Score (%)", fontsize=12, color="#1f2937", labelpad=10)
    ax.set_ylim(0, 110)
    ax.set_title("Held-out door accuracy",
                 fontsize=15, fontweight="bold", color="#0f172a", pad=15)

    ax.legend(loc="lower right", framealpha=0.95,
              fontsize=10, edgecolor="#cbd5e1")

    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6,
            color="#94a3b8", zorder=1)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_confusion_matrix(holdout, out_path):
    c = holdout["confusion"]
    m = np.array([[c["tn"], c["fp"]],
                  [c["fn"], c["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CLOSED", "OPEN"])
    ax.set_yticklabels(["CLOSED", "OPEN"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Held-out door confusion matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if m[i, j] > m.max() / 2 else "black"
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center",
                    color=color, fontsize=14, weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_door(holdout, out_path):
    by_door = defaultdict(lambda: {"correct": 0, "total": 0,
                                   "closed_c": 0, "closed_t": 0,
                                   "open_c": 0,   "open_t": 0})
    for s in holdout["per_sample"]:
        m = _DOOR_RE.search(s["image"])
        if not m:
            continue
        d = by_door[m.group(1)]
        d["total"]   += 1
        d["correct"] += int(s["correct"])
        if s["true_label"] == "CLOSED":
            d["closed_t"] += 1
            d["closed_c"] += int(s["correct"])
        else:
            d["open_t"] += 1
            d["open_c"] += int(s["correct"])

    doors   = sorted(by_door.keys())
    overall = [by_door[d]["correct"]   / by_door[d]["total"]    * 100 for d in doors]
    closed  = [by_door[d]["closed_c"]  / by_door[d]["closed_t"] * 100 for d in doors]
    openv   = [by_door[d]["open_c"]    / by_door[d]["open_t"]   * 100 for d in doors]
    n_per   = [by_door[d]["total"] for d in doors]

    x     = np.arange(len(doors))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - width, overall, width, label="Overall", color="#3b82f6")
    b2 = ax.bar(x,         closed,  width, label="CLOSED",  color="#10b981")
    b3 = ax.bar(x + width, openv,   width, label="OPEN",    color="#f59e0b")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Chance (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n:,})" for d, n in zip(doors, n_per)],
                       fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-door accuracy")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    holdout = json.load(open(HOLDOUT))
    print(f"Held-out doors: {holdout['overall_accuracy']*100:.1f}% "
          f"(n={holdout['n_samples']:,})\n")

    plot_accuracy_bars   (holdout, os.path.join(OUT_DIR, "accuracy_bars.png"))
    plot_confusion_matrix(holdout, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_per_door        (holdout, os.path.join(OUT_DIR, "per_door.png"))

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
