#!/usr/bin/env python3
"""
plot_qwen3vl_finetune.py — visualize the Qwen3-VL-2B fine-tune result on the
bright-only val split (3,694 frames) against the same model's zero-shot baseline.

Reads:
  data/baseline_eval_qwen3_vl_2b_bright.json   (zero-shot)
  data/finetuned_eval_qwen3vl_2b_bright.json   (LoRA + projector fine-tune)

Writes to data/qwen3vl_finetune_plots/:
  1. accuracy_bars.png      — Overall / CLOSED / OPEN / F1: zero-shot vs fine-tuned
  2. confusion_matrices.png — side-by-side 2x2 heatmaps
  3. confidence_hist.png    — fine-tuned confidence: correct (green) vs wrong (red)
  4. per_door.png           — fine-tuned accuracy broken down by door ID
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
BASELINE    = os.path.join(PROJECT_DIR, "data", "baseline_eval_qwen3_vl_2b_bright.json")
FINETUNED   = os.path.join(PROJECT_DIR, "data", "finetuned_eval_qwen3vl_2b_bright.json")
OUT_DIR     = os.path.join(PROJECT_DIR, "data", "qwen3vl_finetune_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Accuracy comparison bars
# ---------------------------------------------------------------------------

def plot_accuracy_bars(base, ft, out_path):
    metrics = ["Overall", "CLOSED", "OPEN", "F1"]
    base_vals = [base["overall_accuracy"] * 100, base["closed_accuracy"] * 100,
                 base["open_accuracy"]    * 100, base["f1"]               * 100]
    ft_vals   = [ft["overall_accuracy"]   * 100, ft["closed_accuracy"]   * 100,
                 ft["open_accuracy"]      * 100, ft["f1"]                 * 100]

    x     = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, base_vals, width, label="Zero-shot", color="#94a3b8")
    b2 = ax.bar(x + width/2, ft_vals,   width, label="Fine-tuned (LoRA + projectors)",
                color="#2563eb")

    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
               label="Chance (50%)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Qwen3-VL-2B fine-tune vs zero-shot — bright-only val (n={ft['n_samples']})")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 2) Confusion matrices — side-by-side
# ---------------------------------------------------------------------------

def _confusion_grid(d):
    # rows = true (CLOSED, OPEN), cols = pred (CLOSED, OPEN)
    c = d["confusion"]
    return np.array([[c["tn"], c["fp"]],
                     [c["fn"], c["tp"]]])


def plot_confusion_matrices(base, ft, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, d, title in [(axes[0], base, f"Zero-shot ({base['overall_accuracy']*100:.1f}%)"),
                         (axes[1], ft,   f"Fine-tuned ({ft['overall_accuracy']*100:.1f}%)")]:
        m = _confusion_grid(d)
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["CLOSED", "OPEN"])
        ax.set_yticklabels(["CLOSED", "OPEN"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                color = "white" if m[i, j] > m.max() / 2 else "black"
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center",
                        color=color, fontsize=14, weight="bold")

    fig.suptitle("Qwen3-VL-2B confusion matrices — bright-only val", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 3) Confidence histogram — correct vs wrong (fine-tuned)
# ---------------------------------------------------------------------------

def plot_confidence_hist(ft, out_path):
    cc = [s["confidence"] for s in ft["per_sample"]
          if s["correct"]      and s["confidence"] is not None]
    cw = [s["confidence"] for s in ft["per_sample"]
          if not s["correct"]  and s["confidence"] is not None]

    bins = np.linspace(0.5, 1.0, 26)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cc, bins=bins, alpha=0.7, color="#10b981",
            label=f"Correct (n={len(cc)})", edgecolor="white", linewidth=0.5)
    ax.hist(cw, bins=bins, alpha=0.7, color="#ef4444",
            label=f"Wrong (n={len(cw)})",   edgecolor="white", linewidth=0.5)

    if cc:
        ax.axvline(np.mean(cc), color="#10b981", linestyle="--", linewidth=1.5,
                   label=f"$\\mu_{{✓}}$={np.mean(cc):.3f}")
    if cw:
        ax.axvline(np.mean(cw), color="#ef4444", linestyle="--", linewidth=1.5,
                   label=f"$\\mu_{{✗}}$={np.mean(cw):.3f}")

    ax.set_xlabel("Confidence at decision step")
    ax.set_ylabel("Frames")
    ax.set_yscale("log")  # log scale: correct dwarfs wrong otherwise
    ax.set_title(f"Qwen3-VL-2B fine-tuned: confidence distribution "
                 f"(overall acc {ft['overall_accuracy']*100:.1f}%)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 4) Per-door breakdown
# ---------------------------------------------------------------------------

_DOOR_RE = re.compile(r"(Doors_\d+)")


def plot_per_door(ft, out_path):
    by_door = defaultdict(lambda: {"correct": 0, "total": 0,
                                   "closed_c": 0, "closed_t": 0,
                                   "open_c":   0, "open_t":   0})
    for s in ft["per_sample"]:
        m = _DOOR_RE.search(s["image"])
        if not m:
            continue
        door = m.group(1)
        d = by_door[door]
        d["total"]   += 1
        d["correct"] += int(s["correct"])
        if s["true_label"] == "CLOSED":
            d["closed_t"] += 1
            d["closed_c"] += int(s["correct"])
        else:
            d["open_t"] += 1
            d["open_c"] += int(s["correct"])

    doors = sorted(by_door.keys(), key=lambda k: -by_door[k]["correct"] / by_door[k]["total"])
    overall = [by_door[d]["correct"] / by_door[d]["total"] * 100 for d in doors]
    closed  = [by_door[d]["closed_c"]  / by_door[d]["closed_t"] * 100
               if by_door[d]["closed_t"] else 0 for d in doors]
    openv   = [by_door[d]["open_c"]    / by_door[d]["open_t"]   * 100
               if by_door[d]["open_t"]   else 0 for d in doors]
    n_per   = [by_door[d]["total"] for d in doors]

    x     = np.arange(len(doors))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, overall, width, label="Overall", color="#3b82f6")
    ax.bar(x,         closed,  width, label="CLOSED",  color="#10b981")
    ax.bar(x + width, openv,   width, label="OPEN",    color="#f59e0b")

    for i, v in enumerate(overall):
        ax.annotate(f"{v:.1f}", xy=(x[i] - width, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(doors, n_per)],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Qwen3-VL-2B fine-tuned: per-door accuracy on bright-only val")
    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Chance (50%)")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base = json.load(open(BASELINE))
    ft   = json.load(open(FINETUNED))

    print(f"Baseline  : {base['overall_accuracy']*100:.1f}% on n={base['n_samples']}")
    print(f"Fine-tuned: {ft['overall_accuracy']*100:.1f}% on n={ft['n_samples']}")
    print()

    plot_accuracy_bars      (base, ft, os.path.join(OUT_DIR, "accuracy_bars.png"))
    plot_confusion_matrices (base, ft, os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_confidence_hist    (ft,       os.path.join(OUT_DIR, "confidence_hist.png"))
    plot_per_door           (ft,       os.path.join(OUT_DIR, "per_door.png"))

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
