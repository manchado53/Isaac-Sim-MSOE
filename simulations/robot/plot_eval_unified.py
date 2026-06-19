#!/usr/bin/env python3
"""Simple per-class accuracy bar charts for the unified door/glass evals.

Reads data/eval_unified_door.json and data/eval_unified_glass.json and writes
one PNG per split under data/.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

LABELS = ["DOOR_OPEN", "DOOR_CLOSED", "GLASS_CRACKED", "GLASS_INTACT"]

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def plot(json_path: str, out_path: str, title: str, color: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    bars = [(L, data["per_class"][L] * 100, sum(data["confusion"][L].values()))
            for L in LABELS
            if sum(data["confusion"][L].values()) > 0]

    names  = [n.replace("_", " ").title() for n, _, _ in bars]
    vals   = [v for _, v, _ in bars]
    totals = [t for _, _, t in bars]

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(len(names))
    bars_ = ax.bar(x, vals, width=0.35, color=color,
                   edgecolor="none", zorder=3)

    # Value label above each bar.
    for b, v, t in zip(bars_, vals, totals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                f"{v:.1f}%",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#222")
        correct = round(v / 100 * t)
        ax.text(b.get_x() + b.get_width() / 2, v / 2,
                f"{correct}/{t}",
                ha="center", va="center",
                fontsize=10, color="white", fontweight="semibold")

    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#333")
    ax.tick_params(axis="y", labelsize=10, colors="#555")
    ax.tick_params(axis="x", colors="#333")

    ax.yaxis.grid(True, color="#DDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BBB")
    ax.spines["bottom"].set_color("#BBB")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98, color="#222")
    ax.set_title(f"Overall accuracy: {data['overall']*100:.1f}%   "
                 f"(n = {data['n_samples']:,})",
                 fontsize=11, color="#555", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_horizontal(json_path: str, out_path: str, title: str, color: str) -> None:
    """Horizontal per-class accuracy bars — cleaner for small class counts."""
    with open(json_path) as f:
        data = json.load(f)

    rows = [(L, data["per_class"][L] * 100, sum(data["confusion"][L].values()))
            for L in LABELS
            if sum(data["confusion"][L].values()) > 0]

    names  = [n.replace("_", " ").title() for n, _, _ in rows]
    vals   = [v for _, v, _ in rows]
    totals = [t for _, _, t in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    y = np.arange(len(names))[::-1]  # first class on top
    ax.barh(y, [100] * len(vals), height=0.55,
            color="#ECECEC", edgecolor="none", zorder=2)  # full-width track
    bars_ = ax.barh(y, vals, height=0.55, color=color,
                    edgecolor="none", zorder=3)

    for bar, v, t, yi in zip(bars_, vals, totals, y):
        correct = round(v / 100 * t)
        ax.text(v + 1.2, yi, f"{v:.1f}%",
                va="center", ha="left",
                fontsize=13, fontweight="bold", color="#222")
        ax.text(v / 2, yi, f"{correct}/{t}",
                va="center", ha="center",
                fontsize=10.5, color="white", fontweight="semibold")

    ax.set_xlim(0, 115)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=13, color="#222")
    ax.set_xlabel("Accuracy (%)", fontsize=11, color="#333")
    ax.tick_params(axis="x", labelsize=10, colors="#555")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#DDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBB")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98, color="#222")
    ax.set_title(f"Overall accuracy: {data['overall']*100:.1f}%   "
                 f"(n = {data['n_samples']:,})",
                 fontsize=11, color="#555", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_baseline_vs_finetuned(out_path: str) -> None:
    with open(os.path.join(_proj, "data", "eval_baseline_unified.json")) as f:
        base = json.load(f)
    with open(os.path.join(_proj, "data", "eval_unified_door.json")) as f:
        ft_door = json.load(f)
    with open(os.path.join(_proj, "data", "eval_unified_glass.json")) as f:
        ft_glass = json.load(f)

    # Each split's JSON still lists all 4 classes; rows with zero samples show
    # per_class=0 there. Pull each label from the split that actually contains
    # it, and compute the true overall accuracy from the sample counts, not a
    # naive mean of per-class rates (the door split has 4472 samples vs 810
    # glass — unequal weight).
    ft_per_class = {
        "DOOR_OPEN":     ft_door["per_class"]["DOOR_OPEN"],
        "DOOR_CLOSED":   ft_door["per_class"]["DOOR_CLOSED"],
        "GLASS_CRACKED": ft_glass["per_class"]["GLASS_CRACKED"],
        "GLASS_INTACT":  ft_glass["per_class"]["GLASS_INTACT"],
    }
    ft_overall = (
        (ft_door["overall"]  * ft_door["n_samples"]  +
         ft_glass["overall"] * ft_glass["n_samples"])
        / (ft_door["n_samples"] + ft_glass["n_samples"])
    )

    base_vals = [base["per_class"][L] * 100 for L in LABELS]
    ft_vals   = [ft_per_class[L]      * 100 for L in LABELS]

    x = np.arange(len(LABELS))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w/2, base_vals, w, label="Zero-shot",
                color="#C44E52", edgecolor="black")
    b2 = ax.bar(x + w/2, ft_vals,   w, label="Fine-tuned",
                color="#4C72B0", edgecolor="black")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([L.replace("_", "\n") for L in LABELS])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot vs Fine-tuned\n"
                 f"Zero-shot overall: {base['overall']*100:.2f}%   "
                 f"Fine-tuned overall: {ft_overall*100:.2f}%")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_dh_doors_zero_vs_finetuned(out_path: str) -> None:
    """Paired-bar chart comparing zero-shot vs fine-tuned on real Diericks doors."""
    with open(os.path.join(_proj, "data", "eval_baseline_dh_doors.json")) as f:
        base = json.load(f)
    with open(os.path.join(_proj, "data", "eval_unified_dh_doors.json")) as f:
        ft = json.load(f)

    classes = ["DOOR_OPEN", "DOOR_CLOSED"]
    pretty  = ["Door Open", "Door Closed"]
    totals  = [sum(base["confusion"][c].values()) for c in classes]
    base_v  = [base["per_class"][c] * 100 for c in classes]
    ft_v    = [ft["per_class"][c]   * 100 for c in classes]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(len(classes))
    w = 0.34
    b1 = ax.bar(x - w/2 - 0.015, base_v, w, color="#7A7A7A",
                label="Zero-shot", edgecolor="none", zorder=3)
    b2 = ax.bar(x + w/2 + 0.015, ft_v,   w, color="#C44E52",
                label="Fine-tuned", edgecolor="none", zorder=3)

    for bars, vals in ((b1, base_v), (b2, ft_v)):
        for b, v, t in zip(bars, vals, totals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#222")
            correct = round(v / 100 * t)
            ax.text(b.get_x() + b.get_width() / 2, v / 2,
                    f"{correct}/{t}", ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="semibold")

    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=12)
    ax.set_xlim(-0.7, len(classes) - 0.3)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#333")
    ax.tick_params(axis="y", labelsize=10, colors="#555")
    ax.tick_params(axis="x", colors="#333")
    ax.yaxis.grid(True, color="#DDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BBB")
    ax.spines["bottom"].set_color("#BBB")

    fig.suptitle("Sim-to-Real Transfer on Diericks Hall Doors",
                 fontsize=14, fontweight="bold", y=0.98, color="#222")
    ax.set_title(f"Zero-shot overall: {base['overall']*100:.1f}%   "
                 f"Fine-tuned overall: {ft['overall']*100:.1f}%   "
                 f"(n = {base['n_samples']})",
                 fontsize=11, color="#555", pad=10)

    leg = ax.legend(loc="lower left", frameon=True, fontsize=11)
    leg.get_frame().set_edgecolor("#DDD")
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    plot(os.path.join(_proj, "data", "eval_unified_door.json"),
         os.path.join(_proj, "data", "eval_unified_door_accuracy.png"),
         "Qwen3-VL Unified Adapter — Door Val",
         color="#4C72B0")
    plot(os.path.join(_proj, "data", "eval_unified_glass.json"),
         os.path.join(_proj, "data", "eval_unified_glass_accuracy.png"),
         "Qwen3-VL Unified Adapter — Glass Val",
         color="#55A868")
    plot_horizontal(
        os.path.join(_proj, "data", "eval_unified_dh_doors.json"),
        os.path.join(_proj, "data", "eval_unified_dh_doors_accuracy.png"),
        "Fine-tuned VLM on Real Diericks Hall Doors",
        color="#C44E52",
    )
    plot_baseline_vs_finetuned(
        os.path.join(_proj, "data", "eval_unified_baseline_vs_finetuned.png")
    )


if __name__ == "__main__":
    main()
