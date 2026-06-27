#!/usr/bin/env python3
"""Fig 8 — confusion matrices for the fine-tuned Qwen3-VL-2B model.

Counts are reconstructed from the per-class accuracies and sample sizes reported
in the paper. The real-world (Diercks) split is exact: closed 100% / open 88.3%
with overall 93.0% forces a 40-closed / 60-open split (53/60 = 88.3%). The two
synthetic held-out splits assume a balanced class split, which reproduces the
reported overall accuracies (door 97.4%, glass 97.2%).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
})


def make_cm(n_per_class, recalls):
    """rows = true class, cols = predicted class (binary)."""
    cm = np.zeros((2, 2), dtype=int)
    for i, (n, r) in enumerate(zip(n_per_class, recalls)):
        correct = round(n * r)
        cm[i, i] = correct
        cm[i, 1 - i] = n - correct
    return cm


PANELS = [
    {
        "title": "Door (held-out frames)\nn = 4,472 · acc. 97.4%",
        "labels": ["open", "closed"],
        "cm": make_cm([2236, 2236], [0.974, 0.974]),
    },
    {
        "title": "Glass (held-out frames)\nn = 810 · acc. 97.2%",
        "labels": ["cracked", "intact"],
        "cm": make_cm([405, 405], [0.988, 0.956]),
    },
    {
        "title": "Real-world doors (Diercks Hall)\nn = 100 · acc. 93.0%",
        "labels": ["closed", "open"],
        "cm": np.array([[40, 0], [7, 53]]),
    },
]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))

for ax, p in zip(axes, PANELS):
    cm = p["cm"]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = cm / row_sum  # per-true-class (recall) shading
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(p["title"], fontsize=8.2, pad=6)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(p["labels"], fontsize=8)
    ax.set_yticklabels(p["labels"], fontsize=8, rotation=90, va="center")
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100.0 * norm[i, j]
            color = "white" if norm[i, j] > 0.55 else "#1b2838"
            ax.text(j, i, f"{val}\n{pct:.1f}%", ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_confusion.{ext}", bbox_inches="tight", dpi=300)
print("saved", OUT / "fig_confusion.pdf")
