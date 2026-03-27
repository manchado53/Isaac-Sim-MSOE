#!/usr/bin/env python3
"""
plot_eval_results.py — Visualize baseline vs fine-tuned evaluation results.

Reads:
  data/baseline_eval_results.json
  data/finetuned_eval_results.json

Outputs:
  data/eval_comparison.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
DATA_DIR     = os.path.join(_project_dir, "data")

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

with open(os.path.join(DATA_DIR, "baseline_eval_results.json")) as f:
    base = json.load(f)
with open(os.path.join(DATA_DIR, "finetuned_eval_results.json")) as f:
    fine = json.load(f)

# Per-sample data for fine-tuned (angle distribution of errors)
ft_samples = fine["per_sample"]
ft_errors  = [s for s in ft_samples if not s["correct"]]
ft_correct = [s for s in ft_samples if s["correct"]]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BASE  = "#E57373"   # red-ish  — baseline
C_FINE  = "#42A5F5"   # blue     — fine-tuned
C_OPEN  = "#66BB6A"   # green    — OPEN class
C_CLOSE = "#FFA726"   # orange   — CLOSED class
C_ERR   = "#EF5350"   # error bars / wrong predictions
C_OK    = "#26A69A"   # correct predictions

# ---------------------------------------------------------------------------
# Figure layout — 2 rows × 3 cols
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#1A1A2E")
PANEL_BG = "#16213E"
TEXT_COL = "#E0E0E0"

plt.rcParams.update({
    "text.color":       TEXT_COL,
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "axes.edgecolor":   "#444466",
    "grid.color":       "#2A2A4A",
})

gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

# ---------------------------------------------------------------------------
# Panel 1 — Per-class accuracy comparison (bar chart)
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL_BG)

categories = ["Overall", "CLOSED", "OPEN"]
base_vals  = [base["overall_accuracy"]*100,
              base["closed_accuracy"]*100,
              base["open_accuracy"]*100]
fine_vals  = [fine["overall_accuracy"]*100,
              fine["closed_accuracy"]*100,
              fine["open_accuracy"]*100]

x     = np.arange(len(categories))
width = 0.35
bars_b = ax1.bar(x - width/2, base_vals, width, color=C_BASE, alpha=0.85, label="Baseline")
bars_f = ax1.bar(x + width/2, fine_vals,  width, color=C_FINE, alpha=0.85, label="Fine-tuned")

for bar, val in zip(bars_b, base_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color=TEXT_COL, fontweight="bold")
for bar, val in zip(bars_f, fine_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color=TEXT_COL, fontweight="bold")

ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=10)
ax1.set_title("Per-Class Accuracy", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)
ax1.legend(fontsize=9, facecolor="#2A2A4A", edgecolor="#444466", labelcolor=TEXT_COL)
ax1.axhline(50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.text(2.6, 51.5, "random\nchance", fontsize=7, color="#888888", ha="right")
ax1.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 2 — Confusion matrix — BASELINE
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL_BG)

cm_base = np.array([[base["confusion"]["tp"], base["confusion"]["fn"]],
                    [base["confusion"]["fp"], base["confusion"]["tn"]]])
labels  = ["OPEN", "CLOSED"]
im = ax2.imshow(cm_base, cmap="Reds", vmin=0, vmax=900)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel("Predicted", fontsize=10)
ax2.set_ylabel("True", fontsize=10)
ax2.set_title("Confusion Matrix — Baseline", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)
for i in range(2):
    for j in range(2):
        val = cm_base[i, j]
        pct = val / 900 * 100
        ax2.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                 fontsize=12, color="white" if val > 450 else "#333333", fontweight="bold")

# ---------------------------------------------------------------------------
# Panel 3 — Confusion matrix — FINE-TUNED
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)

cm_fine = np.array([[fine["confusion"]["tp"], fine["confusion"]["fn"]],
                    [fine["confusion"]["fp"], fine["confusion"]["tn"]]])
im2 = ax3.imshow(cm_fine, cmap="Blues", vmin=0, vmax=900)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(labels, fontsize=10)
ax3.set_yticks([0, 1]); ax3.set_yticklabels(labels, fontsize=10)
ax3.set_xlabel("Predicted", fontsize=10)
ax3.set_ylabel("True", fontsize=10)
ax3.set_title("Confusion Matrix — Fine-Tuned", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)
for i in range(2):
    for j in range(2):
        val = cm_fine[i, j]
        pct = val / 900 * 100
        ax3.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                 fontsize=12, color="white" if val > 450 else "#333333", fontweight="bold")

# ---------------------------------------------------------------------------
# Panel 4 — Key metrics comparison (horizontal bars)
# ---------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PANEL_BG)

metrics      = ["Accuracy", "Precision", "Recall", "F1 Score"]
base_metrics = [base["overall_accuracy"], base["precision"], base["recall"], base["f1"]]
fine_metrics = [fine["overall_accuracy"], fine["precision"], fine["recall"], fine["f1"]]

y     = np.arange(len(metrics))
height = 0.35
bars_bh = ax4.barh(y + height/2, base_metrics, height, color=C_BASE, alpha=0.85, label="Baseline")
bars_fh = ax4.barh(y - height/2, fine_metrics, height, color=C_FINE, alpha=0.85, label="Fine-tuned")

for bar, val in zip(bars_bh, base_metrics):
    ax4.text(min(val + 0.01, 0.99), bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color=TEXT_COL, fontweight="bold")
for bar, val in zip(bars_fh, fine_metrics):
    ax4.text(min(val + 0.01, 0.99), bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color=TEXT_COL, fontweight="bold")

ax4.set_xlim(0, 1.12)
ax4.set_yticks(y)
ax4.set_yticklabels(metrics, fontsize=11)
ax4.set_xlabel("Score", fontsize=10)
ax4.set_title("Metric Comparison", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)
ax4.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
ax4.legend(fontsize=9, facecolor="#2A2A4A", edgecolor="#444466", labelcolor=TEXT_COL)
ax4.grid(axis="x", alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 5 — Error angle distribution (fine-tuned only)
# ---------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL_BG)

closed_err_angles = [s["angle_deg"] for s in ft_errors if s["true_label"] == "CLOSED"]
open_err_angles   = [s["angle_deg"] for s in ft_errors if s["true_label"] == "OPEN"]

bins = np.linspace(0, 16, 17)
ax5.hist(closed_err_angles, bins=bins, color=C_CLOSE, alpha=0.75, label=f"CLOSED→OPEN ({len(closed_err_angles)})")
ax5.hist(open_err_angles,   bins=bins, color=C_OPEN,  alpha=0.75, label=f"OPEN→CLOSED ({len(open_err_angles)})")
ax5.axvline(5.0, color="white", linestyle="--", linewidth=1.2, label="5° threshold")
ax5.set_xlabel("Door Angle (degrees)", fontsize=10)
ax5.set_ylabel("Error Count", fontsize=10)
ax5.set_title("Fine-Tuned: Where Errors Occur", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)
ax5.legend(fontsize=8, facecolor="#2A2A4A", edgecolor="#444466", labelcolor=TEXT_COL)
ax5.grid(alpha=0.3)
ax5.set_xlim(0, 16)

# ---------------------------------------------------------------------------
# Panel 6 — Improvement summary (text scorecard)
# ---------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL_BG)
ax6.axis("off")

rows = [
    ("Metric",           "Baseline",  "Fine-Tuned", "Δ"),
    ("Overall Acc.",     "50.0%",     "93.3%",      "+43.3pp"),
    ("CLOSED Acc.",      "0.0%",      "93.6%",      "+93.6pp"),
    ("OPEN Acc.",        "100.0%",    "93.1%",      "−6.9pp"),
    ("Precision",        "50.0%",     "93.5%",      "+43.5pp"),
    ("Recall",           "100.0%",    "93.1%",      "−6.9pp"),
    ("F1 Score",         "0.667",     "0.933",      "+0.266"),
    ("CLOSED correct",   "0/900",     "842/900",    "+842"),
    ("OPEN correct",     "900/900",   "838/900",    "−62"),
    ("Total correct",    "900/1800",  "1680/1800",  "+780"),
]

col_x = [0.02, 0.32, 0.57, 0.80]
row_h  = 0.085
y_start = 0.95

for r_idx, row in enumerate(rows):
    y_pos = y_start - r_idx * row_h
    is_header = (r_idx == 0)
    for c_idx, cell in enumerate(row):
        color = TEXT_COL
        if not is_header and c_idx == 3:
            color = C_FINE if "+" in cell else (C_BASE if "−" in cell else TEXT_COL)
        weight = "bold" if is_header else "normal"
        size   = 9.5 if is_header else 9
        ax6.text(col_x[c_idx], y_pos, cell, transform=ax6.transAxes,
                 fontsize=size, color=color, fontweight=weight, va="top")
    if is_header:
        line_y = y_start - row_h * 0.6
        ax6.plot([0, 1], [line_y, line_y], color="#444466", linewidth=0.8,
                 transform=ax6.transAxes, clip_on=False)

ax6.set_title("Results Scorecard", fontsize=12, fontweight="bold", color=TEXT_COL, pad=8)

# ---------------------------------------------------------------------------
# Overall title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Qwen2-VL-2B Door State Classification: Baseline vs LoRA Fine-Tuned\n"
    "Dataset: 1800 frames | 900 CLOSED + 900 OPEN | Sim: NVIDIA Isaac Sim",
    fontsize=13, fontweight="bold", color=TEXT_COL, y=0.97
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(DATA_DIR, "eval_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out_path}")
