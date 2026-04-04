#!/usr/bin/env python3
"""
plot_training_curves.py — Training curve visualization for finetune_238471.out
(Qwen2-VL-2B LoRA + vision projector fine-tuning, fixed run with modules_to_save)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Data from finetune_238471.out
# ---------------------------------------------------------------------------

train_epochs = [0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0,
                1.222,  1.333,  1.444,  1.556,  1.667,  1.778,  1.889,  2.0,
                2.222,  2.333,  2.444,  2.556,  2.667,  2.778,  2.889,  3.0]
train_loss   = [2.136,  0.2881, 0.06214,0.02782,0.02081,0.02714,0.01372,0.01208,
                0.005194,0.006986,0.01147,0.005147,0.009222,0.001975,0.007559,0.003136,
                0.005918,0.001677,0.003635,0.002515,0.001294,0.00103, 0.001539,0.002324]
grad_norms   = [1.926,  0.2359, 0.3823, 0.4459, 0.6074, 0.2177, 0.6909, 0.03141,
                0.01464,0.02214,0.1513, 0.04641,0.3537, 0.006417,0.05681,0.02792,
                0.08024,0.05836,0.006113,0.003935,0.2027,0.009405,0.004686,0.007797]
lrs          = [9.987e-6,9.904e-6,9.749e-6,9.522e-6,9.227e-6,8.869e-6,8.453e-6,7.986e-6,
                6.924e-6,6.346e-6,5.747e-6,5.138e-6,4.526e-6,3.922e-6,3.333e-6,2.770e-6,
                1.751e-6,1.311e-6,9.263e-7,6.026e-7,3.448e-7,1.568e-7,4.129e-8,9.376e-11]

val_epochs = [1.0,    2.0,    3.0]
val_loss   = [0.008373, 0.001641, 0.001131]

trainable_params = 36_267_008
total_params     = 2_245_252_608
trainable_pct    = 100 * trainable_params / total_params

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BG       = "#0D0D0D"
PANEL_BG = "#0D0D0D"
TEXT     = "#DDDDDD"
CYAN     = "#00BFFF"
ORANGE   = "#FF8C00"
GREEN    = "#50C878"
PURPLE   = "#BF7FFF"
GRID_COL = "#222222"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        GRID_COL,
    "text.color":        TEXT,
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  "#333333",
    "font.family":       "monospace",
})

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.38, wspace=0.28,
                       left=0.07, right=0.97,
                       top=0.88, bottom=0.10)

# ---------------------------------------------------------------------------
# Panel 1 — Training & Validation Loss (full, log scale)
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(train_epochs, train_loss, color=CYAN,   linewidth=2.0, label="Train loss",  zorder=3)
ax1.scatter(val_epochs, val_loss,  color=ORANGE,  s=90, zorder=5, label="Val loss", marker="D")
ax1.plot(val_epochs,   val_loss,   color=ORANGE,  linewidth=1.4, linestyle="--", zorder=4)

for ep, vl in zip(val_epochs, val_loss):
    ax1.annotate(f"{vl:.4f}", xy=(ep, vl), xytext=(ep + 0.05, vl * 1.8),
                 color=ORANGE, fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

for ep in [1, 2, 3]:
    ax1.axvline(ep, color="#333333", linewidth=0.8, linestyle="--")

ax1.set_yscale("log")
ax1.set_xlim(0, 3.1)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss (log scale)", fontsize=11)
ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=10)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

# ---------------------------------------------------------------------------
# Panel 2 — Loss Zoom (epochs 1–3, linear scale)
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

post_ep1_epochs = [e for e in train_epochs if e >= 1.0]
post_ep1_loss   = [l for e, l in zip(train_epochs, train_loss) if e >= 1.0]

ax2.plot(post_ep1_epochs, post_ep1_loss, color=CYAN,   linewidth=2.0, label="Train loss", zorder=3)
ax2.scatter(val_epochs,   val_loss,       color=ORANGE,  s=90, zorder=5, label="Val loss", marker="D")
ax2.plot(val_epochs,      val_loss,       color=ORANGE,  linewidth=1.4, linestyle="--", zorder=4)

for ep, vl in zip(val_epochs, val_loss):
    offset_y = 0.0003 if ep < 3 else -0.0008
    ax2.annotate(f"{vl:.5f}", xy=(ep, vl), xytext=(ep + 0.05, vl + offset_y + 0.0004),
                 color=ORANGE, fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

for ep in [1, 2, 3]:
    ax2.axvline(ep, color="#333333", linewidth=0.8, linestyle="--")

ax2.set_xlim(1.0, 3.1)
ax2.set_ylim(0, 0.015)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Loss", fontsize=11)
ax2.set_title("Loss Zoom — Epochs 1–3", fontsize=13, fontweight="bold", pad=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4)

# ---------------------------------------------------------------------------
# Panel 3 — Gradient Norm
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])

ax3.fill_between(train_epochs, grad_norms, alpha=0.35, color=GREEN)
ax3.plot(train_epochs, grad_norms, color=GREEN, linewidth=1.8, zorder=3)

for ep in [1, 2, 3]:
    ax3.axvline(ep, color="#333333", linewidth=0.8, linestyle="--")

# Annotate peak
peak_idx = grad_norms.index(max(grad_norms))
ax3.annotate(f"peak {max(grad_norms):.3f}",
             xy=(train_epochs[peak_idx], max(grad_norms)),
             xytext=(train_epochs[peak_idx] + 0.25, max(grad_norms) * 0.92),
             color=GREEN, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))

ax3.set_xlim(0, 3.1)
ax3.set_ylim(bottom=0)
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("Grad Norm", fontsize=11)
ax3.set_title("Gradient Norm", fontsize=13, fontweight="bold", pad=10)
ax3.grid(True, alpha=0.4)

# ---------------------------------------------------------------------------
# Panel 4 — Learning Rate Schedule
# ---------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

lrs_1e6 = [lr * 1e6 for lr in lrs]
ax4.fill_between(train_epochs, lrs_1e6, alpha=0.30, color=PURPLE)
ax4.plot(train_epochs, lrs_1e6, color=PURPLE, linewidth=2.2, zorder=3)

for ep in [1, 2, 3]:
    ax4.axvline(ep, color="#333333", linewidth=0.8, linestyle="--")

ax4.set_xlim(0, 3.1)
ax4.set_ylim(bottom=0)
ax4.set_xlabel("Epoch", fontsize=11)
ax4.set_ylabel("Learning Rate (×10⁻⁶)", fontsize=11)
ax4.set_title("Learning Rate Schedule (Cosine)", fontsize=13, fontweight="bold", pad=10)
ax4.grid(True, alpha=0.4)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
footer = (
    f"trainable params: {trainable_params:,} ({trainable_pct:.3f}% of 2.2B)  "
    f"|  LoRA r=16 on q_proj+v_proj  +  vision projector (merger, 34M)  "
    f"|  Epoch 1 val: {val_loss[0]:.4f}  →  Epoch 2: {val_loss[1]:.5f}  →  Epoch 3: {val_loss[2]:.5f} ✓  "
    f"|  Exit code 0"
)
fig.text(0.5, 0.03, footer, ha="center", fontsize=8.5, color="#888888",
         fontfamily="monospace")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Qwen2-VL-2B LoRA Fine-Tuning — Door State Classification\n"
    "Job 238471  |  3 epochs  |  1440 train / 360 val  |  V100 32GB (dgx)  |  modules_to_save=[merger] ✓",
    fontsize=14, fontweight="bold", color=TEXT, y=0.96
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(_project_dir, "data", "training_curves_238471.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out_path}")
