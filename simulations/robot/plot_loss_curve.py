#!/usr/bin/env python3
"""
Simple training/validation loss curve for slides.
Data sourced from data/qwen2vl_door_lora/checkpoint-540/trainer_state.json
Output: data/loss_curve.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
out_path     = os.path.join(_project_dir, "data", "loss_curve.png")

# ── Data from checkpoint-474/trainer_state.json (job 239013, 1260 train frames) ──
train_epochs = [
    0.127, 0.254, 0.381, 0.508, 0.635, 0.762, 0.889,
    1.013, 1.140, 1.267, 1.394, 1.521, 1.648, 1.775, 1.902,
    2.025, 2.152, 2.279, 2.406, 2.533, 2.660, 2.787, 2.914,
]
train_loss = [
    4.502, 1.990, 0.263, 0.058, 0.032, 0.014, 0.019,
    0.020, 0.009, 0.008, 0.006, 0.004, 0.010, 0.009, 0.006,
    0.008, 0.004, 0.006, 0.002, 0.001, 0.001, 0.004, 0.001,
]

val_epochs = [1.0, 2.0, 3.0]
val_loss   = [0.00834, 0.00372, 0.00242]

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(train_epochs, train_loss, color="#2196F3", linewidth=2.5, label="Train Loss")
ax.plot(val_epochs,   val_loss,   color="#FF9800", linewidth=2.5,
        marker="D", markersize=9, label="Val Loss", zorder=5)

# Annotate val loss values — offset upward in log space
offsets = [1.8, 1.8, 1.8]
for e, v, off in zip(val_epochs, val_loss, offsets):
    ax.annotate(f"{v:.4f}", xy=(e, v), xytext=(e + 0.1, v * off),
                fontsize=11, color="#FF9800", fontweight="bold")

ax.set_yscale("log")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss (log scale)", fontsize=13)
ax.set_title("Training & Validation Loss — Qwen2-VL-2B LoRA Fine-Tuning",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0, 3.2)
ax.legend(fontsize=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
