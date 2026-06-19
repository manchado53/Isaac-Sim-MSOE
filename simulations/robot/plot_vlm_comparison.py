#!/usr/bin/env python3
"""
plot_vlm_comparison.py — visualize the bright-only VLM zero-shot comparison.

Reads every data/baseline_eval_*_bright.json file, sorts by overall accuracy,
and writes three figures to data/vlm_comparison_plots/:

  1. accuracy_bars.png      — grouped bar chart: Overall / CLOSED / OPEN accuracy
  2. confidence_hists.png   — per-model histograms of confidence-when-correct
                              vs confidence-when-wrong (skips models with no
                              extractable confidence)
  3. summary_table.png      — rendered table of headline metrics

Run:  python simulations/robot/plot_vlm_comparison.py
"""

import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_GLOB = os.path.join(PROJECT_DIR, "data", "baseline_eval_*_bright.json")
OUT_DIR     = os.path.join(PROJECT_DIR, "data", "vlm_comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load and rank
# ---------------------------------------------------------------------------

def load_results():
    rows = []
    for fp in sorted(glob.glob(RESULTS_GLOB)):
        d = json.load(open(fp))
        samples = d["per_sample"]
        cc = [s["confidence"] for s in samples if s["correct"] and s["confidence"] is not None]
        cw = [s["confidence"] for s in samples if not s["correct"] and s["confidence"] is not None]
        rows.append({
            "model":        d["model"].split("/")[-1],
            "n":            d["n_samples"],
            "overall":      d["overall_accuracy"],
            "closed":       d["closed_accuracy"],
            "open":         d["open_accuracy"],
            "f1":           d["f1"],
            "mc_correct":   statistics.mean(cc) if cc else None,
            "mc_wrong":     statistics.mean(cw) if cw else None,
            "conf_correct": cc,
            "conf_wrong":   cw,
            "has_conf":     bool(cc) or bool(cw),
        })
    rows.sort(key=lambda r: -r["overall"])
    return rows


# ---------------------------------------------------------------------------
# Figure 1 — grouped accuracy bars
# ---------------------------------------------------------------------------

def plot_accuracy_bars(rows, out_path):
    models   = [r["model"] for r in rows]
    overall  = [r["overall"] * 100 for r in rows]
    closed   = [r["closed"]  * 100 for r in rows]
    openv    = [r["open"]    * 100 for r in rows]

    x      = np.arange(len(models))
    width  = 0.27

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, overall, width, label="Overall", color="#3b82f6")
    b2 = ax.bar(x,         closed,  width, label="CLOSED",  color="#10b981")
    b3 = ax.bar(x + width, openv,   width, label="OPEN",    color="#f59e0b")

    ax.axhline(50, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
               label="Chance (50%)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Zero-shot VLM accuracy evaluation")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — confidence histograms (one panel per model)
# ---------------------------------------------------------------------------

def plot_confidence_hists(rows, out_path):
    plottable = [r for r in rows if r["has_conf"]]
    n = len(plottable)
    cols = 3
    rows_grid = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 4.2, rows_grid * 3.0),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes).flatten()

    bins = np.linspace(0.5, 1.0, 26)  # decision-step conf is bounded below by ~0.5

    for ax, r in zip(axes, plottable):
        ax.hist(r["conf_correct"], bins=bins, alpha=0.65, color="#10b981",
                label=f"correct (n={len(r['conf_correct'])})", edgecolor="white", linewidth=0.5)
        ax.hist(r["conf_wrong"],   bins=bins, alpha=0.65, color="#ef4444",
                label=f"wrong (n={len(r['conf_wrong'])})",   edgecolor="white", linewidth=0.5)

        if r["mc_correct"] is not None:
            ax.axvline(r["mc_correct"], color="#10b981", linestyle="--", linewidth=1)
        if r["mc_wrong"] is not None:
            ax.axvline(r["mc_wrong"],   color="#ef4444", linestyle="--", linewidth=1)

        ax.set_title(f"{r['model']}\n"
                     f"acc {r['overall']*100:.1f}% | "
                     f"$\\mu_{{✓}}$={r['mc_correct']:.2f}, $\\mu_{{✗}}$={r['mc_wrong']:.2f}",
                     fontsize=10)
        ax.set_xlabel("Confidence at decision step")
        ax.set_ylabel("Frames")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    # blank out unused axes
    for ax in axes[len(plottable):]:
        ax.axis("off")

    skipped = [r["model"] for r in rows if not r["has_conf"]]
    suptitle = "Confidence distribution: correct (green) vs wrong (red)"
    if skipped:
        suptitle += f"\nNot shown: {', '.join(skipped)} (no extractable confidence)"
    fig.suptitle(suptitle, fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — summary table image
# ---------------------------------------------------------------------------

def plot_summary_table(rows, out_path):
    headers = ["Model", "Overall", "CLOSED", "OPEN", "F1"]
    cells = []
    for r in rows:
        cells.append([
            r["model"],
            f"{r['overall']*100:.1f}%",
            f"{r['closed']*100:.1f}%",
            f"{r['open']*100:.1f}%",
            f"{r['f1']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(11, 0.55 * (len(rows) + 2)))
    ax.axis("off")
    # Wide first column for model names; narrower numeric columns
    col_widths = [0.40] + [0.13] * (len(headers) - 1)
    tbl = ax.table(cellText=cells, colLabels=headers,
                   cellLoc="center", loc="center",
                   colColours=["#1e293b"] * len(headers),
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    # header styling
    for c in range(len(headers)):
        cell = tbl[0, c]
        cell.set_text_props(color="white", weight="bold")

    # highlight winner
    for c in range(len(headers)):
        tbl[1, c].set_facecolor("#dcfce7")  # light green
    # mark first column bold for all rows
    for r_i in range(1, len(rows) + 1):
        tbl[r_i, 0].set_text_props(weight="bold")

    fig.suptitle(f"Zero-shot VLM comparison — bright-only val (n={rows[0]['n']})",
                 fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = load_results()
    if not rows:
        print(f"No results found matching {RESULTS_GLOB}")
        return
    print(f"Found {len(rows)} result files; generating plots...")
    plot_accuracy_bars   (rows, os.path.join(OUT_DIR, "accuracy_bars.png"))
    plot_confidence_hists(rows, os.path.join(OUT_DIR, "confidence_hists.png"))
    plot_summary_table   (rows, os.path.join(OUT_DIR, "summary_table.png"))
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
