#!/usr/bin/env python3
"""
Generate presentation-ready charts from physics validation results.

Reads JSON results from the validation pipeline and produces PNG charts
suitable for slides. Handles both old-format (pre-native-metrics) and
new-format (with PhysX native metrics) JSON files.

Usage:
    python generate_validation_charts.py
    python generate_validation_charts.py --results logs/validation
    python generate_validation_charts.py --before logs/validation/raw_usd_results \
                                         --after logs/validation/physics_usd_results

Output:
    images/charts/physics_before_after.png
    images/charts/cube_door_pass_rates.png
    images/charts/gpu_compute_cost.png
    images/charts/door_failure_diagnostics.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_DIR = Path(__file__).parent.resolve()

# --- Presentation-quality style ---
COLORS = {
    "before": "#B0BEC5",   # grey
    "after": "#1E88E5",    # blue
    "pass": "#43A047",     # green
    "fail": "#E53935",     # red
    "stall": "#FB8C00",    # orange
    "accent": "#7E57C2",   # purple
    "gpu_solver": "#1E88E5",
    "gpu_broadphase": "#43A047",
    "gpu_narrowphase": "#FB8C00",
    "gpu_articulation": "#7E57C2",
    "gpu_other": "#B0BEC5",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_results_dir(directory):
    """Load all 4 result JSONs from a directory."""
    results = {}
    for name in ["building_stats", "cube_drop", "robot_movement", "door_physics"]:
        results[name] = load_json(os.path.join(directory, f"{name}_results.json"))
    return results


# ---------------------------------------------------------------------------
# Chart 1: Before/After Physics Comparison
# ---------------------------------------------------------------------------

def chart_before_after(before_stats, after_stats, output_path):
    """Arrow-style chart: '0 -> N' per metric, split by scale.

    Each row shows a bold grey '0' on the left, an arrow, then the blue bar
    with the after-value. The transformation from nothing to something is
    the core visual.
    """
    # Two groups by scale
    geo_metrics = [
        ("Collision Shapes", "collider_count"),
        ("Wall Elements", "wall_element_count"),
        ("Floor Elements", "floor_element_count"),
    ]
    dyn_metrics = [
        ("Door Elements", "door_element_count"),
        ("Rigid Bodies", "rigid_body_count"),
        ("Joints", "revolute_joint_count"),
    ]

    geo_after = [after_stats.get(k, 0) if after_stats else 0 for _, k in geo_metrics]
    geo_before = [before_stats.get(k, 0) if before_stats else 0 for _, k in geo_metrics]
    dyn_after = [after_stats.get(k, 0) if after_stats else 0 for _, k in dyn_metrics]
    dyn_before = [before_stats.get(k, 0) if before_stats else 0 for _, k in dyn_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BIM-to-Sim Physics Pipeline", fontsize=18, fontweight="bold", y=1.02)

    for ax, metrics, before_vals, after_vals, title in [
        (axes[0], geo_metrics, geo_before, geo_after, "Collision Geometry"),
        (axes[1], dyn_metrics, dyn_before, dyn_after, "Dynamics & Joints"),
    ]:
        labels = [l for l, _ in metrics]
        n = len(labels)
        y = np.arange(n)
        max_val = max(after_vals) if after_vals else 1

        # Draw the blue "after" bars
        bars = ax.barh(y, after_vals, height=0.55,
                       color=COLORS["after"], edgecolor="white", linewidth=0.5,
                       zorder=3)

        # Column headers: "Before" over the 0s, "After" over the bars
        header_y = -0.8
        ax.text(-max_val * 0.14, header_y, "BEFORE",
                va="center", ha="center", fontsize=11, fontweight="bold",
                color="#999")
        ax.text(max_val * 0.5, header_y, "AFTER",
                va="center", ha="center", fontsize=11, fontweight="bold",
                color="#1565C0")

        # Per-row: "0" box → arrow → blue bar → value
        for i, (bval, aval) in enumerate(zip(before_vals, after_vals)):
            # Bold "0" in grey box
            ax.text(-max_val * 0.14, i, str(bval),
                    va="center", ha="center", fontsize=18,
                    fontweight="bold", color="#888",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE",
                              edgecolor="#BDBDBD", linewidth=1.2),
                    zorder=5)
            # Big arrow
            ax.annotate("",
                        xy=(max_val * 0.005, i),
                        xytext=(-max_val * 0.05, i),
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.2",
                                        color="#666", lw=2.5,
                                        connectionstyle="arc3,rad=0"),
                        zorder=4)
            # Value label at end of bar
            ax.text(aval + max_val * 0.02, i,
                    f"{aval:,}", va="center", fontsize=14,
                    fontweight="bold", color="#1565C0")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlim(-max_val * 0.25, max_val * 1.22)
        ax.set_xticks([])

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Chart 2: Cube Drop + Door Pass Rates
# ---------------------------------------------------------------------------

def chart_pass_rates(cube_data, door_data, output_path):
    """Scorecard-style dashboard: pass/fail per test with key numbers."""
    cube_cameras = cube_data.get("cameras", {}) if cube_data else {}
    door_gs = door_data.get("global_summary", {}) if door_data else {}

    # Collect scorecard rows
    rows = []

    # Cube drop rows — one per floor
    for k in sorted(cube_cameras.keys()):
        c = cube_cameras[k]
        landed = c.get("cubes_landed", 0)
        total = c.get("cubes_total", 0)
        passed = c.get("passed", False)
        settle = c.get("frames_to_settle")
        room = c.get("room", f"Camera {k}")
        rows.append({
            "test": "Floor Solidity",
            "label": room,
            "result": f"{landed}/{total} cubes landed",
            "detail": f"Settled: frame {settle}" if settle else "",
            "passed": passed,
        })

    # Door physics — single summary row
    total_doors = door_gs.get("total_doors", 0)
    passed_doors = door_gs.get("total_passed", 0)
    failed_doors = door_gs.get("total_failed", 0)
    stalled = door_gs.get("total_stalled", 0)
    if total_doors > 0:
        detail = f"{stalled} stalled" if stalled > 0 else ""
        rows.append({
            "test": "Door Physics",
            "label": f"{total_doors} Hinged Doors",
            "result": f"{passed_doors}/{total_doors} swing correctly",
            "detail": detail,
            "passed": failed_doors == 0,
        })

    # Robot movement
    robot_data_src = cube_data  # we don't have robot data passed here; handled below
    # We'll add robot as a parameter

    if not rows:
        print("  [pass_rates] No data to chart, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, max(3.5, len(rows) * 1.0 + 1.5)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    fig.suptitle("Physics Validation Scorecard", fontsize=18, fontweight="bold", y=0.98)

    # Column positions
    col_check = 0.3
    col_label = 1.0
    col_result = 5.0
    col_detail = 8.0

    # Header row
    header_y = -0.6
    ax.text(col_check, header_y, "", fontsize=11, fontweight="bold", color="#666", va="center")
    ax.text(col_label, header_y, "Test", fontsize=11, fontweight="bold", color="#666", va="center")
    ax.text(col_result, header_y, "Result", fontsize=11, fontweight="bold", color="#666", va="center")
    ax.text(col_detail, header_y, "Detail", fontsize=11, fontweight="bold", color="#666", va="center")
    ax.axhline(y=-0.3, color="#DDD", linewidth=1.5, xmin=0.02, xmax=0.98)

    for i, row in enumerate(rows):
        y = i

        # Alternating row background
        if i % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color="#F8F9FA", zorder=0)

        # Pass/fail indicator
        if row["passed"]:
            ax.text(col_check, y, "\u2714", fontsize=20, fontweight="bold",
                    color=COLORS["pass"], va="center", ha="center",
                    family="DejaVu Sans")
        else:
            ax.text(col_check, y, "\u2718", fontsize=20, fontweight="bold",
                    color=COLORS["fail"], va="center", ha="center",
                    family="DejaVu Sans")

        # Label
        ax.text(col_label, y, row["label"], fontsize=12, va="center",
                fontweight="bold", color="#333")

        # Result
        result_color = COLORS["pass"] if row["passed"] else COLORS["fail"]
        ax.text(col_result, y, row["result"], fontsize=12, va="center",
                color=result_color, fontweight="bold")

        # Detail
        if row["detail"]:
            ax.text(col_detail, y, row["detail"], fontsize=10, va="center",
                    color="#888", fontstyle="italic")

    # Bottom summary line
    total_tests = len(rows)
    total_passed = sum(1 for r in rows if r["passed"])
    summary_y = len(rows) - 0.1
    ax.axhline(y=summary_y, color="#DDD", linewidth=1.5, xmin=0.02, xmax=0.98)
    summary_color = COLORS["pass"] if total_passed == total_tests else COLORS["stall"]
    ax.text(5.0, summary_y + 0.5,
            f"{total_passed}/{total_tests} tests passed",
            fontsize=14, fontweight="bold", color=summary_color,
            va="center", ha="center")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Chart 3: GPU Compute Cost
# ---------------------------------------------------------------------------

def chart_compute_cost(stats_data, output_path):
    """GPU memory breakdown + key scene complexity numbers."""
    stats = stats_data.get("stats", {}) if stats_data else {}
    cost = stats.get("compute_cost", {})
    phys = stats.get("physics_scene_stats", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- GPU Memory Pie ---
    gpu_solver = cost.get("gpu_mem_solver_mb", 0)
    gpu_broad = cost.get("gpu_mem_broadphase_mb", 0)
    gpu_narrow = cost.get("gpu_mem_narrowphase_mb", 0)
    gpu_artic = cost.get("gpu_mem_articulation_mb", 0)
    gpu_total = cost.get("gpu_mem_total_mb", 0)

    # If we don't have native metrics yet, show a placeholder with available data
    if gpu_total > 0:
        gpu_other = max(0, gpu_total - gpu_solver - gpu_broad - gpu_narrow - gpu_artic)
        sizes = [gpu_solver, gpu_broad, gpu_narrow, gpu_artic, gpu_other]
        labels = [
            f"Solver\n{gpu_solver:.1f} MB",
            f"Broadphase\n{gpu_broad:.1f} MB",
            f"Narrowphase\n{gpu_narrow:.1f} MB",
            f"Articulation\n{gpu_artic:.1f} MB",
            f"Other\n{gpu_other:.1f} MB",
        ]
        colors = [COLORS["gpu_solver"], COLORS["gpu_broadphase"],
                  COLORS["gpu_narrowphase"], COLORS["gpu_articulation"], COLORS["gpu_other"]]

        # Remove zero segments
        nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0.01]
        if nonzero:
            sizes, labels, colors = zip(*nonzero)

        wedges, texts = ax1.pie(sizes, colors=colors, startangle=90,
                                textprops={"fontsize": 10})
        ax1.legend(labels, loc="center left", bbox_to_anchor=(-0.3, 0.5),
                   fontsize=9, framealpha=0.9)
        ax1.set_title(f"GPU Physics Memory: {gpu_total:.1f} MB Total")
    else:
        # Fallback: show text summary from available stats
        text_lines = [
            f"Total Prims: {stats.get('total_prims', '?'):,}",
            f"Colliders: {stats.get('collider_count', '?'):,}",
            f"Rigid Bodies: {stats.get('rigid_body_count', '?'):,}",
            f"Joints: {stats.get('revolute_joint_count', '?')}",
            "",
            "(Run updated validation pipeline",
            "for GPU memory breakdown)",
        ]
        ax1.text(0.5, 0.5, "\n".join(text_lines), ha="center", va="center",
                 fontsize=12, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5"))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        ax1.set_title("Physics Scene Complexity")

    # --- Scene Complexity Bar ---
    complexity_metrics = [
        ("Static Bodies", phys.get("nb_static_rigids", stats.get("collider_count", 0))),
        ("Dynamic Bodies", phys.get("nb_dynamic_rigids", stats.get("rigid_body_count", 0))),
        ("Active Constraints", phys.get("nb_active_constraints", stats.get("driven_joint_count", 0))),
        ("Contact Pairs", phys.get("nb_discrete_contact_pairs_with_contacts", 0)),
        ("Trimesh Shapes", phys.get("nb_trimesh_shapes", 0)),
        ("Convex Shapes", phys.get("nb_convex_shapes", 0)),
    ]
    # Filter out zeros
    complexity_metrics = [(l, v) for l, v in complexity_metrics if v > 0]

    if complexity_metrics:
        labels_c, values_c = zip(*complexity_metrics)
        y = np.arange(len(labels_c))
        bars = ax2.barh(y, values_c, color=COLORS["after"], edgecolor="white")
        for bar, val in zip(bars, values_c):
            ax2.text(bar.get_width() + max(values_c) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=10, fontweight="bold")
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels_c)
        ax2.invert_yaxis()
        ax2.set_xlabel("Count")
        ax2.set_title("PhysX Scene Complexity")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "Run updated pipeline\nfor PhysX stats", ha="center", va="center", fontsize=14)
        ax2.set_title("PhysX Scene Complexity")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Chart 4: Door Failure Diagnostics
# ---------------------------------------------------------------------------

def chart_door_diagnostics(door_data, output_path):
    """Scatter: drive torque vs angular delta, colored by pass/stall/fail."""
    cameras = door_data.get("cameras", {}) if door_data else {}

    # Deduplicate doors across cameras
    doors = {}
    for cam_data in cameras.values():
        for d in cam_data.get("door_details", []):
            name = d.get("name", "")
            if name and name not in doors:
                doors[name] = d

    door_list = sorted(doors.values(), key=lambda d: d.get("delta", 0), reverse=True)

    if not door_list:
        print("  [door_diagnostics] No door data, skipping")
        return

    has_native = any(d.get("peak_drive_torque", 0) != 0 for d in door_list)

    fig, axes = plt.subplots(1, 2 if has_native else 1,
                             figsize=(14 if has_native else 8, 5.5))
    if not has_native:
        axes = [axes]
    else:
        axes = list(axes)

    # --- Always show: angular delta bar chart ---
    ax1 = axes[0]
    names = [d["name"].replace("Doors_", "D") for d in door_list]
    deltas = [d.get("delta", 0) for d in door_list]
    passed = [d.get("passed", False) for d in door_list]
    stalled = [d.get("stall_detected", False) for d in door_list]

    bar_colors = []
    for p, s in zip(passed, stalled):
        if s:
            bar_colors.append(COLORS["stall"])
        elif p:
            bar_colors.append(COLORS["pass"])
        else:
            bar_colors.append(COLORS["fail"])

    x = np.arange(len(names))
    ax1.bar(x, deltas, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=5.0, color="#E53935", linestyle="--", linewidth=1, alpha=0.7, label="Pass threshold (5 deg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=65, ha="right", fontsize=7)
    ax1.set_ylabel("Angular Delta (degrees)")
    ax1.set_title("Per-Door Swing Results")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["pass"], label="Passed"),
        mpatches.Patch(color=COLORS["stall"], label="Stalled (collision)"),
        mpatches.Patch(color=COLORS["fail"], label="Failed"),
    ]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.9)

    # --- Native metrics: torque vs contact scatter ---
    if has_native:
        ax2 = axes[1]
        torques = [d.get("peak_drive_torque", 0) for d in door_list]
        contacts = [d.get("contact_impulse_total", 0) for d in door_list]

        for i, d in enumerate(door_list):
            color = COLORS["stall"] if stalled[i] else (COLORS["pass"] if passed[i] else COLORS["fail"])
            size = max(20, min(200, deltas[i] * 3))
            ax2.scatter(torques[i], contacts[i], c=color, s=size, alpha=0.7, edgecolors="white", linewidth=0.5)

        ax2.set_xlabel("Peak Drive Torque (Nm)")
        ax2.set_ylabel("Total Contact Impulse")
        ax2.set_title("Failure Diagnosis: Torque vs Contact")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Annotate stalled doors
        for i, d in enumerate(door_list):
            if stalled[i]:
                ax2.annotate(names[i], (torques[i], contacts[i]),
                             fontsize=7, ha="left", va="bottom",
                             xytext=(4, 4), textcoords="offset points")

        ax2.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate validation charts for presentation")
    parser.add_argument("--results", default="logs/validation",
                        help="Directory with current result JSONs (default: logs/validation)")
    parser.add_argument("--before", default=None,
                        help="Directory with raw-USD (before) results for comparison")
    parser.add_argument("--after", default=None,
                        help="Directory with physics-USD (after) results for comparison")
    parser.add_argument("--output", default="images/charts",
                        help="Output directory for PNGs (default: images/charts)")
    args = parser.parse_args()

    # Resolve paths
    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = PROJECT_DIR / results_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING VALIDATION CHARTS")
    print("=" * 60)

    # Load data
    if args.before and args.after:
        before_dir = Path(args.before) if Path(args.before).is_absolute() else PROJECT_DIR / args.before
        after_dir = Path(args.after) if Path(args.after).is_absolute() else PROJECT_DIR / args.after
        print(f"Before: {before_dir}")
        print(f"After:  {after_dir}")
        before_data = load_results_dir(str(before_dir))
        after_data = load_results_dir(str(after_dir))
    else:
        # Try to find before/after subdirectories
        raw_dir = results_dir / "raw_usd_results"
        phys_dir = results_dir / "physics_usd_results"
        if raw_dir.exists() and phys_dir.exists():
            print(f"Found before/after dirs in {results_dir}")
            before_data = load_results_dir(str(raw_dir))
            after_data = load_results_dir(str(phys_dir))
        else:
            # Use the flat results directory as "after" only
            print(f"Using single results dir: {results_dir}")
            before_data = {}
            after_data = load_results_dir(str(results_dir))

    print(f"Output: {output_dir}\n")

    # --- Chart 1: Before/After Physics ---
    print("Chart 1: Before/After Physics Comparison")
    b_stats = (before_data.get("building_stats") or {}).get("stats", {})
    a_stats = (after_data.get("building_stats") or {}).get("stats", {})
    chart_before_after(b_stats, a_stats, output_dir / "physics_before_after.png")

    # --- Chart 2: Pass Rates ---
    print("Chart 2: Cube Drop + Door Pass Rates")
    chart_pass_rates(
        after_data.get("cube_drop"),
        after_data.get("door_physics"),
        output_dir / "cube_door_pass_rates.png",
    )

    # --- Chart 3: GPU Compute Cost ---
    print("Chart 3: GPU Compute Cost")
    chart_compute_cost(
        after_data.get("building_stats"),
        output_dir / "gpu_compute_cost.png",
    )

    # --- Chart 4: Door Failure Diagnostics ---
    print("Chart 4: Door Failure Diagnostics")
    chart_door_diagnostics(
        after_data.get("door_physics"),
        output_dir / "door_failure_diagnostics.png",
    )

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
