#!/usr/bin/env python3
"""
Extract and compare physics validation metrics from before/after runs.

Loads JSON results from two directories (raw USD vs physics USD) and
produces a side-by-side comparison in multiple formats.

Usage:
    python extract_validation_metrics.py \
        --before logs/validation/raw_usd_results \
        --after  logs/validation/physics_usd_results

Output:
    logs/validation/comparison_summary.txt  — slide-ready text
    logs/validation/comparison_metrics.json — structured JSON
    logs/validation/comparison_data.csv     — spreadsheet import
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

RESULT_FILES = {
    "building_stats": "building_stats_results.json",
    "cube_drop": "cube_drop_results.json",
    "robot_movement": "robot_movement_results.json",
    "door_physics": "door_physics_results.json",
}


def load_results(directory):
    """Load all result JSON files from a directory. Returns dict of test_name -> data or None."""
    results = {}
    for test_name, filename in RESULT_FILES.items():
        path = os.path.join(directory, filename)
        try:
            with open(path) as f:
                results[test_name] = json.load(f)
        except FileNotFoundError:
            results[test_name] = None
            print(f"  Warning: {filename} not found in {directory}")
        except json.JSONDecodeError as e:
            results[test_name] = None
            print(f"  Warning: {filename} has invalid JSON: {e}")
    return results


def safe_get(data, *keys, default=None):
    """Safely navigate nested dicts."""
    current = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def extract_building_stats(before, after):
    """Extract building stats comparison."""
    fields = [
        "total_prims", "collider_count", "rigid_body_count",
        "revolute_joint_count", "driven_joint_count", "mesh_count",
        "floor_element_count", "door_element_count", "wall_element_count",
    ]
    rows = []
    for field in fields:
        b_val = safe_get(before, "stats", field, default="N/A")
        a_val = safe_get(after, "stats", field, default="N/A")
        rows.append({"metric": field, "before": b_val, "after": a_val})

    # Building dimensions (meters; older JSONs used "_ft" label)
    b_dims = (safe_get(before, "stats", "building_dimensions_m", default=None)
              or safe_get(before, "stats", "building_dimensions_ft", default=[]))
    a_dims = (safe_get(after, "stats", "building_dimensions_m", default=None)
              or safe_get(after, "stats", "building_dimensions_ft", default=[]))
    rows.append({
        "metric": "building_dimensions_m",
        "before": f"{b_dims[0]} x {b_dims[1]} x {b_dims[2]}" if len(b_dims) == 3 else "N/A",
        "after": f"{a_dims[0]} x {a_dims[1]} x {a_dims[2]}" if len(a_dims) == 3 else "N/A",
    })

    # PhysX scene stats (native metrics)
    phys_fields = [
        "nb_static_rigids", "nb_dynamic_rigids", "nb_active_dynamic_rigids",
        "nb_kinematic_rigids", "nb_active_constraints",
        "nb_convex_shapes", "nb_trimesh_shapes",
        "nb_discrete_contact_pairs_total", "nb_discrete_contact_pairs_with_contacts",
        "nb_partitions",
    ]
    for field in phys_fields:
        b_val = safe_get(before, "stats", "physics_scene_stats", field, default="N/A")
        a_val = safe_get(after, "stats", "physics_scene_stats", field, default="N/A")
        rows.append({"metric": f"phys_{field}", "before": b_val, "after": a_val})

    # Solver residuals
    for field in ["rms_position", "max_position", "rms_velocity", "max_velocity"]:
        b_val = safe_get(before, "stats", "solver_residuals", field, default="N/A")
        a_val = safe_get(after, "stats", "solver_residuals", field, default="N/A")
        rows.append({"metric": f"residual_{field}", "before": b_val, "after": a_val})

    # Compute cost
    cost_fields = [
        "gpu_mem_total_mb", "gpu_mem_solver_mb", "gpu_mem_broadphase_mb",
        "gpu_mem_narrowphase_mb", "peak_constraint_memory_mb",
    ]
    for field in cost_fields:
        b_val = safe_get(before, "stats", "compute_cost", field, default="N/A")
        a_val = safe_get(after, "stats", "compute_cost", field, default="N/A")
        rows.append({"metric": f"cost_{field}", "before": b_val, "after": a_val})

    # Screenshot results
    for label in ["before", "after"]:
        data = before if label == "before" else after
        cameras = safe_get(data, "cameras", default={})
        for cam_key in sorted(cameras.keys()):
            cam = cameras[cam_key]
            rows.append({
                "metric": f"screenshot_{label}_cam_{cam_key}",
                "before": cam.get("non_black", "N/A") if label == "before" else "",
                "after": cam.get("non_black", "N/A") if label == "after" else "",
            })

    return rows


def extract_cube_metrics(before, after):
    """Extract cube drop comparison per camera."""
    rows = []
    # Collect all camera keys from both
    cam_keys = set()
    for data in [before, after]:
        if data:
            cam_keys.update(safe_get(data, "cameras", default={}).keys())

    for cam_key in sorted(cam_keys):
        b_cam = safe_get(before, "cameras", cam_key, default={})
        a_cam = safe_get(after, "cameras", cam_key, default={})

        b_landed = b_cam.get("cubes_landed", "N/A")
        b_total = b_cam.get("cubes_total", "N/A")
        a_landed = a_cam.get("cubes_landed", "N/A")
        a_total = a_cam.get("cubes_total", "N/A")

        room = a_cam.get("room", b_cam.get("room", f"Camera {cam_key}"))

        rows.append({
            "camera": cam_key,
            "room": room,
            "before_landed": b_landed,
            "before_total": b_total,
            "after_landed": a_landed,
            "after_total": a_total,
            "before_pass": b_cam.get("passed", "N/A"),
            "after_pass": a_cam.get("passed", "N/A"),
            "after_frames_to_settle": a_cam.get("frames_to_settle", "N/A"),
        })

        # Per-cube Z positions (after only, for distribution data)
        a_details = a_cam.get("cube_details", [])
        for cube in a_details:
            rows.append({
                "camera": cam_key,
                "room": room,
                "cube_name": cube.get("name", ""),
                "final_z": cube.get("final_position", [0, 0, 0])[2] if cube.get("final_position") else "N/A",
                "floor_z": cube.get("floor_z", "N/A"),
                "cube_passed": cube.get("passed", "N/A"),
                "max_impact_speed": cube.get("max_impact_speed", "N/A"),
                "floor_raycast_hit": cube.get("floor_raycast_hit", "N/A"),
                "floor_raycast_distance": cube.get("floor_raycast_distance", "N/A"),
            })

    # Aggregate
    b_total_landed = 0
    b_total_cubes = 0
    a_total_landed = 0
    a_total_cubes = 0
    for data, label in [(before, "b"), (after, "a")]:
        if data:
            for cam in safe_get(data, "cameras", default={}).values():
                if label == "b":
                    b_total_landed += cam.get("cubes_landed", 0)
                    b_total_cubes += cam.get("cubes_total", 0)
                else:
                    a_total_landed += cam.get("cubes_landed", 0)
                    a_total_cubes += cam.get("cubes_total", 0)

    # Settle detection from "after" data
    settle_info = safe_get(after, "settle_detection", default={})

    return rows, {
        "before_landed": b_total_landed,
        "before_total": b_total_cubes,
        "after_landed": a_total_landed,
        "after_total": a_total_cubes,
        "before_rate": f"{b_total_landed}/{b_total_cubes}" if b_total_cubes else "0/0",
        "after_rate": f"{a_total_landed}/{a_total_cubes}" if a_total_cubes else "0/0",
        "before_pct": round(100 * b_total_landed / b_total_cubes, 1) if b_total_cubes else 0,
        "after_pct": round(100 * a_total_landed / a_total_cubes, 1) if a_total_cubes else 0,
        "all_settled": settle_info.get("all_settled", "N/A"),
        "max_frames_to_settle": settle_info.get("max_frames_to_settle", "N/A"),
    }


def extract_robot_metrics(before, after):
    """Extract robot movement comparison per camera."""
    rows = []
    cam_keys = set()
    for data in [before, after]:
        if data:
            cam_keys.update(safe_get(data, "cameras", default={}).keys())

    for cam_key in sorted(cam_keys):
        b_cam = safe_get(before, "cameras", cam_key, default={})
        a_cam = safe_get(after, "cameras", cam_key, default={})

        room = a_cam.get("room", b_cam.get("room", f"Camera {cam_key}"))
        rows.append({
            "camera": cam_key,
            "room": room,
            "before_z_drop": round(b_cam.get("z_drop", float("nan")), 4) if "z_drop" in b_cam else "N/A",
            "after_z_drop": round(a_cam.get("z_drop", float("nan")), 4) if "z_drop" in a_cam else "N/A",
            "before_initial_z": round(b_cam.get("initial_z", 0), 3) if "initial_z" in b_cam else "N/A",
            "after_initial_z": round(a_cam.get("initial_z", 0), 3) if "initial_z" in a_cam else "N/A",
            "before_final_z": round(b_cam.get("final_z", 0), 3) if "final_z" in b_cam else "N/A",
            "after_final_z": round(a_cam.get("final_z", 0), 3) if "final_z" in a_cam else "N/A",
            "before_pass": b_cam.get("passed", "N/A"),
            "after_pass": a_cam.get("passed", "N/A"),
            "after_mean_speed": round(a_cam.get("mean_speed", 0), 4) if "mean_speed" in a_cam else "N/A",
            "after_max_speed": round(a_cam.get("max_speed", 0), 4) if "max_speed" in a_cam else "N/A",
            "after_min_clearance": round(a_cam.get("min_ground_clearance", 0), 4) if "min_ground_clearance" in a_cam else "N/A",
        })

    return rows


def extract_door_metrics(before, after):
    """Extract door physics comparison."""
    b_global = safe_get(before, "global_summary", default={})
    a_global = safe_get(after, "global_summary", default={})

    summary = {
        "before_total_doors": b_global.get("total_doors", 0),
        "before_total_passed": b_global.get("total_passed", 0),
        "after_total_doors": a_global.get("total_doors", 0),
        "after_total_passed": a_global.get("total_passed", 0),
    }

    # Per-door angular delta from "after" results (deduplicated)
    door_deltas = {}
    if after:
        for cam_data in safe_get(after, "cameras", default={}).values():
            for door in cam_data.get("door_details", []):
                name = door.get("name", "")
                if name and name not in door_deltas:
                    door_deltas[name] = {
                        "name": name,
                        "initial_angle": door.get("initial_angle", 0),
                        "final_angle": door.get("final_angle", 0),
                        "delta": door.get("delta", 0),
                        "passed": door.get("passed", False),
                        "distance": door.get("distance_from_camera", 0),
                        "path": door.get("path", ""),
                        "peak_angular_velocity": door.get("peak_angular_velocity", 0),
                        "final_angular_velocity": door.get("final_angular_velocity", 0),
                        "contact_impulse_total": door.get("contact_impulse_total", 0),
                        "max_penetration_depth": door.get("max_penetration_depth", 0),
                        "peak_drive_torque": door.get("peak_drive_torque", 0),
                        "stall_detected": door.get("stall_detected", False),
                    }

    # Sort by delta for distribution view
    door_list = sorted(door_deltas.values(), key=lambda d: d["delta"], reverse=True)

    # Compute distribution stats
    deltas = [d["delta"] for d in door_list]
    if deltas:
        summary["min_delta"] = round(min(deltas), 1)
        summary["max_delta"] = round(max(deltas), 1)
        summary["mean_delta"] = round(sum(deltas) / len(deltas), 1)
        summary["unique_doors_measured"] = len(deltas)
        summary["stalled_doors"] = sum(1 for d in door_list if d.get("stall_detected", False))
    else:
        summary["min_delta"] = 0
        summary["max_delta"] = 0
        summary["mean_delta"] = 0
        summary["unique_doors_measured"] = 0
        summary["stalled_doors"] = 0

    return summary, door_list


def format_summary(stats_rows, cube_agg, robot_rows, door_summary, door_list,
                   before_data, after_data):
    """Format a human-readable comparison summary for slides."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("PHYSICS VALIDATION: BEFORE vs AFTER")
    lines.append(sep)

    # --- Building Stats ---
    lines.append("")
    lines.append("BUILDING STRUCTURE")
    lines.append("-" * 50)
    lines.append(f"  {'Metric':<30s} {'Before':>10s} {'After':>10s}")
    lines.append(f"  {'-'*30:<30s} {'-'*10:>10s} {'-'*10:>10s}")
    for row in stats_rows:
        m = row["metric"]
        if m.startswith("screenshot_") or m.startswith("phys_") or m.startswith("residual_") or m.startswith("cost_"):
            continue
        label = m.replace("_", " ").title()
        b = str(row["before"])
        a = str(row["after"])
        lines.append(f"  {label:<30s} {b:>10s} {a:>10s}")

    # --- PhysX Engine Metrics ---
    phys_rows = [r for r in stats_rows if r["metric"].startswith("phys_")]
    if phys_rows and any(r["after"] != "N/A" for r in phys_rows):
        lines.append("")
        lines.append("PHYSICS ENGINE METRICS (native PhysX)")
        lines.append("-" * 50)
        lines.append(f"  {'Metric':<40s} {'Before':>10s} {'After':>10s}")
        lines.append(f"  {'-'*40:<40s} {'-'*10:>10s} {'-'*10:>10s}")
        for row in phys_rows:
            label = row["metric"].replace("phys_nb_", "").replace("phys_", "").replace("_", " ").title()
            b = str(row["before"])
            a = str(row["after"])
            lines.append(f"  {label:<40s} {b:>10s} {a:>10s}")

    # --- Solver Residuals ---
    res_rows = [r for r in stats_rows if r["metric"].startswith("residual_")]
    if res_rows and any(r["after"] != "N/A" for r in res_rows):
        lines.append("")
        lines.append("  Solver Residuals:")
        for row in res_rows:
            label = row["metric"].replace("residual_", "").replace("_", " ").title()
            a = row["after"]
            a_str = f"{a:.6f}" if isinstance(a, (int, float)) else str(a)
            lines.append(f"    {label:<30s} {a_str:>12s}")

    # --- Compute Cost ---
    cost_rows = [r for r in stats_rows if r["metric"].startswith("cost_")]
    if cost_rows and any(r["after"] != "N/A" for r in cost_rows):
        lines.append("")
        lines.append("  Compute Cost:")
        for row in cost_rows:
            label = row["metric"].replace("cost_", "").replace("_", " ").title()
            a = row["after"]
            a_str = f"{a:.2f} MB" if isinstance(a, (int, float)) else str(a)
            lines.append(f"    {label:<30s} {a_str:>12s}")

    # --- Cube Drop ---
    lines.append("")
    lines.append("CUBE DROP TEST (9 cubes per floor)")
    lines.append("-" * 50)
    lines.append(f"  {'Floor':<35s} {'Before':>10s} {'After':>10s}")
    lines.append(f"  {'-'*35:<35s} {'-'*10:>10s} {'-'*10:>10s}")

    # Per-camera summary rows only (skip cube detail rows)
    for row in [r for r in cube_agg if "cube_name" not in r]:
        if "before_landed" not in row:
            continue
        room = row.get("room", row.get("camera", "?"))
        b = f"{row['before_landed']}/{row['before_total']}" if row["before_landed"] != "N/A" else "N/A"
        a = f"{row['after_landed']}/{row['after_total']}" if row["after_landed"] != "N/A" else "N/A"
        lines.append(f"  {room:<35s} {b:>10s} {a:>10s}")

    lines.append(f"  {'':35s} {'-'*10:>10s} {'-'*10:>10s}")

    b_total = cube_agg[-1] if isinstance(cube_agg, tuple) else {}
    # Use the aggregate dict passed separately
    lines.append(f"  {'AGGREGATE':35s}")

    # --- Robot Movement ---
    lines.append("")
    lines.append("ROBOT NAVIGATION TEST (Nova Carter patrol)")
    lines.append("-" * 50)
    has_speed = any(row.get("after_mean_speed", "N/A") != "N/A" for row in robot_rows)
    if has_speed:
        lines.append(f"  {'Floor':<25s} {'Z-Drop':>8s} {'Pass':>5s} {'AvgSpd':>7s} {'MaxSpd':>7s} {'MinClr':>7s}")
        lines.append(f"  {'-'*25:<25s} {'-'*8:>8s} {'-'*5:>5s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s}")
        for row in robot_rows:
            room = row.get("room", row.get("camera", "?"))[:25]
            a = str(row["after_z_drop"])
            ap = "PASS" if row.get("after_pass") is True else ("FAIL" if row.get("after_pass") is False else "N/A")
            spd = str(row.get("after_mean_speed", "N/A"))
            mspd = str(row.get("after_max_speed", "N/A"))
            clr = str(row.get("after_min_clearance", "N/A"))
            lines.append(f"  {room:<25s} {a:>8s} {ap:>5s} {spd:>7s} {mspd:>7s} {clr:>7s}")
    else:
        lines.append(f"  {'Floor':<35s} {'Before Z-Drop':>14s} {'After Z-Drop':>14s}")
        lines.append(f"  {'-'*35:<35s} {'-'*14:>14s} {'-'*14:>14s}")
        for row in robot_rows:
            room = row.get("room", row.get("camera", "?"))
            b = str(row["before_z_drop"])
            a = str(row["after_z_drop"])
            bp = "PASS" if row.get("before_pass") is True else ("FAIL" if row.get("before_pass") is False else "N/A")
            ap = "PASS" if row.get("after_pass") is True else ("FAIL" if row.get("after_pass") is False else "N/A")
            lines.append(f"  {room:<35s} {b:>8s} {bp:>5s} {a:>8s} {ap:>5s}")

    # --- Door Physics ---
    lines.append("")
    lines.append("DOOR PHYSICS TEST")
    lines.append("-" * 50)
    lines.append(f"  Doors found:    {door_summary['before_total_doors']:>5d}  ->  {door_summary['after_total_doors']:>5d}")
    lines.append(f"  Doors passed:   {door_summary['before_total_passed']:>5d}  ->  {door_summary['after_total_passed']:>5d}")
    if door_summary["unique_doors_measured"] > 0:
        lines.append(f"  Mean delta:     {'N/A':>5s}  ->  {door_summary['mean_delta']:5.1f}°")
        lines.append(f"  Min delta:      {'N/A':>5s}  ->  {door_summary['min_delta']:5.1f}°")
        lines.append(f"  Max delta:      {'N/A':>5s}  ->  {door_summary['max_delta']:5.1f}°")
        lines.append(f"  Doors measured (in camera range): {door_summary['unique_doors_measured']}")
        if door_summary.get("stalled_doors", 0) > 0:
            lines.append(f"  Stalled doors (collision blocked): {door_summary['stalled_doors']}")

    # --- Per-door detail ---
    if door_list:
        has_native = any(d.get("peak_drive_torque", 0) != 0 for d in door_list)
        lines.append("")
        if has_native:
            lines.append("  Per-Door Details (after physics):")
            lines.append(f"  {'Door Name':<22s} {'Delta':>6s} {'Torque':>7s} {'Contact':>8s} {'AngVel':>7s} {'Pass':>5s} {'Stall':>6s}")
            lines.append(f"  {'-'*22:<22s} {'-'*6:>6s} {'-'*7:>7s} {'-'*8:>8s} {'-'*7:>7s} {'-'*5:>5s} {'-'*6:>6s}")
            for d in door_list:
                p = "PASS" if d["passed"] else "FAIL"
                s = "YES" if d.get("stall_detected") else ""
                lines.append(f"  {d['name']:<22s} {d['delta']:6.1f} {d.get('peak_drive_torque', 0):7.0f} "
                             f"{d.get('contact_impulse_total', 0):8.1f} {d.get('peak_angular_velocity', 0):7.1f} "
                             f"{p:>5s} {s:>6s}")
        else:
            lines.append("  Per-Door Angular Deltas (after physics):")
            lines.append(f"  {'Door Name':<25s} {'Initial':>8s} {'Final':>8s} {'Delta':>8s} {'Pass':>6s}")
            lines.append(f"  {'-'*25:<25s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*6:>6s}")
            for d in door_list:
                p = "PASS" if d["passed"] else "FAIL"
                lines.append(f"  {d['name']:<25s} {d['initial_angle']:8.1f} {d['final_angle']:8.1f} {d['delta']:8.1f} {p:>6s}")

    # --- Overall ---
    lines.append("")
    lines.append(sep)
    lines.append("KEY IMPROVEMENTS")
    lines.append(sep)

    b_stats = safe_get(before_data.get("building_stats"), "stats", default={})
    a_stats = safe_get(after_data.get("building_stats"), "stats", default={})
    lines.append(f"  Colliders added:         {b_stats.get('collider_count', 0):>6,d}  ->  {a_stats.get('collider_count', 0):>6,d}")
    lines.append(f"  Rigid bodies:            {b_stats.get('rigid_body_count', 0):>6,d}  ->  {a_stats.get('rigid_body_count', 0):>6,d}")
    lines.append(f"  Revolute joints:         {b_stats.get('revolute_joint_count', 0):>6,d}  ->  {a_stats.get('revolute_joint_count', 0):>6,d}")
    lines.append(f"  Driven joints:           {b_stats.get('driven_joint_count', 0):>6,d}  ->  {a_stats.get('driven_joint_count', 0):>6,d}")
    lines.append(f"  Door pass rate:          {door_summary['before_total_passed']}/{door_summary['before_total_doors']}  ->  {door_summary['after_total_passed']}/{door_summary['after_total_doors']}")
    lines.append("")

    return "\n".join(lines)


def write_csv(stats_rows, cube_rows, robot_rows, door_summary, door_list, output_path):
    """Write flat CSV for spreadsheet import."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)

        # Section 1: Building Stats
        w.writerow(["Section", "Metric", "Before", "After"])
        for row in stats_rows:
            if row["metric"].startswith("screenshot_"):
                continue
            w.writerow(["Building Stats", row["metric"], row["before"], row["after"]])
        w.writerow([])

        # Section 2: Cube Drop (per-camera summary)
        w.writerow(["Section", "Camera", "Room", "Before Landed", "Before Total",
                     "After Landed", "After Total", "Frames To Settle"])
        for row in [r for r in cube_rows if "cube_name" not in r and "before_landed" in r]:
            w.writerow(["Cube Drop", row["camera"], row["room"],
                        row["before_landed"], row["before_total"],
                        row["after_landed"], row["after_total"],
                        row.get("after_frames_to_settle", "N/A")])
        w.writerow([])

        # Section 3: Robot Movement
        w.writerow(["Section", "Camera", "Room", "Before Z-Drop", "After Z-Drop",
                     "Before Pass", "After Pass", "Mean Speed", "Max Speed", "Min Clearance"])
        for row in robot_rows:
            w.writerow(["Robot Movement", row["camera"], row["room"],
                        row["before_z_drop"], row["after_z_drop"],
                        row["before_pass"], row["after_pass"],
                        row.get("after_mean_speed", "N/A"),
                        row.get("after_max_speed", "N/A"),
                        row.get("after_min_clearance", "N/A")])
        w.writerow([])

        # Section 4: Door Physics
        w.writerow(["Section", "Metric", "Before", "After"])
        w.writerow(["Door Physics", "Total Doors", door_summary["before_total_doors"], door_summary["after_total_doors"]])
        w.writerow(["Door Physics", "Total Passed", door_summary["before_total_passed"], door_summary["after_total_passed"]])
        w.writerow(["Door Physics", "Mean Delta (deg)", "N/A", door_summary["mean_delta"]])
        w.writerow(["Door Physics", "Min Delta (deg)", "N/A", door_summary["min_delta"]])
        w.writerow(["Door Physics", "Max Delta (deg)", "N/A", door_summary["max_delta"]])
        w.writerow(["Door Physics", "Stalled Doors", "N/A", door_summary.get("stalled_doors", 0)])
        w.writerow([])

        # Section 5: Per-door detail
        w.writerow(["Door Name", "Initial Angle", "Final Angle", "Delta", "Passed",
                     "Peak Torque", "Contact Impulse", "Peak Ang Vel", "Stall Detected"])
        for d in door_list:
            w.writerow([d["name"], d["initial_angle"], d["final_angle"], d["delta"], d["passed"],
                        d.get("peak_drive_torque", ""), d.get("contact_impulse_total", ""),
                        d.get("peak_angular_velocity", ""), d.get("stall_detected", "")])


def main():
    parser = argparse.ArgumentParser(description="Compare before/after validation metrics")
    parser.add_argument("--before", required=True, help="Directory with raw USD results")
    parser.add_argument("--after", required=True, help="Directory with physics USD results")
    parser.add_argument("--output", default="logs/validation", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading 'before' results...")
    before_data = load_results(args.before)
    print("Loading 'after' results...")
    after_data = load_results(args.after)

    # Extract metrics
    print("\nExtracting metrics...")
    stats_rows = extract_building_stats(before_data.get("building_stats"), after_data.get("building_stats"))
    cube_rows, cube_agg = extract_cube_metrics(before_data.get("cube_drop"), after_data.get("cube_drop"))
    robot_rows = extract_robot_metrics(before_data.get("robot_movement"), after_data.get("robot_movement"))
    door_summary, door_list = extract_door_metrics(before_data.get("door_physics"), after_data.get("door_physics"))

    # Format summary
    summary_text = format_summary(
        stats_rows, cube_rows, robot_rows, door_summary, door_list,
        before_data, after_data,
    )

    # Write outputs
    txt_path = output_dir / "comparison_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary -> {txt_path}")

    # JSON
    metrics_json = {
        "building_stats": stats_rows,
        "cube_drop": {"per_camera": [r for r in cube_rows if "cube_name" not in r and "before_landed" in r],
                      "aggregate": cube_agg},
        "robot_movement": robot_rows,
        "door_physics": {"summary": door_summary, "per_door": door_list},
    }
    json_path = output_dir / "comparison_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"JSON   -> {json_path}")

    # CSV
    csv_path = output_dir / "comparison_data.csv"
    write_csv(stats_rows, cube_rows, robot_rows, door_summary, door_list, csv_path)
    print(f"CSV    -> {csv_path}")

    # Print summary to stdout
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
