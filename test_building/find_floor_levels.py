#!/usr/bin/env python3
"""
Analyze TestBuilding.usd to find actual floor Z levels.
Examines mesh geometry to determine where floor slabs are.

Run with: sbatch submit_isaac_job.sh test_building/find_floor_levels.py
"""
import os
import sys
import json
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from pxr import Usd, UsdGeom, Gf

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
USD_PATH = os.path.join(os.path.dirname(PROJECT_DIR),
                        "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Floor Level Finder for TestBuilding.usd")
    print("=" * 60)
    sys.stdout.flush()

    omni.usd.get_context().open_stage(USD_PATH)
    for i in range(3):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"Units: {'cm' if meters_per_unit == 0.01 else 'm'}")

    # Collect Z positions of all mesh vertices
    print("\nAnalyzing mesh geometry...")
    sys.stdout.flush()

    all_z_values = []
    mesh_info = []  # (name, z_min, z_max, z_center, is_horizontal)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        name = prim.GetName()
        path = str(prim.GetPath())
        parent_name = str(prim.GetParent().GetName()) if prim.GetParent() else ""

        # Get world-space bounding box
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            continue

        bb_min = bbox_range.GetMin()
        bb_max = bbox_range.GetMax()

        z_min = float(bb_min[2])
        z_max = float(bb_max[2])
        z_thickness = z_max - z_min
        xy_width = float(bb_max[0] - bb_min[0])
        xy_depth = float(bb_max[1] - bb_min[1])
        z_center = (z_min + z_max) / 2.0

        # Determine if this is a horizontal element (floor/ceiling)
        # Horizontal = thin in Z, wide in XY
        is_horizontal = (z_thickness < 50 and xy_width > 100 and xy_depth > 100)

        # Determine if this is a vertical element (wall)
        is_vertical = (z_thickness > 100 and (xy_width < 50 or xy_depth < 50))

        mesh_info.append({
            "name": parent_name,
            "path": path,
            "z_min": z_min,
            "z_max": z_max,
            "z_center": z_center,
            "z_thickness": z_thickness,
            "xy_width": xy_width,
            "xy_depth": xy_depth,
            "is_horizontal": is_horizontal,
            "is_vertical": is_vertical,
        })

        # Collect Z values for histogram
        all_z_values.extend([z_min, z_max])

    print(f"  Total meshes analyzed: {len(mesh_info)}")

    # --- Analyze horizontal elements (floors/ceilings) ---
    horizontals = [m for m in mesh_info if m["is_horizontal"]]
    verticals = [m for m in mesh_info if m["is_vertical"]]
    print(f"  Horizontal elements (floors/ceilings): {len(horizontals)}")
    print(f"  Vertical elements (walls): {len(verticals)}")

    if horizontals:
        print("\n  Horizontal elements (potential floor slabs):")
        # Group by Z center (within 20cm tolerance)
        h_sorted = sorted(horizontals, key=lambda x: x["z_center"])
        groups = []
        current_group = [h_sorted[0]]
        for h in h_sorted[1:]:
            if h["z_center"] - current_group[-1]["z_center"] < 20:
                current_group.append(h)
            else:
                groups.append(current_group)
                current_group = [h]
        groups.append(current_group)

        print(f"  Found {len(groups)} horizontal Z levels:")
        floor_z_levels = []
        for g in groups:
            z_avg = np.mean([m["z_center"] for m in g])
            z_min_g = min(m["z_min"] for m in g)
            z_max_g = max(m["z_max"] for m in g)
            total_area = sum(m["xy_width"] * m["xy_depth"] for m in g)
            print(f"    Z={z_avg:.1f} cm (range {z_min_g:.1f}-{z_max_g:.1f}, "
                  f"{len(g)} elements, area={total_area:.0f} cm²)")
            floor_z_levels.append({
                "z_center": float(z_avg),
                "z_min": float(z_min_g),
                "z_max": float(z_max_g),
                "count": len(g),
                "total_area": float(total_area),
            })

    # --- Analyze wall Z ranges to find floor extents ---
    if verticals:
        print(f"\n  Wall Z ranges:")
        wall_z_mins = [m["z_min"] for m in verticals]
        wall_z_maxs = [m["z_max"] for m in verticals]

        # Group walls by their Z range to identify floors
        # Walls on the same floor will have similar z_min and z_max
        wall_sorted = sorted(verticals, key=lambda x: x["z_min"])

        # Find clusters of wall z_min values
        z_min_values = sorted(set([round(m["z_min"], -1) for m in verticals]))  # round to 10cm
        z_max_values = sorted(set([round(m["z_max"], -1) for m in verticals]))

        print(f"    Unique wall Z-min levels (rounded to 10cm): {len(z_min_values)}")
        for z in z_min_values[:20]:
            count = sum(1 for m in verticals if abs(m["z_min"] - z) < 15)
            print(f"      Z-min = {z:.0f} cm ({count} walls)")

        print(f"\n    Unique wall Z-max levels (rounded to 10cm): {len(z_max_values)}")
        for z in z_max_values[:20]:
            count = sum(1 for m in verticals if abs(m["z_max"] - z) < 15)
            print(f"      Z-max = {z:.0f} cm ({count} walls)")

    # --- Z histogram of all geometry ---
    print("\n  Z-value histogram (all mesh boundaries):")
    z_arr = np.array(all_z_values)
    z_hist_min = float(np.min(z_arr))
    z_hist_max = float(np.max(z_arr))
    bins = np.arange(z_hist_min, z_hist_max + 50, 50)  # 50cm bins
    hist, bin_edges = np.histogram(z_arr, bins=bins)

    for i, count in enumerate(hist):
        if count > 0:
            bar = "#" * min(count, 50)
            print(f"    Z {bin_edges[i]:7.0f} - {bin_edges[i+1]:7.0f}: {count:4d} {bar}")

    # --- Categorize all meshes by name patterns ---
    print("\n  Mesh categories by Revit name:")
    categories = {}
    for m in mesh_info:
        # Extract category from name (e.g., "Basic_Wall", "Floor", etc.)
        name = m["name"]
        if "Wall" in name or "wall" in name:
            cat = "Wall"
        elif "Floor" in name or "floor" in name:
            cat = "Floor"
        elif "Roof" in name or "roof" in name:
            cat = "Roof"
        elif "Ceiling" in name or "ceiling" in name:
            cat = "Ceiling"
        elif "Door" in name or "door" in name:
            cat = "Door"
        elif "Panel" in name or "panel" in name:
            cat = "Panel"
        elif "Mullion" in name or "mullion" in name:
            cat = "Mullion"
        elif "Stair" in name or "stair" in name:
            cat = "Stair"
        else:
            cat = "Other"

        if cat not in categories:
            categories[cat] = {"count": 0, "z_mins": [], "z_maxs": []}
        categories[cat]["count"] += 1
        categories[cat]["z_mins"].append(m["z_min"])
        categories[cat]["z_maxs"].append(m["z_max"])

    for cat, info in sorted(categories.items()):
        z_lo = min(info["z_mins"])
        z_hi = max(info["z_maxs"])
        print(f"    {cat}: {info['count']} meshes, Z range [{z_lo:.0f}, {z_hi:.0f}] cm")

    # --- Save analysis ---
    analysis = {
        "total_meshes": len(mesh_info),
        "horizontal_elements": len(horizontals),
        "vertical_elements": len(verticals),
        "floor_z_levels": floor_z_levels if horizontals else [],
        "categories": {cat: {"count": info["count"],
                              "z_min": float(min(info["z_mins"])),
                              "z_max": float(max(info["z_maxs"]))}
                       for cat, info in categories.items()},
        "z_range": {"min": float(z_hist_min), "max": float(z_hist_max)},
    }
    with open(os.path.join(OUTPUT_DIR, "floor_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved analysis to output/floor_analysis.json")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    sys.stdout.flush()

    simulation_app.close()


if __name__ == "__main__":
    main()
