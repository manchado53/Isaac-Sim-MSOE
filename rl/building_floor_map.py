"""Extract per-floor bounding boxes from the Innovation Center USD.

Uses sim_utils.building_physics.discover_floor_levels() to cluster floor
elements by Z-height, then queries UsdGeom.BBoxCache for XY extents per
floor. Outputs training/nova_carter_patrol/floor_map.json.

All output values are in meters (USD is in feet; metersPerUnit=0.3048).

Usage (no GPU needed):
    ~/isaacsim/python.sh training/nova_carter_patrol/building_floor_map.py
"""

import json
import os
import sys

# Add project root so sim_utils is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_root)

USD_PATH = os.path.join(
    _project_root,
    "assets", "environments", "msoe_building", "innovation_center_physics.usd",
)
OUTPUT_PATH = os.path.join(_script_dir, "floor_map.json")

FLOOR_LABELS = ["G", "1", "2", "3"]
FT_TO_M = 0.3048


def main():
    from pxr import Usd, UsdGeom, Gf

    if not os.path.exists(USD_PATH):
        raise FileNotFoundError(
            f"Physics USD not found: {USD_PATH}\n"
            "Run simulations/environment/setup_innovation_center_physics.py first."
        )

    print(f"[INFO] Opening USD: {USD_PATH}")
    stage = Usd.Stage.Open(USD_PATH)

    from sim_utils.building_physics import discover_floor_levels

    floor_z_ft = discover_floor_levels(stage, "/World/Building")
    print(f"[INFO] Discovered {len(floor_z_ft)} floor levels (feet): {floor_z_ft}")

    # BBoxCache for world-space bounds
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    building_prim = stage.GetPrimAtPath("/World/Building")
    world_bbox = bbox_cache.ComputeWorldBound(building_prim)
    world_range = world_bbox.GetRange()

    bmin = world_range.GetMin()
    bmax = world_range.GetMax()

    # Assign discovered Z values to floor labels by sorted order
    sorted_z = sorted(floor_z_ft)
    floor_map = {}
    for i, label in enumerate(FLOOR_LABELS):
        if i < len(sorted_z):
            z_ft = sorted_z[i]
        else:
            # Fallback: estimate from spacing if fewer floors detected
            z_ft = sorted_z[-1] + (i - len(sorted_z) + 1) * 13.0  # ~13ft typical floor height
            print(f"[WARNING] Floor '{label}' not detected — estimated z={z_ft:.1f} ft")

        floor_map[label] = {
            "z_m": round(z_ft * FT_TO_M, 3),
            "x_range_m": [round(bmin[0] * FT_TO_M, 3), round(bmax[0] * FT_TO_M, 3)],
            "y_range_m": [round(bmin[1] * FT_TO_M, 3), round(bmax[1] * FT_TO_M, 3)],
        }

    # Sanity check: corridor widths should be 2–20m
    x_span = floor_map["G"]["x_range_m"][1] - floor_map["G"]["x_range_m"][0]
    y_span = floor_map["G"]["y_range_m"][1] - floor_map["G"]["y_range_m"][0]
    print(f"[INFO] Building footprint: {x_span:.1f}m × {y_span:.1f}m")
    if x_span < 5 or y_span < 5:
        print("[WARNING] Building extents look too small — check metersPerUnit in the USD.")

    # Check floor Z separation (~3–4m expected)
    if len(sorted_z) >= 2:
        dz_m = (sorted_z[1] - sorted_z[0]) * FT_TO_M
        print(f"[INFO] Floor G → Floor 1 Z separation: {dz_m:.2f}m (expected 3–4m)")
        if not (2.5 <= dz_m <= 5.0):
            print("[WARNING] Floor Z separation outside expected range.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(floor_map, f, indent=2)
    print(f"[INFO] floor_map.json written to {OUTPUT_PATH}")
    print(json.dumps(floor_map, indent=2))


if __name__ == "__main__":
    main()
