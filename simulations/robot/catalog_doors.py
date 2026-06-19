#!/usr/bin/env python3
"""
Door Catalog — Pure USD Stage Traversal (no Isaac Sim required)

Loads the building USD and catalogs every door prim. For each door, extracts:
  - Prim path and Revit element ID
  - World position (translation from xformOp:transform)
  - Bounding box / mesh extents
  - Bound material names
  - Inferred floor level
  - Single vs double door classification
  - Child mesh names
  - Full 4x4 transform matrices

Output:
  simulations/robot/door_catalog.json  — structured data for all doors
  stdout — human-readable summary table

Usage (login node, no GPU needed):
  conda run -n vlm python simulations/robot/catalog_doors.py
"""

import json
import math
import os
import sys

from pxr import Usd, UsdGeom, UsdShade, Gf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUTPUT_JSON = os.path.join(_script_dir, "door_catalog.json")

# Floor Z ranges (approximate, in metres — Revit export uses metres for this building)
# These will be refined after seeing actual Z distribution.
# From CLAUDE.md: Floor 4 camera Z = 27.8 m, door hinge at cm values but
# the existing door_snapshot.py uses metres for bbox midpoints.
# We'll auto-detect floor boundaries from the data.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_bound_material_name(prim, stage):
    """Get the name of the material bound to a prim (or its children)."""
    materials = set()

    # Check the prim itself
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.ComputeBoundMaterial()
    if bound and bound[0]:
        mat_path = bound[0].GetPath().pathString
        mat_name = mat_path.split("/")[-1]
        materials.add(mat_name)

    # Check child meshes
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            child_binding = UsdShade.MaterialBindingAPI(child)
            child_bound = child_binding.ComputeBoundMaterial()
            if child_bound and child_bound[0]:
                mat_path = child_bound[0].GetPath().pathString
                mat_name = mat_path.split("/")[-1]
                materials.add(mat_name)

    return sorted(materials) if materials else ["unknown"]


def get_mesh_children(prim):
    """Get info about child mesh prims under a door prim."""
    meshes = []
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            mesh_info = {"name": child.GetName(), "path": str(child.GetPath())}

            # Get the xformOp:transform matrix
            xformable = UsdGeom.Xformable(child)
            for xop in xformable.GetOrderedXformOps():
                if xop.GetOpName() == "xformOp:transform":
                    mat = xop.Get()
                    if mat is not None:
                        # Extract translation (row 3 in USD row-vector convention)
                        mesh_info["translation_cm"] = [
                            float(mat[3][0]),
                            float(mat[3][1]),
                            float(mat[3][2]),
                        ]
                        # Store full matrix as nested list
                        mesh_info["matrix_4x4"] = [
                            [float(mat[r][c]) for c in range(4)]
                            for r in range(4)
                        ]
                        mesh_info["has_xform_transform"] = True
                    break
            else:
                mesh_info["has_xform_transform"] = False

            # Get mesh extent if available
            mesh_prim = UsdGeom.Mesh(child)
            extent_attr = mesh_prim.GetExtentAttr()
            if extent_attr and extent_attr.HasValue():
                ext = extent_attr.Get()
                if ext and len(ext) == 2:
                    mesh_info["extent_min"] = [float(ext[0][0]), float(ext[0][1]), float(ext[0][2])]
                    mesh_info["extent_max"] = [float(ext[1][0]), float(ext[1][1]), float(ext[1][2])]

            meshes.append(mesh_info)

    return meshes


def get_door_type_from_path(path_str):
    """Extract the door type/family from the USD path hierarchy."""
    parts = path_str.split("/")
    for j, part in enumerate(parts):
        if part == "Doors" and j + 1 < len(parts):
            return parts[j + 1]  # e.g. "RS___Curtain_Wall___Type_FG___New"
    return "unknown"


def infer_floor(z_metres):
    """Infer floor number from Z position in metres.

    Calibrated from door_snapshot.py and the existing project convention:
      Floor 1: Z ≈ 10m (ground/basement level)
      Floor 2: Z ≈ 13–16m
      Floor 3: Z ≈ 19–20m
      Floor 4: Z ≈ 23–24m
      Floor 5: Z ≈ 27–28m (existing training door Doors_13606713 is here)
      Floor 6: Z ≈ 30–33m (roof/mechanical)
    """
    if z_metres < 12:
        return 1
    elif z_metres < 17:
        return 2
    elif z_metres < 22:
        return 3
    elif z_metres < 26:
        return 4
    elif z_metres < 30:
        return 5
    else:
        return 6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[DoorCatalog] Loading USD: {USD_PATH}")
    if not os.path.isfile(USD_PATH):
        print(f"ERROR: USD file not found: {USD_PATH}")
        sys.exit(1)

    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        print("ERROR: Could not open USD stage")
        sys.exit(1)

    print(f"[DoorCatalog] Stage opened. Root prims: "
          f"{[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}")

    # BBox cache for computing world bounds
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    # ---------------------------------------------------------------------------
    # Traverse and find all door prims
    # ---------------------------------------------------------------------------

    print("\n[DoorCatalog] Scanning for door prims ...")
    doors = []
    door_count_raw = 0

    for prim in stage.TraverseAll():
        path_str = prim.GetPath().pathString
        prim_name = prim.GetName()

        # Must be under a /Doors/ category in the hierarchy
        if "/Doors/" not in path_str:
            continue

        # Must start with "Doors_" (the instance prim, not child meshes)
        if not prim_name.startswith("Doors_"):
            continue

        # Must be an Xform (parent of mesh children), not a Mesh itself
        if prim.GetTypeName() not in ("Xform", ""):
            continue

        door_count_raw += 1

        # Check for child meshes with xformOp:transform
        mesh_children = get_mesh_children(prim)
        meshes_with_xform = [m for m in mesh_children if m.get("has_xform_transform", False)]

        # Compute world bounding box
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            if bbox_range.IsEmpty():
                continue
            bb_min = bbox_range.GetMin()
            bb_max = bbox_range.GetMax()
            bb_mid = bbox_range.GetMidpoint()
            bb_size = bbox_range.GetSize()
        except Exception as e:
            print(f"  [WARN] Could not compute bbox for {prim_name}: {e}")
            continue

        # Skip degenerate prims
        max_dim = max(bb_size[0], bb_size[1], bb_size[2])
        if max_dim < 0.3:
            continue

        # Extract position
        cx, cy, cz = float(bb_mid[0]), float(bb_mid[1]), float(bb_mid[2])
        width_x = float(bb_size[0])
        width_y = float(bb_size[1])
        height = float(bb_size[2])

        # Determine thin axis (face normal direction)
        if width_x < width_y:
            approach_axis = "X"
            door_width = width_y
            door_thickness = width_x
        else:
            approach_axis = "Y"
            door_width = width_x
            door_thickness = width_y

        # Door type from path
        door_type = get_door_type_from_path(path_str)

        # Materials
        materials = get_bound_material_name(prim, stage)

        # Infer floor
        floor = infer_floor(cz)

        # Get the parent xform matrix if available
        parent_matrix = None
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        for xop in xform_ops:
            if xop.GetOpName() == "xformOp:transform":
                mat = xop.Get()
                if mat is not None:
                    parent_matrix = [
                        [float(mat[r][c]) for c in range(4)]
                        for r in range(4)
                    ]
                break

        # Estimate hinge position from mesh translations
        # The hinge is typically at the edge of the door (min or max X/Y of mesh origins)
        hinge_estimate = None
        if meshes_with_xform:
            xs = [m["translation_cm"][0] for m in meshes_with_xform]
            ys = [m["translation_cm"][1] for m in meshes_with_xform]
            # Hinge is at one edge — use min or max depending on door orientation
            hinge_estimate = {
                "min_x_cm": min(xs),
                "max_x_cm": max(xs),
                "min_y_cm": min(ys),
                "max_y_cm": max(ys),
                "note": "Hinge is likely at one edge — verify visually with snapshots",
            }

        door_info = {
            "id": prim_name,
            "revit_id": prim_name.replace("Doors_", ""),
            "prim_path": path_str,
            "door_type_family": door_type,
            "position": {
                "center_x": cx,
                "center_y": cy,
                "center_z": cz,
                "units": "metres (from bbox midpoint)",
            },
            "bounding_box": {
                "min": [float(bb_min[0]), float(bb_min[1]), float(bb_min[2])],
                "max": [float(bb_max[0]), float(bb_max[1]), float(bb_max[2])],
                "size": [width_x, width_y, height],
                "units": "metres",
            },
            "dimensions": {
                "width": round(door_width, 4),
                "thickness": round(door_thickness, 4),
                "height": round(height, 4),
                "approach_axis": approach_axis,
                "units": "metres",
            },
            "floor": floor,
            "materials": materials,
            "mesh_count": len(mesh_children),
            "meshes_with_xform_transform": len(meshes_with_xform),
            "mesh_children": mesh_children,
            "parent_matrix_4x4": parent_matrix,
            "hinge_estimate": hinge_estimate,
            "is_double_door": False,  # will be updated below
            "double_door_pair": None,
        }

        doors.append(door_info)

    print(f"[DoorCatalog] Found {door_count_raw} raw door prims, "
          f"{len(doors)} with valid bounding boxes")

    # ---------------------------------------------------------------------------
    # Identify double doors (pairs within 2m of each other on the same floor)
    # ---------------------------------------------------------------------------

    print("\n[DoorCatalog] Identifying double-door pairs ...")
    for i, a in enumerate(doors):
        for j, b in enumerate(doors):
            if j <= i:
                continue
            if a["floor"] != b["floor"]:
                continue
            dist = math.sqrt(
                (a["position"]["center_x"] - b["position"]["center_x"]) ** 2
                + (a["position"]["center_y"] - b["position"]["center_y"]) ** 2
            )
            if dist < 2.0:
                a["is_double_door"] = True
                b["is_double_door"] = True
                a["double_door_pair"] = b["id"]
                b["double_door_pair"] = a["id"]
                print(f"  Double door: {a['id']} <-> {b['id']} "
                      f"({dist:.2f}m apart, floor {a['floor']})")

    # Sort by floor then X position
    doors.sort(key=lambda d: (d["floor"], d["position"]["center_x"]))

    # ---------------------------------------------------------------------------
    # Check for revolute joints and tag doors that have them
    # ---------------------------------------------------------------------------

    print("\n[DoorCatalog] Scanning for revolute joints (articulated doors) ...")
    # Build a set of door IDs that have hinge joints
    doors_with_joints = set()
    joint_count = 0
    for prim in stage.TraverseAll():
        type_name = prim.GetTypeName()
        if "Joint" in type_name or "Revolute" in type_name:
            path_str = prim.GetPath().pathString
            if "door" in path_str.lower() or "Door" in path_str:
                joint_count += 1
                # Extract the parent door ID (joint is a child of the door prim)
                parent_path = prim.GetParent().GetPath().pathString
                parent_name = prim.GetParent().GetName()
                if parent_name.startswith("Doors_"):
                    doors_with_joints.add(parent_name)
                print(f"  Joint found: {path_str} (type: {type_name})")

    if joint_count == 0:
        print("  No revolute joints found on door prims")
    else:
        print(f"  {joint_count} joints found on {len(doors_with_joints)} door prims")

    # Tag doors with joint info
    for d in doors:
        d["has_revolute_joint"] = d["id"] in doors_with_joints

    # Determine rotation readiness:
    # A door is "rotation ready" if it has child meshes with xformOp:transform
    # (needed for the direct 4x4 matrix rotation method used in door_static_collection.py)
    for d in doors:
        d["rotation_ready"] = d["meshes_with_xform_transform"] > 0

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 130}")
    print(f"DOOR CATALOG — {len(doors)} doors found")
    print(f"{'=' * 130}")

    # Separate single and double doors
    single_doors = [d for d in doors if not d["is_double_door"]]
    double_doors = [d for d in doors if d["is_double_door"]]

    print(f"\nSingle doors: {len(single_doors)}  |  Double door leaves: {len(double_doors)}")

    header = (f"{'#':>3}  {'Door ID':<20}  {'Fl':>2}  {'X(m)':>8}  {'Y(m)':>8}  "
              f"{'Z(m)':>6}  {'W(m)':>5}  {'H(m)':>5}  {'Axis':>4}  {'Mesh':>4}  "
              f"{'D/S':>3}  {'Jnt':>3}  {'Rot':>3}  {'Materials':<25}  Type Family")
    print(f"\n{header}")
    print("-" * 160)

    for i, d in enumerate(doors):
        ds = "D" if d["is_double_door"] else "S"
        jnt = "Y" if d.get("has_revolute_joint") else " "
        rot = "Y" if d.get("rotation_ready") else " "
        mats = ", ".join(d["materials"])[:25]
        type_short = d["door_type_family"][:35]
        print(f"{i+1:>3}  {d['id']:<20}  {d['floor']:>2}  "
              f"{d['position']['center_x']:>8.2f}  "
              f"{d['position']['center_y']:>8.2f}  "
              f"{d['position']['center_z']:>6.1f}  "
              f"{d['dimensions']['width']:>5.2f}  "
              f"{d['dimensions']['height']:>5.2f}  "
              f"{d['dimensions']['approach_axis']:>4}  "
              f"{d['mesh_count']:>4}  "
              f"{ds:>3}  "
              f"{jnt:>3}  "
              f"{rot:>3}  "
              f"{mats:<25}  {type_short}")

    # Floor distribution
    from collections import Counter
    floor_counts = Counter(d["floor"] for d in doors)
    print(f"\nFloor distribution:")
    for fl in sorted(floor_counts):
        count = floor_counts[fl]
        singles = sum(1 for d in doors if d["floor"] == fl and not d["is_double_door"])
        doubles = sum(1 for d in doors if d["floor"] == fl and d["is_double_door"])
        print(f"  Floor {fl}: {count} doors ({singles} single, {doubles} double leaves)")

    # Material distribution
    all_mats = Counter()
    for d in doors:
        for m in d["materials"]:
            all_mats[m] += 1
    print(f"\nMaterial distribution:")
    for mat, count in all_mats.most_common():
        print(f"  {mat}: {count} doors")

    # Door type family distribution
    type_counts = Counter(d["door_type_family"] for d in doors)
    print(f"\nDoor type families:")
    for dtype, count in type_counts.most_common():
        print(f"  {dtype}: {count} doors")

    # Z position histogram for floor calibration
    z_values = sorted(set(round(d["position"]["center_z"], 1) for d in doors))
    print(f"\nUnique Z values (for floor calibration): {z_values}")

    # ---------------------------------------------------------------------------
    # Highlight the existing training door
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 80}")
    print("EXISTING TRAINING DOOR (for reference):")
    for d in doors:
        if d["id"] == "Doors_13606713":
            print(f"  ID: {d['id']}")
            print(f"  Path: {d['prim_path']}")
            print(f"  Position: ({d['position']['center_x']:.2f}, "
                  f"{d['position']['center_y']:.2f}, "
                  f"{d['position']['center_z']:.2f}) m")
            print(f"  Floor: {d['floor']}")
            print(f"  Dimensions: W={d['dimensions']['width']:.2f}m, "
                  f"H={d['dimensions']['height']:.2f}m, "
                  f"T={d['dimensions']['thickness']:.4f}m")
            print(f"  Materials: {d['materials']}")
            print(f"  Meshes: {d['mesh_count']} "
                  f"({d['meshes_with_xform_transform']} with xformOp:transform)")
            print(f"  Has revolute joint: {d.get('has_revolute_joint', False)}")
            print(f"  Double door: {d['is_double_door']} "
                  f"(pair: {d['double_door_pair']})")
            break
    else:
        print("  WARNING: Doors_13606713 not found in catalog!")

    # ---------------------------------------------------------------------------
    # Best candidates for data collection
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 80}")
    print("BEST CANDIDATES FOR DATA COLLECTION")
    print("(Doors with revolute joints + xformOp:transform meshes = rotation-ready)")
    print(f"{'=' * 80}\n")

    # Doors with revolute joints are the best candidates — they were
    # explicitly set up for rotation in the USD
    rotation_ready = [d for d in doors if d.get("has_revolute_joint")
                      and d["meshes_with_xform_transform"] > 0]
    joint_only = [d for d in doors if d.get("has_revolute_joint")
                  and d["meshes_with_xform_transform"] == 0]
    mesh_only = [d for d in doors if not d.get("has_revolute_joint")
                 and d["meshes_with_xform_transform"] > 0]

    print(f"Rotation-ready (joint + xform meshes): {len(rotation_ready)}")
    print(f"Joint only (no xform meshes): {len(joint_only)}")
    print(f"Xform meshes only (no joint): {len(mesh_only)}")

    if rotation_ready:
        print(f"\n--- Rotation-ready doors ---")
        for d in rotation_ready:
            ds = "DOUBLE" if d["is_double_door"] else "SINGLE"
            mats = ", ".join(d["materials"])
            print(f"  {d['id']:<20} Floor {d['floor']}  "
                  f"Z={d['position']['center_z']:.1f}m  "
                  f"W={d['dimensions']['width']:.2f}m  "
                  f"H={d['dimensions']['height']:.2f}m  "
                  f"{ds:<6}  {d['mesh_count']} meshes  "
                  f"Materials: {mats}")

    if joint_only:
        print(f"\n--- Joint-only doors (may need mesh xform setup) ---")
        for d in joint_only:
            ds = "DOUBLE" if d["is_double_door"] else "SINGLE"
            mats = ", ".join(d["materials"])
            print(f"  {d['id']:<20} Floor {d['floor']}  "
                  f"Z={d['position']['center_z']:.1f}m  "
                  f"W={d['dimensions']['width']:.2f}m  "
                  f"{ds:<6}  Materials: {mats}")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------

    # Strip large matrix data from mesh_children for the summary JSON
    # (keep it in a separate detailed section)
    output = {
        "metadata": {
            "usd_file": USD_PATH,
            "total_doors": len(doors),
            "single_doors": len(single_doors),
            "double_door_leaves": len(double_doors),
            "floors": dict(sorted(floor_counts.items())),
        },
        "doors": doors,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DoorCatalog] Saved catalog to {OUTPUT_JSON}")
    print(f"[DoorCatalog] File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
