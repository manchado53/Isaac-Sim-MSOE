#!/usr/bin/env python3
"""
Door Discovery Script for MSOEBuildingTake2.usd

Discovers all door prims in the Innovation Center USD and produces:
  1. logs/door_discovery_report.txt  — human-readable report
  2. logs/door_catalog.json          — reusable JSON catalog for physics scripts

Three search strategies:
  a) Category folder: leaf elements under Geometry/Doors/
  b) FamilyData: entries under Geometry/FamilyData/Doors/
  c) Name search: any prim with "door" in name not already found
"""

import os
import sys
import json
import time
from datetime import date

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

# Add project root for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.building_physics import (
    decode_tn_name,
    _find_mesh_child,
    _iter_leaf_elements,
    _find_wrapper,
)

# Output paths
_report_path = os.path.join(_project_dir, "logs", "door_discovery_report.txt")
_catalog_path = os.path.join(_project_dir, "logs", "door_catalog.json")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)

_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


def compute_bbox(stage, prim):
    """Compute world-space bounding box for a prim.

    Returns dict with center, dimensions, min, max — or None if bbox is empty.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn = rng.GetMin()
    mx = rng.GetMax()
    center = (mn + mx) / 2.0
    dims = mx - mn
    return {
        "center": [round(center[i], 4) for i in range(3)],
        "dimensions": [round(dims[i], 4) for i in range(3)],
        "min": [round(mn[i], 4) for i in range(3)],
        "max": [round(mx[i], 4) for i in range(3)],
    }


def compute_door_info(stage, prim, search_method):
    """Build a door info dict for one prim."""
    name = prim.GetName()
    return {
        "prim_path": str(prim.GetPath()),
        "name": name,
        "decoded_name": decode_tn_name(name).replace("___", " - ").replace("_", " "),
        "prim_type": prim.GetTypeName(),
        "instanceable": prim.IsInstanceable(),
        "search_method": search_method,
        "bbox": compute_bbox(stage, prim),
    }


def find_geometry_prim(stage, building_prim):
    """Navigate to the Geometry prim under the building."""
    outer = _find_wrapper(building_prim)
    if outer is None:
        return None
    for child in outer.GetChildren():
        if child.GetName() == "Geometry" and child.IsA(UsdGeom.Xform):
            return child
    return None


def find_child_by_name(prim, name):
    """Find a direct child prim by name."""
    for child in prim.GetChildren():
        if child.GetName() == name:
            return child
    return None


def main():
    global _report_file
    _report_file = open(_report_path, "w")
    t_start = time.time()

    log("=" * 70)
    log("DOOR DISCOVERY — MSOEBuildingTake2")
    log("=" * 70)

    # --- Load building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingTake2.usd"
    )
    log(f"\nLoading: {building_usd}")
    add_reference_to_stage(building_usd, prim_path="/World/Building")
    log("Building loaded.")

    building_prim = stage.GetPrimAtPath("/World/Building")
    geometry_prim = find_geometry_prim(stage, building_prim)
    if geometry_prim is None:
        log("ERROR: Could not find Geometry prim. Aborting.")
        _report_file.close()
        simulation_app.close()
        return

    log(f"Geometry prim: {geometry_prim.GetPath()}")

    # ======================================================================
    # Section 1: Structure Overview
    # ======================================================================
    log(f"\n{'=' * 70}")
    log("SECTION 1 — STRUCTURE OVERVIEW (Geometry category folders)")
    log("=" * 70)

    cat_folders = []
    for child in geometry_prim.GetChildren():
        child_count = len(child.GetChildren())
        cat_folders.append((child.GetName(), child_count))

    log(f"\n{'Folder Name':<50s} {'Children':>8s}")
    log("-" * 60)
    for name, count in sorted(cat_folders, key=lambda x: x[0]):
        log(f"  {name:<48s} {count:>8d}")
    log(f"\nTotal category folders: {len(cat_folders)}")

    # ======================================================================
    # Section 2: Door Discovery
    # ======================================================================
    all_doors = []

    # --- 2a: Category folder (Geometry/Doors/) ---
    log(f"\n{'=' * 70}")
    log("SECTION 2a — DOORS CATEGORY FOLDER (Geometry/Doors/)")
    log("=" * 70)

    doors_folder = find_child_by_name(geometry_prim, "Doors")
    category_doors = []
    if doors_folder:
        for leaf in _iter_leaf_elements(doors_folder):
            info = compute_door_info(stage, leaf, "category_folder")
            category_doors.append(info)
            all_doors.append(info)
        log(f"\nFound {len(category_doors)} door elements under Geometry/Doors/")
        # List door type groups
        type_groups = set()
        for child in doors_folder.GetChildren():
            type_groups.add(child.GetName())
        log(f"Door type groups: {len(type_groups)}")
        for tg in sorted(type_groups):
            log(f"  {decode_tn_name(tg)}")
    else:
        log("WARNING: No Doors folder found under Geometry.")

    log(f"\nDoor details:")
    for d in category_doors:
        bbox_str = ""
        if d["bbox"]:
            c = d["bbox"]["center"]
            dims = d["bbox"]["dimensions"]
            bbox_str = f"  center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  size=({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f})"
        inst = " [instanceable]" if d["instanceable"] else ""
        log(f"  {d['prim_path']}")
        log(f"    type={d['prim_type']}{inst}{bbox_str}")

    # --- 2b: FamilyData/Doors ---
    log(f"\n{'=' * 70}")
    log("SECTION 2b — FAMILY DATA DOORS (Geometry/FamilyData/Doors/)")
    log("=" * 70)

    family_data = find_child_by_name(geometry_prim, "FamilyData")
    family_doors = []
    if family_data:
        family_doors_folder = find_child_by_name(family_data, "Doors")
        if family_doors_folder:
            for leaf in _iter_leaf_elements(family_doors_folder):
                info = compute_door_info(stage, leaf, "family_data")
                family_doors.append(info)
                all_doors.append(info)
            log(f"\nFound {len(family_doors)} door family entries")
        else:
            log("No Doors subfolder under FamilyData.")
    else:
        log("No FamilyData folder found.")

    for d in family_doors:
        log(f"  {d['prim_path']}")
        log(f"    decoded: {d['decoded_name']}  type={d['prim_type']}")

    # --- 2c: Global name search ---
    log(f"\n{'=' * 70}")
    log("SECTION 2c — GLOBAL NAME SEARCH ('door' in prim name)")
    log("=" * 70)

    found_paths = {d["prim_path"] for d in all_doors}
    name_search_doors = []
    for prim in stage.Traverse():
        prim_name = prim.GetName().lower()
        if "door" in prim_name:
            prim_path = str(prim.GetPath())
            if prim_path not in found_paths:
                # Avoid adding parents/ancestors of already-found doors
                info = compute_door_info(stage, prim, "name_search")
                name_search_doors.append(info)
                found_paths.add(prim_path)

    all_doors.extend(name_search_doors)
    log(f"\nFound {len(name_search_doors)} additional door-related prims")

    for d in name_search_doors:
        log(f"  {d['prim_path']}")
        log(f"    decoded: {d['decoded_name']}  type={d['prim_type']}")

    # ======================================================================
    # Section 3: Summary
    # ======================================================================
    log(f"\n{'=' * 70}")
    log("SECTION 3 — SUMMARY")
    log("=" * 70)

    log(f"\nDoors by search method:")
    log(f"  category_folder:  {len(category_doors)}")
    log(f"  family_data:      {len(family_doors)}")
    log(f"  name_search:      {len(name_search_doors)}")
    log(f"  TOTAL:            {len(all_doors)}")

    # Unique door types (from decoded names, grouping by type group name)
    door_types = sorted(set(d["name"] for d in all_doors))
    log(f"\nUnique door names: {len(door_types)}")
    for dt in door_types:
        decoded = decode_tn_name(dt).replace("___", " - ").replace("_", " ")
        log(f"  {decoded}")

    # ======================================================================
    # JSON Catalog
    # ======================================================================
    catalog = {
        "building_usd": "MSOEBuildingTake2.usd",
        "discovery_date": str(date.today()),
        "total_doors": len(all_doors),
        "doors": all_doors,
        "door_types": door_types,
        "counts_by_method": {
            "category_folder": len(category_doors),
            "family_data": len(family_doors),
            "name_search": len(name_search_doors),
        },
    }

    with open(_catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    log(f"\nJSON catalog written: {_catalog_path}")

    t_end = time.time()
    log(f"\n{'=' * 70}")
    log("Done!")
    log(f"  Report  -> {_report_path}")
    log(f"  Catalog -> {_catalog_path}")
    log(f"  Runtime: {t_end - t_start:.1f}s")
    log("=" * 70)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
