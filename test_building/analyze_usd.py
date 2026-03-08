#!/usr/bin/env python3
"""
Analyze and compare USD files side-by-side.

Compares TestBuilding.usd and MSOEBuildingTake2.usd to determine which has
more embedded geometry (meshes, interior walls, doors, etc.) and fewer
missing external references.

Run with:
  sbatch submit_isaac_job.sh test_building/analyze_usd.py
"""

import os
import sys
import json

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from pxr import Usd, UsdGeom, Sdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
BUILDING_DIR = os.path.join(os.path.dirname(PROJECT_DIR),
                            "assets", "environments", "msoe_building")

USD_FILES = [
    os.path.join(BUILDING_DIR, "TestBuilding.usd"),
    os.path.join(BUILDING_DIR, "MSOEBuildingTake2.usd"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_usd(usd_path):
    """Analyze a single USD file and return a summary dict."""
    filename = os.path.basename(usd_path)
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {filename} ({os.path.getsize(usd_path) / 1e6:.1f} MB)")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    # Open stage
    omni.usd.get_context().open_stage(usd_path)
    for _ in range(3):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ERROR: Failed to open {filename}")
        return None

    # Stage metadata
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  Up axis: {up_axis}")
    print(f"  Meters per unit: {meters_per_unit} ({'cm' if meters_per_unit == 0.01 else 'ft' if abs(meters_per_unit - 0.3048) < 0.01 else 'm'})")

    # Count prims by type
    total_prims = 0
    mesh_count = 0
    xform_count = 0
    other_prim_types = {}
    mesh_names = []

    # Categorize meshes
    categories = {
        "walls": [],
        "doors": [],
        "floors_ceilings": [],
        "windows": [],
        "hvac": [],
        "structural": [],
        "stairs_ramps": [],
        "fixtures": [],
        "other": [],
    }

    def categorize_mesh(name):
        name_lower = name.lower()
        if any(k in name_lower for k in ["wall", "partition", "shaftwall"]):
            return "walls"
        elif any(k in name_lower for k in ["door", "gate"]):
            return "doors"
        elif any(k in name_lower for k in ["floor", "ceiling", "slab", "roof"]):
            return "floors_ceilings"
        elif any(k in name_lower for k in ["window", "glazed", "glazing", "curtain_wall", "mullion"]):
            return "windows"
        elif any(k in name_lower for k in ["duct", "diffuser", "grille", "vav", "exhaust", "supply"]):
            return "hvac"
        elif any(k in name_lower for k in ["w_shape", "hss_", "c_shape", "footing", "beam", "column"]):
            return "structural"
        elif any(k in name_lower for k in ["stair", "ramp", "elevator", "escalator"]):
            return "stairs_ramps"
        elif any(k in name_lower for k in ["sink", "toilet", "fountain", "receptacle", "switch", "light"]):
            return "fixtures"
        else:
            return "other"

    for prim in stage.Traverse():
        total_prims += 1
        type_name = prim.GetTypeName()

        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            name = prim.GetName()
            mesh_names.append(name)
            cat = categorize_mesh(name)
            categories[cat].append(name)
        elif prim.IsA(UsdGeom.Xform):
            xform_count += 1
        else:
            other_prim_types[type_name] = other_prim_types.get(type_name, 0) + 1

    print(f"\n  Total prims: {total_prims}")
    print(f"  Mesh prims: {mesh_count}")
    print(f"  Xform prims: {xform_count}")
    if other_prim_types:
        print(f"  Other prim types:")
        for t, c in sorted(other_prim_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    {t}: {c}")

    # Mesh categories
    print(f"\n  Mesh categories:")
    for cat, names in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"    {cat}: {len(names)}")
        if names:
            for n in names[:5]:
                print(f"      - {n}")
            if len(names) > 5:
                print(f"      ... and {len(names) - 5} more")

    # Check external references
    print(f"\n  Checking external references...")
    missing_refs = []
    found_refs = []
    all_ref_paths = set()

    for prim in stage.Traverse():
        refs = prim.GetReferences()
        prim_refs = omni.usd.get_composed_references_from_prim(prim) if hasattr(omni.usd, 'get_composed_references_from_prim') else []

        # Also check via metadata
        stack = prim.GetPrimStack()
        for spec in stack:
            ref_list = spec.referenceList
            for ref in ref_list.prependedItems:
                if ref.assetPath:
                    all_ref_paths.add(ref.assetPath)
            for ref in ref_list.appendedItems:
                if ref.assetPath:
                    all_ref_paths.add(ref.assetPath)

    # Try to resolve each reference
    resolver = stage.GetPathResolverContext()
    for ref_path in sorted(all_ref_paths):
        # Check if the file exists relative to the USD location
        if ref_path.startswith("./") or ref_path.startswith("../") or not ref_path.startswith("/"):
            abs_path = os.path.join(os.path.dirname(usd_path), ref_path)
        else:
            abs_path = os.path.expanduser(ref_path)

        if os.path.exists(abs_path):
            found_refs.append(ref_path)
        else:
            missing_refs.append(ref_path)

    print(f"  Total external references: {len(all_ref_paths)}")
    print(f"  Found/resolved: {len(found_refs)}")
    print(f"  Missing: {len(missing_refs)}")
    if missing_refs:
        print(f"  Sample missing refs:")
        for r in missing_refs[:20]:
            print(f"    - {r}")
        if len(missing_refs) > 20:
            print(f"    ... and {len(missing_refs) - 20} more")

    # Bounding box
    print(f"\n  Computing bounding box...")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        world_prim = stage.GetPseudoRoot()

    bbox = bbox_cache.ComputeWorldBound(world_prim)
    bbox_range = bbox.ComputeAlignedRange()
    bb_min = bbox_range.GetMin()
    bb_max = bbox_range.GetMax()

    print(f"  Min: ({bb_min[0]:.1f}, {bb_min[1]:.1f}, {bb_min[2]:.1f})")
    print(f"  Max: ({bb_max[0]:.1f}, {bb_max[1]:.1f}, {bb_max[2]:.1f})")
    print(f"  Size: ({bb_max[0]-bb_min[0]:.1f} x {bb_max[1]-bb_min[1]:.1f} x {bb_max[2]-bb_min[2]:.1f})")

    sys.stdout.flush()

    return {
        "filename": filename,
        "file_size_mb": round(os.path.getsize(usd_path) / 1e6, 1),
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "total_prims": total_prims,
        "mesh_count": mesh_count,
        "xform_count": xform_count,
        "categories": {k: len(v) for k, v in categories.items()},
        "category_samples": {k: v[:10] for k, v in categories.items()},
        "total_external_refs": len(all_ref_paths),
        "missing_refs": len(missing_refs),
        "found_refs": len(found_refs),
        "missing_ref_samples": missing_refs[:30],
        "bbox_min": [float(bb_min[0]), float(bb_min[1]), float(bb_min[2])],
        "bbox_max": [float(bb_max[0]), float(bb_max[1]), float(bb_max[2])],
    }


def main():
    print("=" * 60)
    print("USD File Comparison Analysis")
    print("=" * 60)
    sys.stdout.flush()

    results = []
    for usd_path in USD_FILES:
        if not os.path.exists(usd_path):
            print(f"\nWARNING: {usd_path} not found, skipping")
            continue
        result = analyze_usd(usd_path)
        if result:
            results.append(result)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "usd_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {out_path}")

    # Print summary comparison
    if len(results) == 2:
        a, b = results
        print(f"\n{'=' * 60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Metric':<30} {a['filename']:<25} {b['filename']:<25}")
        print(f"{'-'*80}")
        print(f"{'File size':<30} {a['file_size_mb']} MB{'':<20} {b['file_size_mb']} MB")
        print(f"{'Total prims':<30} {a['total_prims']:<25} {b['total_prims']:<25}")
        print(f"{'Mesh count':<30} {a['mesh_count']:<25} {b['mesh_count']:<25}")
        print(f"{'Walls':<30} {a['categories'].get('walls',0):<25} {b['categories'].get('walls',0):<25}")
        print(f"{'Doors':<30} {a['categories'].get('doors',0):<25} {b['categories'].get('doors',0):<25}")
        print(f"{'Floors/Ceilings':<30} {a['categories'].get('floors_ceilings',0):<25} {b['categories'].get('floors_ceilings',0):<25}")
        print(f"{'Windows':<30} {a['categories'].get('windows',0):<25} {b['categories'].get('windows',0):<25}")
        print(f"{'HVAC':<30} {a['categories'].get('hvac',0):<25} {b['categories'].get('hvac',0):<25}")
        print(f"{'Structural':<30} {a['categories'].get('structural',0):<25} {b['categories'].get('structural',0):<25}")
        print(f"{'Stairs/Ramps':<30} {a['categories'].get('stairs_ramps',0):<25} {b['categories'].get('stairs_ramps',0):<25}")
        print(f"{'Fixtures':<30} {a['categories'].get('fixtures',0):<25} {b['categories'].get('fixtures',0):<25}")
        print(f"{'Other':<30} {a['categories'].get('other',0):<25} {b['categories'].get('other',0):<25}")
        print(f"{'External refs (total)':<30} {a['total_external_refs']:<25} {b['total_external_refs']:<25}")
        print(f"{'External refs (missing)':<30} {a['missing_refs']:<25} {b['missing_refs']:<25}")
        print(f"{'Up axis':<30} {a['up_axis']:<25} {b['up_axis']:<25}")
        print(f"{'Units':<30} {a['meters_per_unit']:<25} {b['meters_per_unit']:<25}")

        # Recommendation
        print(f"\n{'=' * 60}")
        print(f"RECOMMENDATION")
        print(f"{'=' * 60}")
        better = None
        reasons = []
        if b['mesh_count'] > a['mesh_count']:
            reasons.append(f"more meshes ({b['mesh_count']} vs {a['mesh_count']})")
            better = b['filename']
        elif a['mesh_count'] > b['mesh_count']:
            reasons.append(f"more meshes ({a['mesh_count']} vs {b['mesh_count']})")
            better = a['filename']

        if b['missing_refs'] < a['missing_refs']:
            reasons.append(f"fewer missing refs ({b['missing_refs']} vs {a['missing_refs']})")
            better = b['filename']
        elif a['missing_refs'] < b['missing_refs']:
            reasons.append(f"fewer missing refs ({a['missing_refs']} vs {b['missing_refs']})")
            better = a['filename']

        walls_a = a['categories'].get('walls', 0)
        walls_b = b['categories'].get('walls', 0)
        if walls_b > walls_a:
            reasons.append(f"more wall meshes ({walls_b} vs {walls_a})")
        elif walls_a > walls_b:
            reasons.append(f"more wall meshes ({walls_a} vs {walls_b})")

        if better and reasons:
            print(f"  Use {better}: {', '.join(reasons)}")
        else:
            print(f"  No clear winner — review details above")

    sys.stdout.flush()
    simulation_app.close()


if __name__ == "__main__":
    main()
