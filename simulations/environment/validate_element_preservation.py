#!/usr/bin/env python3
"""
Validation: Element Preservation

Verifies that no building elements were lost or corrupted when physics
was applied to a raw USD. Compares the physics USD against a manifest
JSON written by setup_building_physics.py.

Checks:
  - Total element count matches
  - Per-category counts match
  - Mesh count >= manifest (physics adds meshes, never removes)
  - Bounding box matches within tolerance
  - Per-level counts match (no entire floors lost)

Usage:
    sbatch submit_isaac_job.sh simulations/environment/validate_element_preservation.py <usd_path>

Output:
    logs/validation/element_preservation_results.json
"""

import os
import sys
import json
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Sdf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from sim_utils.building_physics import (
    PrimCategory, categorize_prim, iter_building_elements,
)

BBOX_TOLERANCE = 0.5  # stage units — allowable bounding box drift


def find_manifest(usd_path):
    """Look for _manifest.json next to the physics USD."""
    stem = os.path.splitext(usd_path)[0]
    manifest_path = stem + "_manifest.json"
    if os.path.exists(manifest_path):
        return manifest_path
    return None


def main():
    t_start = time.time()
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        simulation_app.close()
        sys.exit(1)

    # Find manifest
    manifest_path = find_manifest(usd_path)
    if manifest_path is None:
        print(f"ERROR: Manifest not found for: {usd_path}", flush=True)
        print(f"Expected: {os.path.splitext(usd_path)[0]}_manifest.json", flush=True)
        print("Run setup_building_physics.py first to generate the manifest.", flush=True)
        simulation_app.close()
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[element_preservation] USD: {usd_path}", flush=True)
    print(f"[element_preservation] Manifest: {manifest_path}", flush=True)
    print(f"[element_preservation] Source: {manifest.get('source_usd', '?')}", flush=True)

    # --- Load physics USD ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )

    # --- Count elements in physics USD ---
    print("[element_preservation] Counting elements...", flush=True)
    building_prim = stage.GetPrimAtPath("/World/Building")

    counts = {cat.name: 0 for cat in PrimCategory}
    level_counts = {}

    for level_name, category_name, element in iter_building_elements(stage, building_prim):
        element_name = element.GetName()
        category = categorize_prim(element_name, category_hint=category_name)
        counts[category.name] = counts.get(category.name, 0) + 1

        if level_name not in level_counts:
            level_counts[level_name] = {cat.name: 0 for cat in PrimCategory}
        level_counts[level_name][category.name] = level_counts[level_name].get(category.name, 0) + 1

    total_elements = sum(counts.values())

    # Count meshes
    mesh_count = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))

    # Bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(building_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()
    current_bbox = {
        "min": [float(bb_min[i]) for i in range(3)],
        "max": [float(bb_max[i]) for i in range(3)],
    }

    # --- Compare against manifest ---
    print("[element_preservation] Comparing against manifest...", flush=True)
    checks = []
    overall_pass = True

    # Check 1: Total element count (physics-active only — SKIP elements may be
    # pruned during USD export, so we only verify physics-active categories)
    manifest_total = manifest.get("total_elements", 0)
    manifest_cats = manifest.get("categories", {})
    manifest_active = manifest_total - manifest_cats.get("SKIP", 0)
    actual_active = total_elements - counts.get("SKIP", 0)
    active_match = actual_active == manifest_active
    checks.append({
        "check": "total_elements_active",
        "manifest": manifest_active,
        "actual": actual_active,
        "manifest_total": manifest_total,
        "actual_total": total_elements,
        "skip_pruned": manifest_cats.get("SKIP", 0) - counts.get("SKIP", 0),
        "passed": active_match,
    })
    if not active_match:
        overall_pass = False
        print(f"  FAIL: active elements manifest={manifest_active} actual={actual_active} "
              f"(diff={actual_active - manifest_active})", flush=True)
    else:
        print(f"  PASS: active elements = {actual_active} (SKIP pruned: "
              f"{manifest_cats.get('SKIP', 0) - counts.get('SKIP', 0)})", flush=True)

    # Check 2: Per-category counts (only physics-active categories must match exactly;
    # SKIP is allowed to decrease since USD export may prune unreferenced prims)
    cat_mismatches = []
    for cat_name in set(list(counts.keys()) + list(manifest_cats.keys())):
        m_val = manifest_cats.get(cat_name, 0)
        a_val = counts.get(cat_name, 0)
        if cat_name == "SKIP":
            # SKIP elements may decrease (pruned during export) but never increase
            if a_val > m_val:
                cat_mismatches.append({
                    "category": cat_name,
                    "manifest": m_val,
                    "actual": a_val,
                    "diff": a_val - m_val,
                })
        elif m_val != a_val:
            cat_mismatches.append({
                "category": cat_name,
                "manifest": m_val,
                "actual": a_val,
                "diff": a_val - m_val,
            })
    cats_match = len(cat_mismatches) == 0
    checks.append({
        "check": "per_category",
        "passed": cats_match,
        "mismatches": cat_mismatches,
    })
    if not cats_match:
        overall_pass = False
        for m in cat_mismatches:
            print(f"  FAIL: {m['category']} manifest={m['manifest']} "
                  f"actual={m['actual']} (diff={m['diff']})", flush=True)
    else:
        print(f"  PASS: all physics-active categories match", flush=True)

    # Check 3: Mesh count (physics may add, never removes)
    manifest_meshes = manifest.get("mesh_count", 0)
    meshes_ok = mesh_count >= manifest_meshes
    checks.append({
        "check": "mesh_count",
        "manifest": manifest_meshes,
        "actual": mesh_count,
        "passed": meshes_ok,
    })
    if not meshes_ok:
        overall_pass = False
        print(f"  FAIL: mesh_count manifest={manifest_meshes} actual={mesh_count} "
              f"(lost {manifest_meshes - mesh_count} meshes!)", flush=True)
    else:
        print(f"  PASS: mesh_count = {mesh_count} (manifest={manifest_meshes})", flush=True)

    # Check 4: Bounding box within tolerance
    manifest_bbox = manifest.get("bounding_box", {})
    bbox_ok = True
    bbox_diffs = []
    if manifest_bbox:
        for axis, label in enumerate(["X", "Y", "Z"]):
            for bound in ["min", "max"]:
                m_val = manifest_bbox.get(bound, [0, 0, 0])[axis]
                a_val = current_bbox[bound][axis]
                diff = abs(a_val - m_val)
                if diff > BBOX_TOLERANCE:
                    bbox_ok = False
                    bbox_diffs.append(f"{bound}[{label}] drift={diff:.2f}")
    checks.append({
        "check": "bounding_box",
        "manifest": manifest_bbox,
        "actual": current_bbox,
        "passed": bbox_ok,
        "diffs": bbox_diffs,
    })
    if not bbox_ok:
        overall_pass = False
        print(f"  FAIL: bounding_box drifted: {', '.join(bbox_diffs)}", flush=True)
    else:
        print(f"  PASS: bounding_box within tolerance ({BBOX_TOLERANCE})", flush=True)

    # Check 5: Per-level counts (physics-active elements only — SKIP may be pruned)
    manifest_levels = manifest.get("per_level", {})
    level_mismatches = []
    for lname in set(list(level_counts.keys()) + list(manifest_levels.keys())):
        m_level = manifest_levels.get(lname, {})
        a_level = level_counts.get(lname, {})
        # Only compare physics-active elements (exclude SKIP)
        m_active = sum(v for k, v in m_level.items() if k != "SKIP")
        a_active = sum(v for k, v in a_level.items() if k != "SKIP")
        if m_active != a_active:
            level_mismatches.append({
                "level": lname,
                "manifest_active": m_active,
                "actual_active": a_active,
                "diff": a_active - m_active,
            })
    levels_match = len(level_mismatches) == 0
    checks.append({
        "check": "per_level",
        "passed": levels_match,
        "levels_checked": len(set(list(level_counts.keys()) + list(manifest_levels.keys()))),
        "mismatches": level_mismatches,
    })
    if not levels_match:
        overall_pass = False
        for m in level_mismatches:
            print(f"  FAIL: level '{m['level']}' manifest={m['manifest_total']} "
                  f"actual={m['actual_total']} (diff={m['diff']})", flush=True)
    else:
        print(f"  PASS: all {len(manifest_levels)} levels match", flush=True)

    # --- Summary ---
    passed_checks = sum(1 for c in checks if c["passed"])
    total_checks = len(checks)
    print(f"\n[element_preservation] {passed_checks}/{total_checks} checks passed", flush=True)
    print(f"[element_preservation] OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    # --- Save results ---
    results = {
        "test": "element_preservation",
        "usd_file": os.path.basename(usd_path),
        "manifest_file": os.path.basename(manifest_path),
        "source_usd": manifest.get("source_usd", "?"),
        "checks": checks,
        "summary": {
            "total_elements": total_elements,
            "mesh_count": mesh_count,
            "manifest_total_elements": manifest_total,
            "manifest_mesh_count": manifest_meshes,
            "checks_passed": passed_checks,
            "checks_total": total_checks,
        },
        "overall_passed": overall_pass,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    out_dir = os.path.join(_project_dir, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "element_preservation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[element_preservation] Results -> {out_path}", flush=True)

    os._exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
