#!/usr/bin/env python3
"""
Build a USD stage with the Innovation Center + N random cracked panels and
export it to a standalone USD file. Designed to be loaded by the full Isaac
Sim Kit app (via `isaac-sim.streaming.sh <path>`) so the user gets the
standard GUI — stage tree, menus, property panels, viewport tools — with
the cracked scene already prepared.

Unlike cracked_glass_floor4_stream.py, this script does NOT start a
livestream or enter a keep-alive loop. It builds → exports → exits.

Usage:
    ./python.sh SyntheticDataGen/cracked_glass/build_cracked_scene_usd.py
    ./python.sh SyntheticDataGen/cracked_glass/build_cracked_scene_usd.py \\
        --output /tmp/cracked_floor4.usd --n-panels 40 --seed 42
"""

import argparse
import datetime
import json
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)
sys.path.insert(0, _script_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--output", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "InnovationCenter_cracked.usd"),
        help="Path where the cracked stage is written (default under "
             "assets/environments/InnovationCenter/).")
    p.add_argument("--n-panels", type=int, default=0,
                   help="How many panels to crack. 0 (default) = crack ALL "
                        "panels on the selected floor.")
    p.add_argument("--floor", type=int, default=4,
                   help="Which floor to sample glass from. 0 = all floors "
                        "(whole-building spread). Default: 4.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--usd", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--variants", default="edge_v1,v4,tempered_v5",
                   help="Comma-separated crack variants to sample from")
    p.add_argument("--positions-json", default=os.path.join(
        _project_dir, "logs", "cracked_glass_floor4_positions.json"))
    return p.parse_args()


args = parse_args()

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_info import get_floor  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene, _find_mesh_descendant,
)
from glass_anomaly import (  # noqa: E402
    create_cracked_glass_materials, enumerate_glass_prims,
)
from sim_utils.test_cameras import add_camera_headlight  # noqa: E402

TEXTURE_DIR = os.path.join(_project_dir, "assets", "materials", "cracked_glass")


def bind_visual_material_deep(stage, prim, material_path):
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    mesh = _find_mesh_descendant(prim, max_depth=4)
    for target in (prim, mesh):
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try:
            api.UnbindDirectBinding()
        except Exception:
            pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def panel_world_center(prim, bbox_cache):
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return (
        float((bb.GetMin()[0] + bb.GetMax()[0]) / 2.0),
        float((bb.GetMin()[1] + bb.GetMax()[1]) / 2.0),
        float((bb.GetMin()[2] + bb.GetMax()[2]) / 2.0),
    )


random.seed(args.seed)
print(f"[build] USD source: {args.usd}", flush=True)

world = World()
stage = omni.usd.get_context().get_stage()

building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(
    args.usd, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)
# No stage-wide lighting — a SphereLight parented under /World/StreamCam
# at the bottom of this script acts as a "headlight" that follows the
# active viewport camera, illuminating whatever the user is looking at
# regardless of where they navigate inside the building.

variant_list = tuple(v.strip() for v in args.variants.split(",") if v.strip())
variant_mat_paths = {}
for v in variant_list:
    paths = create_cracked_glass_materials(
        stage, texture_dir=TEXTURE_DIR,
        root_path=f"/World/Looks/Crack_{v}", variants=(v,))
    if paths:
        variant_mat_paths[v] = paths[0]
print(f"[build] Variants authored: {list(variant_mat_paths.keys())}", flush=True)

all_glass = enumerate_glass_prims(stage)
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def center_z(p):
    bb = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
    return float((bb.GetMin()[2] + bb.GetMax()[2]) / 2.0)


if args.floor > 0:
    f = get_floor("InnovationCenter", args.floor)
    deck_z = f["floor_deck_z"]
    ceiling_z = deck_z + 4.0
    candidate_glass = [p for p in all_glass if deck_z <= center_z(p) < ceiling_z]
    print(f"[build] Glass on floor {args.floor}: {len(candidate_glass)}",
          flush=True)
else:
    candidate_glass = all_glass
    print(f"[build] Glass across ALL floors: {len(candidate_glass)}",
          flush=True)

n = len(candidate_glass) if args.n_panels <= 0 else min(args.n_panels, len(candidate_glass))
selected = random.sample(candidate_glass, n)
variant_names = list(variant_mat_paths.keys())
positions = []
variant_counts = {v: 0 for v in variant_names}
for i, panel in enumerate(selected):
    v = random.choice(variant_names)
    bind_visual_material_deep(stage, panel, variant_mat_paths[v])
    variant_counts[v] += 1
    cx, cy, cz = panel_world_center(panel, bbox_cache)
    positions.append({
        "idx": i, "prim": str(panel.GetPath()),
        "name": panel.GetName(), "variant": v,
        "world_xyz": [cx, cy, cz],
    })
print(f"[build] Variant distribution: {variant_counts}", flush=True)

os.makedirs(os.path.dirname(args.positions_json), exist_ok=True)
with open(args.positions_json, "w") as f:
    json.dump({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_usd": args.usd,
        "output_usd": args.output,
        "floor": args.floor if args.floor > 0 else "all",
        "variants": variant_names,
        "variant_counts": variant_counts,
        "seed": args.seed,
        "panels": positions,
    }, f, indent=2)
print(f"[build] Position sidecar: {args.positions_json}", flush=True)

# Camera-tracked headlight (follows whichever camera the viewport uses).
# Replaces the stage-wide fixed lights — works wherever the user flies.
add_camera_headlight(stage)

# Export the flattened stage so the standalone Kit app can open it directly
# with all material bindings already authored.
os.makedirs(os.path.dirname(args.output), exist_ok=True)
stage.Export(args.output)
print(f"[build] Wrote {args.output}", flush=True)
print(f"[build] Launch with: ~/isaacsim/isaac-sim.streaming.sh "
      f"{args.output} <streaming flags>", flush=True)

simulation_app.close()
