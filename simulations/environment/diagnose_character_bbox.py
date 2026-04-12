#!/usr/bin/env python3
"""
Diagnose character bounding box + actual floor height (via raycast).

Loads the Innovation Center building AND F_Business_02 at the usual XY,
then raycasts down from above the character position to find the true
walkable floor height. Compares with character bbox min Z to compute
the Z offset needed so feet plant on the floor.
"""
import os
import sys
import json

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.utils import prims as prims_utils
from isaacsim.storage.native import get_assets_root_path
from sim_utils.building_physics import setup_physics_scene
from sim_utils.test_cameras import TEST_CAMERAS

CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"
FLOOR_Z = 14.56


def main():
    assets_root = get_assets_root_path()
    character_usd = assets_root + CHARACTER_SUBPATH

    usd_path = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingV4_physics.usd"
    )

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)

    # Freeze bodies so doors don't move
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    # Camera G placement target
    CAM = TEST_CAMERAS["G"]
    T = 0.55
    px = CAM["pos"][0] + (CAM["target"][0] - CAM["pos"][0]) * T
    py = CAM["pos"][1] + (CAM["target"][1] - CAM["pos"][1]) * T

    # Load character
    person = prims_utils.create_prim(
        "/World/Person_01",
        "Xform",
        usd_path=character_usd,
    )
    person.GetAttribute("xformOp:translate").Set(Gf.Vec3d(px, py, FLOOR_Z))

    world.reset()
    for _ in range(10):
        world.step(render=False)

    # Character bbox
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bbox_cache.ComputeWorldBound(person).ComputeAlignedRange()
    bb_min = rng.GetMin()
    bb_max = rng.GetMax()

    # Also full building bbox
    bldg_rng = bbox_cache.ComputeWorldBound(building_prim).ComputeAlignedRange()
    bldg_min = bldg_rng.GetMin()
    bldg_max = bldg_rng.GetMax()

    # Raycast down from various origins to find the ground floor
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    # Cast from just above expected ground floor — this finds the IC floor directly
    # even if Engineering Building floors stack above it.
    floor_raycast_z = None
    raycast_body = None
    for origin_z in [FLOOR_Z + 1.0, FLOOR_Z + 3.0, FLOOR_Z + 5.0, FLOOR_Z + 50.0]:
        h = query.raycast_closest((px, py, origin_z), (0.0, 0.0, -1.0), 100.0)
        if h and h.get("hit"):
            if floor_raycast_z is None:
                floor_raycast_z = float(h["position"][2])
                raycast_body = h.get("rigidBody", "")

    # Try a few sample points with origin just above FLOOR_Z
    sample_hits = []
    for (sx, sy) in [(px, py), (px - 2, py), (px + 2, py), (px, py - 2), (px, py + 2),
                     (CAM["target"][0], CAM["target"][1])]:
        h = query.raycast_closest((sx, sy, FLOOR_Z + 1.5), (0.0, 0.0, -1.0), 20.0)
        if h and h.get("hit"):
            sample_hits.append({
                "xy": [sx, sy],
                "z": float(h["position"][2]),
                "body": h.get("rigidBody", ""),
            })
        else:
            sample_hits.append({"xy": [sx, sy], "z": None, "body": None})

    # Also: cast from high up but allow multiple hits via slab heights
    # Use raycast_all if available to get all floors below
    all_hits = []
    try:
        def hit_callback(hit):
            all_hits.append({
                "z": float(hit.position[2]),
                "distance": float(hit.distance),
                "body": str(hit.rigid_body) if hasattr(hit, "rigid_body") else "",
            })
            return True  # continue
        query.raycast_all((px, py, 40.0), (0.0, 0.0, -1.0), 50.0, hit_callback)
    except Exception as e:
        all_hits.append({"error": str(e)})

    result = {
        "placement_xy": [px, py],
        "floor_z_constant": FLOOR_Z,
        "character_bbox_min": [float(bb_min[0]), float(bb_min[1]), float(bb_min[2])],
        "character_bbox_max": [float(bb_max[0]), float(bb_max[1]), float(bb_max[2])],
        "character_height": float(bb_max[2] - bb_min[2]),
        "building_bbox_min": [float(bldg_min[0]), float(bldg_min[1]), float(bldg_min[2])],
        "building_bbox_max": [float(bldg_max[0]), float(bldg_max[1]), float(bldg_max[2])],
        "raycast_at_person": {"hit_z": floor_raycast_z, "body": raycast_body},
        "raycast_samples": sample_hits,
        "raycast_all_hits": all_hits,
    }

    # Derived recommendation
    if floor_raycast_z is not None:
        # We want character feet (bbox_min_Z) == floor_raycast_z
        # Currently bbox_min_Z = FLOOR_Z + delta, where delta = bbox_min_Z - FLOOR_Z
        delta = float(bb_min[2]) - FLOOR_Z
        # To land feet on floor: set translate_Z = floor_raycast_z - delta
        # i.e. offset = floor_raycast_z - FLOOR_Z - delta
        offset = (floor_raycast_z - FLOOR_Z) - delta
        result["recommended_character_z_offset_feet"] = offset
        result["recommended_translate_z"] = FLOOR_Z + offset

    out_path = os.path.join(_project_dir, "logs", "diagnose_character_bbox.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[diagnose] wrote {out_path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
