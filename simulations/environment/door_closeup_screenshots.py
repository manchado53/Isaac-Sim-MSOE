#!/usr/bin/env python3
"""
Door Close-Up Screenshots

Positions a camera 5ft in front of each door near Camera G and takes a
screenshot to confirm whether the door geometry actually renders.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/door_closeup_screenshots.py \
        assets/environments/msoe_building/MSOEBuildingApril.usd
"""

import os, sys, json, math, time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera as IsaacCamera
from sim_utils.test_cameras import (
    look_at_orientation, add_interior_lighting, CAMERA_FREQUENCY, CAMERA_RESOLUTION,
)
from sim_utils.building_physics import setup_physics_scene


def save_screenshot(camera, output_path):
    rgba = camera.get_rgba()
    if rgba is None:
        return 0.0
    try:
        from PIL import Image
        img = Image.fromarray(rgba[:, :, :3])
        img.save(output_path)
    except ImportError:
        np.save(output_path.replace(".png", ".npy"), rgba)
    return float(np.mean(rgba[:, :, :3]))


def main():
    usd_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "MSOEBuildingApril.usd"
    )
    usd_path = os.path.abspath(usd_path)

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        os._exit(1)

    out_dir = os.path.join(_project_dir, "images", "diagnostics", "door_closeups")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[door_closeup] Loading: {usd_path}", flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Detect and load USD
    root_layer = Sdf.Layer.FindOrOpen(usd_path)
    has_world_building = root_layer and root_layer.GetPrimAtPath("/World/Building")
    if has_world_building:
        building_prim = stage.DefinePrim("/World/Building", "Xform")
        building_prim.GetReferences().AddReference(usd_path, primPath=Sdf.Path("/World/Building"))
    else:
        add_reference_to_stage(usd_path, prim_path="/World/Building")

    setup_physics_scene(stage)
    add_interior_lighting(stage)

    # Camera G target — doors are near here
    cam_g_target = np.array([-27.42, 83.72, 14.56])

    # Find all door prims under /Geometry/Doors/
    print(f"[door_closeup] Searching for doors...", flush=True)
    door_info = []
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "/Geometry/Doors/" not in path_str:
            continue
        name = prim.GetName()
        if not name.startswith("Doors_"):
            continue

        bbox = bbox_cache.ComputeWorldBound(prim)
        box_range = bbox.ComputeAlignedRange()
        bmin = box_range.GetMin()
        bmax = box_range.GetMax()
        cx = (bmin[0] + bmax[0]) / 2.0
        cy = (bmin[1] + bmax[1]) / 2.0
        cz = (bmin[2] + bmax[2]) / 2.0
        center = np.array([cx, cy, cz])
        dist = np.linalg.norm(center - cam_g_target)

        # Door dimensions
        dx = bmax[0] - bmin[0]
        dy = bmax[1] - bmin[1]
        dz = bmax[2] - bmin[2]

        # Count mesh children with actual points
        mesh_info = []
        for child in prim.GetAllChildren():
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                pts = mesh.GetPointsAttr().Get()
                n_pts = len(pts) if pts else 0
                mesh_info.append(f"{child.GetName()}({n_pts}pts)")

        if dist < 15.0:  # Within 15ft of Camera G target (limited for GPU memory)
            door_info.append({
                "name": name,
                "path": path_str,
                "center": center,
                "dist": dist,
                "size": (dx, dy, dz),
                "meshes": mesh_info,
            })
            print(f"  {name}: center=({cx:.1f},{cy:.1f},{cz:.1f}) "
                  f"size=({dx:.1f}x{dy:.1f}x{dz:.1f}) dist={dist:.1f}ft "
                  f"meshes={mesh_info}", flush=True)

    door_info.sort(key=lambda d: d["dist"])
    print(f"\n[door_closeup] Found {len(door_info)} doors within 25ft of Camera G", flush=True)

    if not door_info:
        print("[door_closeup] No doors found! Exiting.", flush=True)
        os._exit(1)

    # Create a camera for each door — positioned 5ft away, looking at door center
    cameras = []
    for i, d in enumerate(door_info):
        door_center = d["center"]
        # Camera offset: step back from door center toward Camera G
        cam_g_pos = np.array([-42.15, 68.44, 16.44])
        dir_from_cam = door_center[:2] - cam_g_pos[:2]
        dir_norm = dir_from_cam / np.linalg.norm(dir_from_cam)

        # Position camera 5ft from door, on Camera G side, at door mid-height
        cam_pos = np.array([
            door_center[0] - dir_norm[0] * 5.0,
            door_center[1] - dir_norm[1] * 5.0,
            door_center[2],  # same height as door center
        ])
        cam_target = door_center

        orient = look_at_orientation(cam_pos, cam_target)
        cam = IsaacCamera(
            prim_path=f"/World/DoorCloseup_{i}",
            position=cam_pos,
            frequency=CAMERA_FREQUENCY,
            resolution=CAMERA_RESOLUTION,
            orientation=orient,
        )
        cameras.append((cam, d))

    # Also add a bright point light near the doors
    from pxr import UsdLux
    for i, d in enumerate(door_info):
        light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"/World/DoorLight_{i}"))
        light.CreateIntensityAttr(30000)
        light.CreateRadiusAttr(0.5)
        c = d["center"]
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
            (float(c[0]), float(c[1]), float(c[2]) + 2.0)
        )

    world.reset()
    for cam, _ in cameras:
        cam.initialize()
        cam_prim = stage.GetPrimAtPath(cam.prim_path)
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(24.0)  # tighter lens for close-up

    # Warmup
    print("[door_closeup] Warming up...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # Capture
    print("[door_closeup] Capturing close-ups...", flush=True)
    results = []
    for cam, d in cameras:
        for _ in range(10):
            world.step(render=True)

        png_name = f"door_{d['name']}.png"
        png_path = os.path.join(out_dir, png_name)
        brightness = save_screenshot(cam, png_path)
        print(f"  {d['name']}: brightness={brightness:.1f} -> {png_path}", flush=True)
        results.append({
            "name": d["name"],
            "file": png_name,
            "brightness": round(brightness, 1),
            "dist_from_cam_g": round(d["dist"], 1),
            "size_ft": [round(s, 2) for s in d["size"]],
            "meshes": d["meshes"],
        })

    # Save results
    with open(os.path.join(out_dir, "closeup_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[door_closeup] Done. {len(results)} close-ups saved to {out_dir}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
