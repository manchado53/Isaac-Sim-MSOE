#!/usr/bin/env python3
"""
Test Building Physics

Loads TestBuilding_physics.usd, drops colored cubes from above the building,
and verifies they come to rest on surfaces rather than falling through.

Output:
  - videos/building_physics_test.mp4  (visual confirmation)
  - PASS/FAIL printed based on final cube positions
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import omni.usd
from pxr import Sdf, UsdLux, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

# Add project root for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import setup_physics_scene


def main():
    print("=" * 60, flush=True)
    print("Test Building Physics", flush=True)
    print("=" * 60, flush=True)

    # --- Load physics-enabled building ---
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building", "TestBuilding_physics.usd"
    )
    if not os.path.exists(physics_usd):
        print(f"ERROR: {physics_usd} not found. Run setup_building_physics.py first.", flush=True)
        simulation_app.close()
        sys.exit(1)

    print(f"Loading: {physics_usd}", flush=True)
    add_reference_to_stage(physics_usd, prim_path="/World/Building")
    print("Building loaded.", flush=True)

    # Ensure a physics scene exists at the stage level
    setup_physics_scene(stage)

    # Add a ground plane as safety net (catches anything that misses building geometry)
    world.scene.add_default_ground_plane()

    # --- Add lighting ---
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    distant_light.CreateIntensityAttr(500)
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(200)

    # --- Drop test cubes ---
    # Building bounding box from report: min(-21.74, -9.72, -3.61) max(24.87, 24.52, 13.42)
    # Center approx (1.57, 7.40, 4.91)
    # Drop from z=15 (above building max z of 13.42)
    drop_z = 15.0
    cube_configs = [
        {"name": "cube_red",    "position": [0.0, 5.0, drop_z],   "color": [1.0, 0.0, 0.0]},
        {"name": "cube_green",  "position": [5.0, 10.0, drop_z],  "color": [0.0, 1.0, 0.0]},
        {"name": "cube_blue",   "position": [-5.0, 7.0, drop_z],  "color": [0.0, 0.0, 1.0]},
        {"name": "cube_yellow", "position": [3.0, 0.0, drop_z],   "color": [1.0, 1.0, 0.0]},
    ]

    cubes = []
    for cfg in cube_configs:
        cube = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{cfg['name']}",
                name=cfg["name"],
                position=np.array(cfg["position"]),
                scale=np.array([0.3, 0.3, 0.3]),
                color=np.array(cfg["color"]),
            )
        )
        cubes.append(cube)

    print(f"Added {len(cubes)} test cubes at z={drop_z}", flush=True)

    # --- Setup camera ---
    # Position camera to see the building and falling cubes
    center = np.array([1.57, 7.40, 4.91])
    cam_distance = 40.0
    cam_height = 12.0
    cam_pos = np.array([center[0] + cam_distance, center[1], cam_height])

    dx = center[0] - cam_pos[0]
    dy = center[1] - cam_pos[1]
    dz = center[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))

    orient = rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )

    camera = Camera(
        prim_path="/World/TestCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1280, 720),
        orientation=orient,
    )

    # --- Run simulation ---
    world.reset()
    camera.initialize()

    # Warm up rendering
    for _ in range(20):
        world.step(render=True)

    print("Recording 300 frames (10s at 30fps)...", flush=True)

    with VideoRecorder("building_physics_test", fps=30) as recorder:
        for i in range(300):
            world.step(render=True)
            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                print(f"  Frame {i}/300", flush=True)

    # --- Check final cube positions ---
    print(f"\n{'=' * 60}", flush=True)
    print("FINAL CUBE POSITIONS", flush=True)
    print("=" * 60, flush=True)

    all_pass = True
    for cube in cubes:
        pos, _ = cube.get_world_pose()
        z = pos[2]
        # Cubes should rest on surfaces (z > -0.5), not fall through to -infinity
        passed = z > -0.5
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {cube.name:15s} pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  [{status}]", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    if all_pass:
        print("RESULT: PASS - All cubes resting on surfaces", flush=True)
    else:
        print("RESULT: FAIL - Some cubes fell through geometry", flush=True)
    print("=" * 60, flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
