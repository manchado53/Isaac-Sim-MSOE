#!/usr/bin/env python3
"""
Isaac Sim - JetBot pushes a block off a raised platform.

A JetBot drives forward on a raised platform and pushes an orange block
off the edge, where it falls to the ground below.
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import carb
import omni.usd
from pxr import Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

# Add project root to path for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder


def main():
    print("=" * 60)
    print("Isaac Sim - JetBot Block Push")
    print("=" * 60)

    # Get assets
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    print("\n* World created with ground plane")

    # Add lighting
    stage = omni.usd.get_context().get_stage()
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    distant_light.CreateIntensityAttr(500)
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(200)

    # Platform: 0.8m x 0.5m x 0.5m, top surface at Z=0.50
    platform = world.scene.add(
        FixedCuboid(
            prim_path="/World/Platform",
            name="platform",
            position=np.array([0.0, 0.0, 0.25]),
            size=1.0,
            scale=np.array([0.8, 0.5, 0.5]),
            color=np.array([0.5, 0.5, 0.5]),
        )
    )
    print("* Platform placed (0.8x0.5x0.5m, top at Z=0.50)")

    # Block: 10cm cube, 0.5 kg, orange, near platform edge
    block = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Block",
            name="block",
            position=np.array([0.25, 0.0, 0.55]),
            size=0.1,
            color=np.array([1.0, 0.5, 0.0]),
            mass=0.1,
        )
    )
    print("* Block placed (10cm, 0.5kg, orange) at x=0.25, z=0.55")

    # JetBot
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=jetbot_asset_path,
            position=np.array([-0.25, 0.0, 0.52]),
        )
    )
    print(f"* JetBot loaded from: {jetbot_asset_path}")

    # Differential controller (JetBot: 0.03m wheel radius, 0.1125m wheel base)
    controller = DifferentialController(name="jetbot_control", wheel_radius=0.03, wheel_base=0.1125)

    # Static camera: side-angle view capturing the push and fall
    camera_orient = rot_utils.euler_angles_to_quats(np.array([0, 25, 135]), degrees=True)
    camera = Camera(
        prim_path="/World/camera",
        position=np.array([3.0, -2.0, 2.0]),
        frequency=20,
        resolution=(640, 480),
        orientation=camera_orient,
    )

    world.reset()
    camera.initialize()
    print("* World reset, camera initialized\n")

    # Navigation sequence
    SETTLE_END = 60       # Frames 0-59: let physics settle (2s)
    DRIVE_END = 300       # Frames 60-299: drive forward at 0.3 m/s (8s)
    TOTAL_FRAMES = 450    # Frames 300-449: stop and watch block fall (5s)

    print(f"Simulating {TOTAL_FRAMES} frames (~15s)...")
    print(f"  Frames 0-{SETTLE_END-1}: Settling")
    print(f"  Frames {SETTLE_END}-{DRIVE_END-1}: Driving forward")
    print(f"  Frames {DRIVE_END}-{TOTAL_FRAMES-1}: Watching block fall\n")

    with VideoRecorder("jetbot_block_push", fps=15) as recorder:
        for frame in range(TOTAL_FRAMES):
            if world.is_playing():
                # Determine wheel command
                if SETTLE_END <= frame < DRIVE_END:
                    command = [0.3, 0.0]  # 0.3 m/s forward, no turning
                else:
                    command = [0.0, 0.0]  # stopped

                jetbot.apply_wheel_actions(controller.forward(command=command))

            world.step(render=True)

            # Capture every 2nd frame for 15 FPS video
            if world.is_playing() and frame % 2 == 0 and frame > 0:
                recorder.capture_rgba(camera.get_rgba())

            # Log positions periodically
            if frame % 60 == 0:
                block_pos, _ = block.get_world_pose()
                jetbot_pos, _ = jetbot.get_world_pose()
                print(
                    f"  Frame {frame:4d} | "
                    f"JetBot x={jetbot_pos[0]:+.3f} z={jetbot_pos[2]:+.3f} | "
                    f"Block x={block_pos[0]:+.3f} z={block_pos[2]:+.3f}"
                )

    # Final positions
    block_pos, _ = block.get_world_pose()
    jetbot_pos, _ = jetbot.get_world_pose()
    print(f"\n{'=' * 60}")
    print(f"Final positions:")
    print(f"  JetBot: x={jetbot_pos[0]:+.3f} y={jetbot_pos[1]:+.3f} z={jetbot_pos[2]:+.3f}")
    print(f"  Block:  x={block_pos[0]:+.3f} y={block_pos[1]:+.3f} z={block_pos[2]:+.3f}")
    print(f"* Simulation completed!")
    print(f"* Video saved to: videos/jetbot_block_push.mp4")
    print(f"{'=' * 60}")

    world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
