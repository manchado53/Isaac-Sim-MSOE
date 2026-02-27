#!/usr/bin/env python3
"""
Nova Carter — Onboard Camera for Anomaly Detection

Loads the Isaac Sim hospital environment and drives a Nova Carter AMR
along a corridor route. Two cameras are recorded:

  1. ONBOARD camera  — 1st-person, mounted on the robot front (~0.5m high),
                       facing forward. This is the feed used for anomaly
                       detection (open doors, obstructions, irregular states).

  2. FOLLOW camera   — 3rd-person external view for visual debugging.

Outputs (in output/nova_carter_onboard_cam/):
  - onboard.mp4   -> what the robot "sees"
  - follow.mp4    -> external view of the robot navigating
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from scipy.spatial.transform import Rotation as R

import carb
import omni.usd
from pxr import Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

# Add project root to path for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# ---------------------------------------------------------------------------
# World & Environment
# ---------------------------------------------------------------------------

my_world = World(stage_units_in_meters=1.0)

hospital_path = assets_root_path + "/Isaac/Environments/Hospital/hospital.usd"
print(f"Loading hospital from: {hospital_path}")
add_reference_to_stage(hospital_path, prim_path="/World/Hospital")

# Lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# ---------------------------------------------------------------------------
# Robot — Nova Carter AMR
# ---------------------------------------------------------------------------

carter_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
my_carter = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="my_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=carter_asset_path,
        position=np.array([0.0, 0.0, 0.0]),
    )
)

my_controller = DifferentialController(
    name="simple_control", wheel_radius=0.14, wheel_base=0.4
)

# ---------------------------------------------------------------------------
# Camera 1 — ONBOARD (1st person, front-mounted)
#
# Offset from robot center:
#   X = +0.36m  (forward, in front of the robot body)
#   Y =  0.0m   (centered left-right)
#   Z = +0.5m   (sensor height — typical for anomaly detection use)
#
# Orientation: matches the robot's heading (updated every frame)
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.36, 0.0, 0.5])

onboard_camera = Camera(
    prim_path="/World/OnboardCamera",
    position=np.array([0.36, 0.0, 0.5]),   # will be updated every frame
    frequency=20,
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)

# ---------------------------------------------------------------------------
# Camera 2 — FOLLOW (3rd person, for debugging)
# ---------------------------------------------------------------------------

FOLLOW_OFFSET = np.array([5.0, -5.0, 4.0])
follow_orient = rot_utils.euler_angles_to_quats(np.array([0, 30, 135]), degrees=True)

follow_camera = Camera(
    prim_path="/World/FollowCamera",
    position=np.array([5.0, -5.0, 4.0]),
    frequency=20,
    resolution=(640, 480),
    orientation=follow_orient,
)

# ---------------------------------------------------------------------------
# Reset & initialize cameras
# ---------------------------------------------------------------------------

my_world.reset()
onboard_camera.initialize()
follow_camera.initialize()

# ---------------------------------------------------------------------------
# Navigation sequence: (start_frame, end_frame, [linear_vel, angular_vel])
# ---------------------------------------------------------------------------

nav_sequence = [
    (0,    300,  [0.5,  0.0]),
    (300,  420,  [0.0,  np.pi / 6]),
    (420,  720,  [0.5,  0.0]),
    (720,  840,  [0.0,  np.pi / 6]),
    (840,  1140, [0.5,  0.0]),
    (1140, 1260, [0.0,  np.pi / 6]),
    (1260, 1560, [0.5,  0.0]),
    (1560, 1680, [0.0,  np.pi / 6]),
    (1680, 1980, [0.5,  0.0]),
    (1980, 2100, [0.0, -np.pi / 6]),
    (2100, 2400, [0.5,  0.0]),
]

def get_command(step):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0.0, 0.0]

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

max_frames = 2500
frame_count = 0

print(f"Simulating {max_frames} frames...")
print("Recording: onboard.mp4 (1st person) + follow.mp4 (3rd person)")

with VideoRecorder("nova_carter_onboard_cam_onboard", fps=15) as onboard_recorder, \
     VideoRecorder("nova_carter_onboard_cam_follow",  fps=15) as follow_recorder:

    while frame_count < max_frames:

        if my_world.is_playing():
            # Drive robot
            my_carter.apply_wheel_actions(
                my_controller.forward(command=get_command(frame_count))
            )

            # Get robot pose
            carter_pos, carter_orient = my_carter.get_world_pose()

            # Update ONBOARD camera — rotate offset by robot orientation
            # so the camera stays in front of the robot even when it turns.
            # carter_orient is [w, x, y, z], scipy expects [x, y, z, w]
            rot = R.from_quat([carter_orient[1], carter_orient[2],
                               carter_orient[3], carter_orient[0]])
            rotated_offset = rot.apply(ONBOARD_OFFSET)

            onboard_camera.set_world_pose(
                position=carter_pos + rotated_offset,
                orientation=carter_orient,
            )

            # Update FOLLOW camera
            follow_camera.set_world_pose(
                position=carter_pos + FOLLOW_OFFSET,
                orientation=follow_orient,
            )

        # Render step AFTER pose updates (critical for async pipeline)
        my_world.step(render=True)

        if my_world.is_playing() and frame_count % 2 == 0 and frame_count > 0:
            onboard_rgba = onboard_camera.get_rgba()
            follow_rgba  = follow_camera.get_rgba()

            if onboard_rgba is not None:
                onboard_recorder.capture_rgba(onboard_rgba)
            if follow_rgba is not None:
                follow_recorder.capture_rgba(follow_rgba)

        if frame_count % 500 == 0:
            print(f"Frame {frame_count}/{max_frames}")

        frame_count += 1

my_world.stop()
simulation_app.close()

print("Done! Videos saved to output/")
