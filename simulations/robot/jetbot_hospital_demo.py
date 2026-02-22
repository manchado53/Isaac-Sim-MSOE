from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

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
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder

# Get assets
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Create world
my_world = World(stage_units_in_meters=1.0)

# Load hospital environment
hospital_path = assets_root_path + "/Isaac/Environments/Hospital/hospital.usd"
print(f"Loading hospital from: {hospital_path}")
add_reference_to_stage(hospital_path, prim_path="/World/Hospital")

# Add extra lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# Spawn Nova Carter (full-size AMR: 0.72m x 0.5m x 0.56m, 280mm wheels)
carter_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
my_carter = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="my_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=carter_asset_path,
        position=np.array([0, 0.0, 0.0]),
    )
)

# Differential controller (Nova Carter: 280mm wheel diameter, ~400mm wheelbase)
my_controller = DifferentialController(name="simple_control", wheel_radius=0.14, wheel_base=0.4)

# Third-person side-angle follow camera
# Key: set_world_pose BEFORE world.step(render=True) due to async render pipeline
CAMERA_OFFSET = np.array([5.0, -5.0, 4.0])
side_orient = rot_utils.euler_angles_to_quats(np.array([0, 30, 135]), degrees=True)

camera = Camera(
    prim_path="/World/follow_camera",
    position=np.array([5.0, -5.0, 4.0]),
    frequency=20,
    resolution=(640, 480),
    orientation=side_orient,
)

my_world.reset()
camera.initialize()

# Navigation sequence: [linear_velocity m/s, angular_velocity rad/s]
nav_sequence = [
    (0,    300,  [0.5, 0]),
    (300,  420,  [0.0, np.pi / 6]),
    (420,  720,  [0.5, 0]),
    (720,  840,  [0.0, np.pi / 6]),
    (840,  1140, [0.5, 0]),
    (1140, 1260, [0.0, np.pi / 6]),
    (1260, 1560, [0.5, 0]),
    (1560, 1680, [0.0, np.pi / 6]),
    (1680, 1980, [0.5, 0]),
    (1980, 2100, [0.0, -np.pi / 6]),
    (2100, 2400, [0.5, 0]),
]

def get_command(step):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0, 0]

max_frames = 2500
frame_count = 0

print(f"Simulating {max_frames} frames...")

with VideoRecorder("jetbot_hospital", fps=15) as recorder:
    while frame_count < max_frames:
        if my_world.is_playing():
            # Drive the robot
            my_carter.apply_wheel_actions(my_controller.forward(command=get_command(frame_count)))

            # Update camera position BEFORE render step
            carter_pos, _ = my_carter.get_world_pose()
            camera.set_world_pose(position=carter_pos + CAMERA_OFFSET, orientation=side_orient)

        my_world.step(render=True)

        if my_world.is_playing() and frame_count % 2 == 0 and frame_count > 0:
            recorder.capture_rgba(camera.get_rgba())

        if frame_count % 500 == 0:
            print(f"Frame {frame_count}/{max_frames}")

        frame_count += 1

my_world.stop()
simulation_app.close()
