from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom
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

# Add lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# Hide ceiling prims so the overhead camera can see inside the building.
# Only visual geometry is hidden — UsdGeom.Imageable does not affect collision meshes,
# so the robot still navigates correctly.
# From USD BBox inspection: ceiling prim names contain "Ceiling"; roof exterior ~Z=6.3m.
print("Hiding ceiling prims...")
hidden_count = 0
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if "ceiling" in name or "roof" in name:
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"  Hidden: {prim.GetPath()}")
        hidden_count += 1
print(f"Hidden {hidden_count} ceiling prims.")

# Spawn Nova Carter
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

# Overhead camera: static bird's-eye view of the full hospital.
#
# Hospital USD BBox (from BBoxCache inspection):
#   X: [-49.6, 27.0]  →  center X ≈ -11.3,  width  ≈ 76.6 m
#   Y: [ -4.7, 37.1]  →  center Y ≈  16.2,  depth  ≈ 41.8 m
#
# At Z=40m, a ~60° FOV covers 2 * 40 * tan(30°) ≈ 46 m radius — enough for the full building.
# pitch=90° points straight down (confirmed by prior USD testing).
camera = Camera(
    prim_path="/World/overhead_camera",
    position=np.array([-11.3, 16.2, 40.0]),
    frequency=20,
    resolution=(1280, 720),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
    annotator_device="cpu",
)

my_world.reset()
simulation_app.update()
camera.initialize()

# Warmup — renderer needs 10 steps before get_rgba() returns valid data
print("Warming up renderer...")
for _ in range(10):
    my_world.step(render=True)

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

print(f"Simulating {max_frames} frames (1280x720 overhead, ceiling hidden)...")

with VideoRecorder("jetbot_hospital_overhead", fps=15) as recorder:
    while frame_count < max_frames:
        my_world.step(render=True)

        if my_world.is_playing():
            my_carter.apply_wheel_actions(my_controller.forward(command=get_command(frame_count)))

            if frame_count % 2 == 0:
                recorder.capture_rgba(camera.get_rgba())

        if frame_count % 500 == 0:
            print(f"Frame {frame_count}/{max_frames}")

        frame_count += 1

my_world.stop()
simulation_app.close()
