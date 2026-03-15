from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom, Gf
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

# TestBuilding is local — no S3 download needed for the environment.
# USD is authored in centimeters (metersPerUnit=0.01); we apply a scale
# transform on the prim after loading to convert to meters.
BUILDING_PATH = os.path.abspath(
    os.path.join(_script_dir, "../assets/environments/msoe_building/TestBuilding.usd")
)

# Nova Carter is not cached locally — still sourced from S3.
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Create world (simulation units in meters)
my_world = World(stage_units_in_meters=1.0)

# Add a default ground plane so the robot has a surface to drive on.
# The TestBuilding has no CollisionAPI on any mesh (Revit export default),
# so without this the robot would fall through the floor.
my_world.scene.add_default_ground_plane()

# Load TestBuilding
print(f"Loading building from: {BUILDING_PATH}")
add_reference_to_stage(BUILDING_PATH, prim_path="/World/Building")

# Apply a transform to the building prim to:
#   1. Scale 0.01 — converts centimeters (USD authored units) to meters
#   2. Translate +3.61 m in Z — lifts the floor (originally at −3.61 m after
#      scaling) up to Z = 0, flush with the ground plane.
#
# USD 4×4 matrix convention (row-major, translation in last row):
#   P_world = P_local * M
#   → scale 0.01 applied to all axes, then Z shifted by +3.61 m
#
# For a point at z = −361 cm (raw USD):  0.01 × (−361) + 3.61 = 0.0 m  ✓
stage = omni.usd.get_context().get_stage()
building_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Building"))
building_xform.AddTransformOp().Set(
    Gf.Matrix4d(
        0.01, 0.0,  0.0,  0.0,
        0.0,  0.01, 0.0,  0.0,
        0.0,  0.0,  0.01, 0.0,
        0.0,  0.0,  3.61, 1.0,
    )
)

# Add lighting
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# Hide ONLY the exterior roof panels so the overhead camera can see into
# the building. Interior ceiling tiles (Compound_Ceiling_*) are kept visible
# — they are the main horizontal surfaces seen from above once the roof is gone.
#
# From USD inspection, exterior roof prim pattern:
#   Basic_Roof_Insulation_on_Metal_Deck___EPDM_*  (7 instances × 2 = 14 prims)
print("Hiding exterior roof prims...")
hidden_count = 0
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if "basic_roof" in name:
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_count += 1
print(f"Hidden {hidden_count} exterior roof prims.")

# Spawn Nova Carter
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

# Differential controller (Nova Carter: 280 mm wheel diameter, ~400 mm wheelbase)
my_controller = DifferentialController(name="simple_control", wheel_radius=0.14, wheel_base=0.4)

# Overhead camera: static bird's-eye view of the full building.
#
# TestBuilding USD BBox (from BBoxCache, scaled to meters):
#   X: [−21.74, 24.87]  →  center X ≈  1.57,  width  ≈ 46.6 m
#   Y: [ −9.72, 24.52]  →  center Y ≈  7.40,  depth  ≈ 34.2 m
#   Z: [  0.00, 17.03]  →  height  ≈ 17.0 m  (after floor-to-Z=0 translation)
#
# At Z=60m, a 60° FOV covers 2 × 60 × tan(30°) ≈ 69 m radius — enough for
# the full 46.6 m × 34.2 m footprint.  pitch=90° points straight down.
camera = Camera(
    prim_path="/World/overhead_camera",
    position=np.array([1.57, 7.40, 60.0]),
    frequency=20,
    resolution=(1280, 720),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
    annotator_device="cpu",
)

my_world.reset()
simulation_app.update()
camera.initialize()

# Warmup — renderer needs several steps before get_rgba() returns valid data
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

with VideoRecorder("testbuilding_overhead", fps=15) as recorder:
    while frame_count < max_frames:
        # Only run a full RTX render pass on frames we actually capture.
        # Physics runs every step regardless of render=True/False.
        do_render = (frame_count % 2 == 0)
        my_world.step(render=do_render)

        if my_world.is_playing():
            my_carter.apply_wheel_actions(
                my_controller.forward(command=get_command(frame_count))
            )
            if do_render:
                recorder.capture_rgba(camera.get_rgba())

        if frame_count % 500 == 0:
            print(f"Frame {frame_count}/{max_frames}")

        frame_count += 1

my_world.stop()
simulation_app.close()
