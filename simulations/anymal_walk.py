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
from isaacsim.core.utils.prims import define_prim
from isaacsim.robot.policy.examples.robots import AnymalFlatTerrainPolicy
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
physics_dt = 1 / 200.0
render_dt = 1 / 60.0
my_world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

# Add lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# Spawn warehouse environment
prim = define_prim("/World/Warehouse", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
prim.GetReferences().AddReference(asset_path)

# Spawn ANYmal robot
anymal = AnymalFlatTerrainPolicy(
    prim_path="/World/Anymal",
    name="Anymal",
    position=np.array([0, 0, 0.7]),
)

# Third-person follow camera
CAMERA_OFFSET = np.array([-3.0, -3.0, 2.0])
side_orient = rot_utils.euler_angles_to_quats(np.array([0, 20, 45]), degrees=True)

camera = Camera(
    prim_path="/World/follow_camera",
    position=np.array([0, 0, 0.7]) + CAMERA_OFFSET,
    frequency=20,
    resolution=(640, 480),
    orientation=side_orient,
)

# Initialize
my_world.reset()
simulation_app.update()
camera.initialize()

# Walk forward command: [forward_vel, lateral_vel, yaw_rate]
walk_command = np.array([1.0, 0.0, 0.0])

max_frames = 2000
frame_count = 0
initialized = False

print(f"Simulating {max_frames} frames...")

with VideoRecorder("anymal_walk", fps=15) as recorder:
    while frame_count < max_frames:
        if my_world.is_playing():
            if not initialized:
                anymal.initialize()
                initialized = True
            else:
                anymal.forward(physics_dt, walk_command)

                # Update camera position BEFORE render step
                robot_pos, _ = anymal.robot.get_world_pose()
                camera.set_world_pose(position=robot_pos + CAMERA_OFFSET, orientation=side_orient)

        my_world.step(render=True)

        if my_world.is_playing() and frame_count % 3 == 0 and frame_count > 10:
            recorder.capture_rgba(camera.get_rgba())

        if frame_count % 500 == 0:
            print(f"Frame {frame_count}/{max_frames}")

        frame_count += 1

my_world.stop()
simulation_app.close()
