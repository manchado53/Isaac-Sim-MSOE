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
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import ParallelGripper
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
my_world.scene.add_default_ground_plane()

# Add lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# Add Franka robot
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.02, 0.02]),
    action_deltas=np.array([0.01, 0.01]),
)
my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    )
)

# Add cube to pick up
cube = my_world.scene.add(
    DynamicCuboid(
        name="cube",
        position=np.array([0.3, 0.3, 0.3]),
        prim_path="/World/Cube",
        scale=np.array([0.0515, 0.0515, 0.0515]),
        size=1.0,
        color=np.array([0, 0, 1]),
    )
)

# Side-angle camera: euler (0, 23, 135) — pitch 23° down, yaw 135° to face robot from side
camera = Camera(
    prim_path="/World/camera",
    position=np.array([2.0, -2.0, 1.5]),
    frequency=20,
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 23, 135]), degrees=True),
    annotator_device="cpu",  # stable CPU-side readback per NVIDIA example
)

my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
my_world.reset()
simulation_app.update()
camera.initialize()

# Warmup — renderer needs 10 steps before get_rgba() returns valid data
print("Warming up renderer...")
for _ in range(10):
    my_world.step(render=True)

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()

max_frames = 1000
frame_count = 0
task_completed = False

print(f"Simulating {max_frames} frames...")

with VideoRecorder("franka_pick_place", fps=15) as recorder:
    while frame_count < max_frames:
        my_world.step(render=True)

        if my_world.is_playing():
            actions = my_controller.forward(
                picking_position=cube.get_local_pose()[0],
                placing_position=np.array([-0.3, -0.3, 0.0515 / 2.0]),
                current_joint_positions=my_franka.get_joint_positions(),
                end_effector_offset=np.array([0, 0.005, 0]),
            )
            if my_controller.is_done() and not task_completed:
                print(f"Pick and place completed at frame {frame_count}")
                task_completed = True
            articulation_controller.apply_action(actions)

            if frame_count % 2 == 0:
                recorder.capture_rgba(camera.get_rgba())

        frame_count += 1

simulation_app.close()
