"""
Hospital Navigation — Isaac Sim Simulation

Loads the NVIDIA hospital environment, builds the navigation graph,
runs Dijkstra's to find the shortest path, and visualizes the route
using debug draw lines.

The output video shows:
    - The hospital from an overhead camera angle
    - Gray lines for every walkable connection (the full graph)
    - Colored dots at every node (blue=rooms, yellow=hallways, etc.)
    - Bright green line showing the shortest path
    - Green dot at start, red dot at destination

Usage:
    cd ~/isaac-sim-msoe
    sbatch submit_isaac_job.sh simulations/building_navigation.py
"""

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
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

# Add project root to path for navigation and sim_utils imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _project_dir)

from navigation import build_hospital_graph, draw_graph, draw_path
from sim_utils.video_capture import VideoRecorder

# =========================================================================
# Scene Setup
# =========================================================================

# Get NVIDIA cloud assets
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)

# Load hospital environment from NVIDIA cloud
hospital_path = assets_root_path + "/Isaac/Environments/Hospital/hospital.usd"
print(f"Loading hospital from: {hospital_path}")
add_reference_to_stage(hospital_path, prim_path="/World/Hospital")

# Add lighting
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(500)
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(200)

# =========================================================================
# Navigation Graph
# =========================================================================

# Build the hospital graph
graph = build_hospital_graph()
print(f"Built navigation graph: {graph}")

# Find shortest path — change these to test different routes
start_node = "entrance_main"
end_node = "room_103"

path, distance = graph.dijkstra(start_node, end_node)

if path is not None:
    start_name = graph.get_node(start_node).name
    end_name = graph.get_node(end_node).name
    print(f"Shortest path from '{start_name}' to '{end_name}':")
    print(f"  Route: {' -> '.join(path)}")
    print(f"  Distance: {distance:.2f} meters")
else:
    print(f"No path found from '{start_node}' to '{end_node}'")

# =========================================================================
# Camera Setup
# =========================================================================

# Overhead camera looking down at the hospital floor plan
# TODO: Adjust position once real coordinates are known
camera = Camera(
    prim_path="/World/overhead_camera",
    position=np.array([10.0, 0.0, 30.0]),  # High above, centered on hospital
    frequency=20,
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),  # Looking straight down
)

# =========================================================================
# Initialize and Draw
# =========================================================================

my_world.reset()
simulation_app.update()
camera.initialize()

# Draw the full graph (gray edges + colored nodes)
draw_graph(graph)

# Highlight the shortest path in green on top
if path is not None:
    draw_path(graph, path)

# =========================================================================
# Simulation Loop — Capture Video
# =========================================================================

max_frames = 300  # ~10 seconds (static scene, doesn't need many frames)
frame_count = 0

print(f"Recording {max_frames} frames...")

with VideoRecorder("hospital_navigation", fps=15) as recorder:
    while frame_count < max_frames:
        my_world.step(render=True)

        if my_world.is_playing() and frame_count % 2 == 0 and frame_count > 5:
            recorder.capture_rgba(camera.get_rgba())

        frame_count += 1

# =========================================================================
# Cleanup
# =========================================================================

my_world.stop()
simulation_app.close()
