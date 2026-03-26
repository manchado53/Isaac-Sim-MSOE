#!/usr/bin/env python3
"""Save a single screenshot of the node visualization to verify rendering."""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from isaacsim.sensors.camera import Camera
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILDING_USD = os.path.join(SCRIPT_DIR, "..", "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")
NAV_GRAPH = os.path.join(SCRIPT_DIR, "test_nodes", "Final_Data", "floor_4_nav_graph_world.json")

with open(NAV_GRAPH) as f:
    graph = json.load(f)

nodes = graph["nodes"]
z_height = graph.get("z_height_cm", 2825.0)

world = World()
stage = world.stage

# Load building
building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(BUILDING_USD)

# Spawn BIGGER cubes (100cm) so they're visible from above
for i, node in enumerate(nodes):
    wx, wy = node["world_xy_cm"]
    wz = z_height + 80.0
    if i % 3 == 0:
        color = np.array([1.0, 0.0, 0.0])
    elif i % 3 == 1:
        color = np.array([0.0, 1.0, 0.0])
    else:
        color = np.array([0.0, 0.3, 1.0])
    VisualCuboid(
        prim_path=f"/World/NavNodes/{node['id']}",
        name=node["id"],
        position=np.array([wx, wy, wz]),
        size=100.0,
        color=color,
    )

# Compute center
all_x = [n["world_xy_cm"][0] for n in nodes]
all_y = [n["world_xy_cm"][1] for n in nodes]
center_x = sum(all_x) / len(all_x)
center_y = sum(all_y) / len(all_y)

# Overhead camera - not too high
cam_height = z_height + 6000.0

cam = Camera(
    prim_path="/World/ScreenshotCam",
    position=np.array([center_x, center_y, cam_height]),
    resolution=(1920, 1080),
    orientation=np.array([0.5, -0.5, 0.5, -0.5]),
)

world.reset()
cam.initialize()

# Long warmup
for _ in range(100):
    world.step(render=True)

# Save screenshot
rgba = cam.get_rgba()
if rgba is not None:
    from PIL import Image
    img = Image.fromarray(rgba[:, :, :3])
    out_path = os.path.join(SCRIPT_DIR, "floor4_nodes_screenshot.png")
    img.save(out_path)
    sys.stderr.write(f"Screenshot saved: {out_path}\n")
    sys.stderr.write(f"Frame shape: {rgba.shape}, min: {rgba.min()}, max: {rgba.max()}\n")
else:
    sys.stderr.write("ERROR: Camera returned None\n")

simulation_app.close()
