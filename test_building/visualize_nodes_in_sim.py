#!/usr/bin/env python3
"""
Spawn colored cubes at nav graph node positions and record video.
Matches the working bouncing_balls.py approach exactly.

Run: sbatch submit_isaac_job.sh test_building/visualize_nodes_in_sim.py
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim_utils.video_capture import VideoRecorder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILDING_USD = os.path.join(SCRIPT_DIR, "..", "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")
NAV_GRAPH = os.path.join(SCRIPT_DIR, "test_nodes", "Final_Data", "floor_4_nav_graph_world.json")

with open(NAV_GRAPH) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph.get("edges", [])
z_height = graph.get("z_height_cm", 2825.0)

print("=" * 60)
print("Node Visualization in Isaac Sim")
print("=" * 60)

world = World()
world.scene.add_default_ground_plane()
print("World created")

# Add building
stage = world.stage
building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(BUILDING_USD)
print("Building loaded")

# Spawn cubes at node positions
CUBE_SIZE = 100.0
Z_OFFSET = 80.0

print(f"Spawning {len(nodes)} cubes...")
for i, node in enumerate(nodes):
    wx, wy = node["world_xy_cm"]
    wz = z_height + Z_OFFSET
    if i % 3 == 0:
        color = np.array([1.0, 0.0, 0.0])
    elif i % 3 == 1:
        color = np.array([0.0, 1.0, 0.0])
    else:
        color = np.array([0.0, 0.3, 1.0])
    world.scene.add(
        VisualCuboid(
            prim_path=f"/World/NavNodes/{node['id']}",
            name=node["id"],
            position=np.array([wx, wy, wz]),
            size=CUBE_SIZE,
            color=color,
        )
    )

print(f"All {len(nodes)} cubes spawned")

# Edge markers
from pxr import Gf, UsdGeom
node_lookup = {n["id"]: n for n in nodes}
for i, edge in enumerate(edges):
    fn = node_lookup.get(edge["from"])
    tn = node_lookup.get(edge["to"])
    if not fn or not tn:
        continue
    x1, y1 = fn["world_xy_cm"]
    x2, y2 = tn["world_xy_cm"]
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    mz = z_height + Z_OFFSET
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx)
    edge_prim = stage.DefinePrim(f"/World/NavEdges/edge_{i}", "Cube")
    xformable = UsdGeom.Xformable(edge_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(mx, my, mz))
    xformable.AddRotateZOp().Set(float(np.degrees(angle)))
    xformable.AddScaleOp().Set(Gf.Vec3d(length / 2.0, 5.0, 5.0))
    UsdGeom.Gprim(edge_prim).CreateDisplayColorAttr([(0.9, 0.9, 0.0)])

print(f"Spawned {len(edges)} edge markers")

world.reset()
print("World reset")

# Record video using default viewport (same as bouncing_balls.py)
num_frames = 300
print(f"Recording {num_frames} frames...")

with VideoRecorder("floor4_nodes") as recorder:
    for i in range(num_frames):
        world.step(render=True)
        recorder.capture()
        if i % 60 == 0:
            print(f"  Frame {i}/{num_frames}")

print("Video recording complete")

# Export USD file for teammates
EXPORT_PATH = os.path.join(SCRIPT_DIR, "..", "assets", "environments", "msoe_building", "floor4_with_nav_nodes.usd")
stage.Export(EXPORT_PATH)
print(f"Exported USD to: {EXPORT_PATH}")

print("=" * 60)
print("Done!")
print("=" * 60)

simulation_app.close()
