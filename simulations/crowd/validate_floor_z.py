"""Validate floor Z height at nav graph node positions using physics raycasting.
Drops a ray straight down from each node's XY at nav_z+5m and reports where it hits.
Run: ~/isaacsim/python.sh simulations/crowd/validate_floor_z.py
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
import omni.physx
from pxr import UsdPhysics, PhysxSchema

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

with open(GRAPH_PATH) as f:
    gdata = json.load(f)
nav_z = gdata["z_height_cm"] / 100.0
nodes = {n["id"]: np.array([n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100]) for n in gdata["nodes"]}
print(f"[INFO] Nav graph z = {nav_z:.3f}m, {len(nodes)} nodes")

omni.usd.get_context().open_stage(USD_PATH)
world = World(stage_units_in_meters=1.0)
world.reset()

# Warm up physics
for _ in range(10):
    world.step(render=False)

from omni.physx import get_physx_scene_query_interface

results = []
for nid, nxy in nodes.items():
    origin    = (float(nxy[0]), float(nxy[1]), nav_z + 5.0)  # start 5m above nav_z
    direction = (0.0, 0.0, -1.0)
    distance  = 20.0  # cast 20m down

    hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance)
    if hit["hit"]:
        hit_z = origin[2] + direction[2] * hit["distance"]
        results.append((nid, nxy[0], nxy[1], hit_z, hit_z - nav_z))
    else:
        results.append((nid, nxy[0], nxy[1], None, None))

print(f"\n{'Node':<12} {'X':>8} {'Y':>8} {'Hit Z':>8} {'Offset':>8}")
print("-" * 50)
hit_offsets = []
for nid, x, y, hz, off in results:
    if hz is not None:
        hit_offsets.append(off)
        flag = " <<<" if abs(off) > 1.0 else ""
        print(f"{nid:<12} {x:>8.2f} {y:>8.2f} {hz:>8.3f} {off:>+8.3f}{flag}")
    else:
        print(f"{nid:<12} {x:>8.2f} {y:>8.2f} {'NO HIT':>8} {'':>8}")

if hit_offsets:
    print(f"\nSummary:")
    print(f"  nav_z          = {nav_z:.3f}m")
    print(f"  median floor Z = {np.median([nav_z + o for _, _, _, hz, o in results if hz is not None]):.3f}m")
    print(f"  median offset  = {np.median(hit_offsets):+.3f}m")
    print(f"  mean offset    = {np.mean(hit_offsets):+.3f}m")
    print(f"  min/max offset = {min(hit_offsets):+.3f} / {max(hit_offsets):+.3f}m")
    no_hits = sum(1 for _, _, _, hz, _ in results if hz is None)
    print(f"  no-hit nodes   = {no_hits}/{len(nodes)}")

# Write results to file (stdout is swallowed by Isaac Sim in headless mode)
out_path = os.path.join(PROJECT_DIR, "videos/floor_z_results.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(f"nav_z = {nav_z:.3f}m\n\n")
    f.write(f"{'Node':<12} {'X':>8} {'Y':>8} {'Hit Z':>8} {'Offset':>8}\n")
    f.write("-" * 50 + "\n")
    for nid, x, y, hz, off in results:
        if hz is not None:
            flag = " <<<" if abs(off) > 1.0 else ""
            f.write(f"{nid:<12} {x:>8.2f} {y:>8.2f} {hz:>8.3f} {off:>+8.3f}{flag}\n")
        else:
            f.write(f"{nid:<12} {x:>8.2f} {y:>8.2f} {'NO HIT':>8} {'':>8}\n")
    if hit_offsets:
        f.write(f"\nSummary:\n")
        f.write(f"  nav_z          = {nav_z:.3f}m\n")
        f.write(f"  median floor Z = {np.median([nav_z + o for _, _, _, hz, o in results if hz is not None]):.3f}m\n")
        f.write(f"  median offset  = {np.median(hit_offsets):+.3f}m\n")
        f.write(f"  mean offset    = {np.mean(hit_offsets):+.3f}m\n")
        f.write(f"  min/max offset = {min(hit_offsets):+.3f} / {max(hit_offsets):+.3f}m\n")
        no_hits = sum(1 for _, _, _, hz, _ in results if hz is None)
        f.write(f"  no-hit nodes   = {no_hits}/{len(nodes)}\n")
print(f"[INFO] Results written to {out_path}", flush=True)

app.close()
