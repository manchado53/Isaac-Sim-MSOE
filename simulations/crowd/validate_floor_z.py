"""Validate floor Z height at nav graph node positions using physics raycasting.
Drops a ray straight down from each node's XY at nav_z+5m and reports where it hits.
Run: ~/isaacsim/python.sh simulations/crowd/validate_floor_z.py
"""
import os, sys, json, traceback
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_PATH    = os.path.join(PROJECT_DIR, "videos/floor_z_results.txt")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

def log(msg):
    with open(OUT_PATH, "a") as f:
        f.write(msg + "\n")
        f.flush()

log("[START] validate_floor_z.py")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
log("[OK] SimulationApp started")

import numpy as np
import omni.usd
from isaacsim.core.api import World

GRAPH_PATH = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")
USD_PATH   = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")

try:
    with open(GRAPH_PATH) as f:
        gdata = json.load(f)
    nav_z = gdata["z_height_cm"] / 100.0
    nodes = {n["id"]: np.array([n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100]) for n in gdata["nodes"]}
    log(f"[OK] Nav graph: z={nav_z:.3f}m, {len(nodes)} nodes")

    log("[INFO] Opening USD stage...")
    omni.usd.get_context().open_stage(USD_PATH)
    log("[OK] Stage opened")

    world = World(stage_units_in_meters=1.0)
    log("[OK] World created")

    world.reset()
    log("[OK] World reset")

    for i in range(10):
        world.step(render=False)
    log("[OK] 10 warmup steps done")

    from omni.physx import get_physx_scene_query_interface
    qi = get_physx_scene_query_interface()
    log("[OK] Got physx scene query interface")

    results = []
    for nid, nxy in nodes.items():
        origin    = (float(nxy[0]), float(nxy[1]), nav_z + 5.0)
        direction = (0.0, 0.0, -1.0)
        distance  = 20.0
        hit = qi.raycast_closest(origin, direction, distance)
        if hit["hit"]:
            hit_z = origin[2] + direction[2] * hit["distance"]
            results.append((nid, float(nxy[0]), float(nxy[1]), hit_z, hit_z - nav_z))
        else:
            results.append((nid, float(nxy[0]), float(nxy[1]), None, None))
    log(f"[OK] Raycasting done, {len(results)} results")

    log(f"\n{'Node':<12} {'X':>8} {'Y':>8} {'Hit Z':>8} {'Offset':>8}")
    log("-" * 50)
    hit_offsets = []
    for nid, x, y, hz, off in results:
        if hz is not None:
            hit_offsets.append(off)
            flag = " <<<" if abs(off) > 1.0 else ""
            log(f"{nid:<12} {x:>8.2f} {y:>8.2f} {hz:>8.3f} {off:>+8.3f}{flag}")
        else:
            log(f"{nid:<12} {x:>8.2f} {y:>8.2f} {'NO HIT':>8}")

    if hit_offsets:
        no_hits = sum(1 for _, _, _, hz, _ in results if hz is None)
        log(f"\nSummary:")
        log(f"  nav_z          = {nav_z:.3f}m")
        log(f"  median floor Z = {np.median([nav_z + o for _, _, _, hz, o in results if hz is not None]):.3f}m")
        log(f"  median offset  = {np.median(hit_offsets):+.3f}m")
        log(f"  mean offset    = {np.mean(hit_offsets):+.3f}m")
        log(f"  min/max offset = {min(hit_offsets):+.3f} / {max(hit_offsets):+.3f}m")
        log(f"  no-hit nodes   = {no_hits}/{len(nodes)}")

except Exception as e:
    log(f"[ERROR] {e}")
    log(traceback.format_exc())

log("[DONE]")
app.close()
