"""Validate nav graph edges via physics raycasting.

For each edge A→B, casts a horizontal ray from A to B at chest height.
If it hits geometry before reaching B, the edge crosses a wall → bad edge.

Outputs:
  videos/edge_validation.txt   — full results
  videos/navgraph_validated.png — floorplan with good edges (green) / bad (red)

Run: ~/isaacsim/python.sh simulations/crowd/validate_edges.py
"""
import os, sys, json, traceback
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

PROJECT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_TXT      = os.path.join(PROJECT_DIR, "videos/edge_validation.txt")
OUT_PNG      = os.path.join(PROJECT_DIR, "videos/navgraph_validated.png")
os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)

def log(msg):
    with open(OUT_TXT, "a") as f:
        f.write(msg + "\n")
        f.flush()

log("[START] validate_edges.py")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
log("[OK] SimulationApp started")

import numpy as np
import omni.usd
from isaacsim.core.api import World

GRAPH_PATH = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")
USD_PATH   = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
FLOOR_Z    = 31.039   # from raycasting validation
CHEST_Z    = FLOOR_Z + 1.0   # ray height
TOLERANCE  = 0.4     # metres — clearance at node endpoints before declaring a hit

try:
    with open(GRAPH_PATH) as f:
        gdata = json.load(f)

    nodes = {n["id"]: np.array([n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100])
             for n in gdata["nodes"]}
    edges = [(e["from"], e["to"], e["distance_cm"]/100)
             for e in gdata["edges"]]
    log(f"[OK] Graph: {len(nodes)} nodes, {len(edges)} edges")

    omni.usd.get_context().open_stage(USD_PATH)
    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(10):
        world.step(render=False)
    log("[OK] Physics warmed up")

    from omni.physx import get_physx_scene_query_interface
    qi = get_physx_scene_query_interface()

    bad_edges  = []
    good_edges = []

    for src, dst, dist_m in edges:
        if src not in nodes or dst not in nodes:
            log(f"[SKIP] {src}→{dst}: node missing")
            continue

        a = nodes[src]
        b = nodes[dst]
        diff = b - a
        length = float(np.linalg.norm(diff))
        if length < 0.01:
            continue
        direction = diff / length

        origin = (float(a[0]), float(a[1]), CHEST_Z)
        ray_dir = (float(direction[0]), float(direction[1]), 0.0)
        cast_dist = length + 0.5   # cast slightly past the target node

        hit = qi.raycast_closest(origin, ray_dir, cast_dist)

        if hit["hit"] and hit["distance"] < (length - TOLERANCE):
            bad_edges.append((src, dst, length, hit["distance"]))
            log(f"[BAD]  {src:10s}→{dst:10s}  edge={length:.2f}m  wall_hit={hit['distance']:.2f}m")
        else:
            good_edges.append((src, dst, length))
            log(f"[OK]   {src:10s}→{dst:10s}  edge={length:.2f}m  clear")

    log(f"\nSummary: {len(bad_edges)} bad edges, {len(good_edges)} good edges")
    log("\nBad edges to remove:")
    for src, dst, length, hit_d in bad_edges:
        log(f"  {src} → {dst}  ({length:.2f}m, wall at {hit_d:.2f}m)")

except Exception as e:
    log(f"[ERROR] {e}")
    log(traceback.format_exc())

# ── Overlay plot ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()

    # Collect wall geometry (same approach as navgraph_floorplan.py)
    wall_segments = []
    FLOOR_Z_MIN = FLOOR_Z - 0.5
    FLOOR_Z_MAX = FLOOR_Z + 4.0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr()
        if not pts_attr:
            continue
        pts = np.array(pts_attr.Get())
        if pts is None or len(pts) < 3:
            continue
        zs = pts[:, 2]
        if not (np.any(zs > FLOOR_Z_MIN) and np.any(zs < FLOOR_Z_MAX)):
            continue
        mask = (zs > FLOOR_Z_MIN) & (zs < FLOOR_Z_MAX)
        if mask.sum() < 2:
            continue
        visible = pts[mask]
        if len(visible) >= 2:
            wall_segments.append(visible[:, :2])

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    for seg in wall_segments:
        ax.scatter(seg[:, 0], seg[:, 1], s=0.3, c="#666666", linewidths=0)

    # Draw edges
    for src, dst, _ in good_edges:
        if src in nodes and dst in nodes:
            a, b = nodes[src], nodes[dst]
            ax.plot([a[0], b[0]], [a[1], b[1]], c="#00cc44", lw=1.2, alpha=0.8)

    for src, dst, _, _ in bad_edges:
        if src in nodes and dst in nodes:
            a, b = nodes[src], nodes[dst]
            ax.plot([a[0], b[0]], [a[1], b[1]], c="#ff2222", lw=2.0, alpha=1.0)
            mid = (a + b) / 2
            ax.text(mid[0], mid[1], f"{src}→{dst}", fontsize=5, color="yellow",
                    ha="center", va="center")

    # Draw nodes
    for nid, pos in nodes.items():
        ax.plot(pos[0], pos[1], "o", ms=4, color="#ffffff")
        ax.text(pos[0], pos[1] + 0.3, nid, fontsize=4, color="#aaaaaa", ha="center")

    ax.set_aspect("equal")
    ax.set_title(f"Edge Validation — {len(bad_edges)} bad (red), {len(good_edges)} good (green)",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"[OK] Plot saved: {OUT_PNG}")

except Exception as e:
    log(f"[WARN] Plot failed: {e}")

log("[DONE]")
app.close()
