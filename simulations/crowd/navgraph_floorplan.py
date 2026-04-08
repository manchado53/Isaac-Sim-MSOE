"""Extract floor 4 wall geometry from USD and overlay nav graph.
No rendering needed — pure USD geometry extraction + matplotlib.
Run on GPU node: ~/isaacsim/python.sh simulations/crowd/navgraph_floorplan.py
Output: videos/navgraph_floorplan.png
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from pxr import Usd, UsdGeom, Gf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

# ── Load nav graph ─────────────────────────────────────────────────────────────
with open(GRAPH_PATH) as f:
    gdata = json.load(f)
floor_z = gdata["z_height_cm"] / 100.0
nodes = {n["id"]: np.array([n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100]) for n in gdata["nodes"]}
edges = [(e["from"], e["to"]) for e in gdata["edges"]]
print(f"[INFO] Nav graph: {len(nodes)} nodes, {len(edges)} edges, floor_z={floor_z:.2f}m")

# ── Load USD geometry ──────────────────────────────────────────────────────────
print("[INFO] Opening USD stage...")
stage = Usd.Stage.Open(USD_PATH)
print("[INFO] Stage opened, extracting geometry...")

# Collect all mesh face polygons near floor 4 z height
# Floor 4 z range: floor_z ± 5m (walls span floor to ceiling)
Z_MIN = floor_z - 1.0
Z_MAX = floor_z + 6.0

wall_segments = []   # list of (x0,y0,x1,y1) projected segments
floor_polys   = []   # filled floor polygons

total_meshes = 0
kept_meshes  = 0

for prim in stage.Traverse():
    if prim.GetTypeName() != "Mesh":
        continue
    total_meshes += 1

    mesh = UsdGeom.Mesh(prim)
    pts_attr = mesh.GetPointsAttr()
    if not pts_attr:
        continue
    pts = pts_attr.Get()
    if pts is None or len(pts) == 0:
        continue

    # Get world transform
    xform = UsdGeom.Xformable(prim)
    try:
        xform_mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    except Exception:
        continue

    # Transform points to world space
    world_pts = []
    for p in pts:
        wp = xform_mat.Transform(p)
        world_pts.append((wp[0], wp[1], wp[2]))
    world_pts = np.array(world_pts)

    # Check if any point is in floor 4 z range
    zs = world_pts[:, 2]
    if zs.max() < Z_MIN or zs.min() > Z_MAX:
        continue

    kept_meshes += 1

    # Project to XY and compute bounding box edges as wall segments
    # Use faces if available, otherwise just bbox edges
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    if face_counts and face_indices:
        idx = 0
        for fc in face_counts:
            verts = [world_pts[face_indices[idx + i]] for i in range(fc)]
            idx += fc
            # Only include faces that are roughly vertical (walls) or horizontal (floor)
            zv = [v[2] for v in verts]
            z_span = max(zv) - min(zv)
            xy = [(v[0], v[1]) for v in verts]
            if z_span > 0.3:  # vertical face = wall
                # Add projected edges
                for i in range(len(xy)):
                    a, b = xy[i], xy[(i+1) % len(xy)]
                    wall_segments.append([a, b])
    else:
        # Just bbox
        xs, ys = world_pts[:, 0], world_pts[:, 1]
        bbox = [(xs.min(), ys.min()), (xs.max(), ys.min()),
                (xs.max(), ys.max()), (xs.min(), ys.max())]
        for i in range(4):
            wall_segments.append([bbox[i], bbox[(i+1)%4]])

print(f"[INFO] Meshes: {total_meshes} total, {kept_meshes} in floor 4 z range")
print(f"[INFO] Wall segments: {len(wall_segments)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 16))
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")

# Draw wall segments
if wall_segments:
    lc = LineCollection(wall_segments, colors="#aaaacc", linewidths=0.3, alpha=0.5)
    ax.add_collection(lc)

# Draw nav graph edges
for src, dst in edges:
    a, b = nodes[src], nodes[dst]
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#ff4444", lw=2.0, alpha=0.9, zorder=3)

# Draw nodes
for nid, pos in nodes.items():
    ax.plot(pos[0], pos[1], "o", color="#ffdd00", ms=7, zorder=4)
    ax.annotate(nid, pos, fontsize=6, color="#ffdd00", zorder=5,
                xytext=(3, 3), textcoords="offset points")

# Set view to nav graph bounds with margin
xs = [v[0] for v in nodes.values()]
ys = [v[1] for v in nodes.values()]
margin = 10
ax.set_xlim(min(xs)-margin, max(xs)+margin)
ax.set_ylim(min(ys)-margin, max(ys)+margin)
ax.set_aspect("equal")
ax.set_title(f"Floor 4 Nav Graph (z={floor_z:.1f}m) — red=edges, yellow=nodes, gray=walls",
             color="white", fontsize=12)

out = os.path.join(PROJECT_DIR, "videos/navgraph_floorplan.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[INFO] Saved {out}")
plt.close()

app.close()
