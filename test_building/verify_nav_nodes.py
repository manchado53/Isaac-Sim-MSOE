#!/usr/bin/env python3
"""
Verify floor_4_nav_graph_manual.json by placing colored spheres at each node's
world coordinates in the USD building. Captures screenshots from multiple
camera angles so you can visually confirm nodes land in hallways/rooms.

Run with:
  sbatch submit_isaac_job.sh test_building/verify_nav_nodes.py
  OR (interactive):
  srun --partition=teaching --gres=gpu:1 --time=2:00:00 --pty bash
  ./run_isaac_interactive.sh test_building/verify_nav_nodes.py
"""

import os
import sys
import json
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1920, "height": 1080})

import omni
import omni.kit.app
from pxr import Usd, UsdGeom, Gf, Sdf

# Add project root for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_dir)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
VERIFY_DIR = os.path.join(OUTPUT_DIR, "nav_verification_screenshots")
USD_PATH = os.path.join(os.path.dirname(PROJECT_DIR),
                        "assets", "environments", "msoe_building", "MSOEBuildingTake2.usd")
NAV_GRAPH_PATH = os.path.join(OUTPUT_DIR,
                              "ToDo (Doing_Coords_For_Floor_4)",
                              "floor_4_nav_graph_manual.json")

os.makedirs(VERIFY_DIR, exist_ok=True)


def create_sphere(stage, path, position, radius, color):
    """Create a colored sphere at the given world position."""
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.GetRadiusAttr().Set(radius)

    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])
    return sphere


def create_edge_line(stage, path, pos_a, pos_b, color):
    """Create a thin cylinder between two points to visualize an edge."""
    a = np.array(pos_a)
    b = np.array(pos_b)
    mid = (a + b) / 2.0
    diff = b - a
    length = np.linalg.norm(diff)
    if length < 0.01:
        return

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.GetRadiusAttr().Set(5.0)
    cylinder.GetHeightAttr().Set(length)
    cylinder.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])

    direction = diff / length
    up = np.array([0.0, 1.0, 0.0])
    cross = np.cross(up, direction)
    cross_len = np.linalg.norm(cross)

    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))

    if cross_len > 1e-6:
        cross /= cross_len
        dot = np.dot(up, direction)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        rotation = Gf.Rotation(Gf.Vec3d(float(cross[0]), float(cross[1]), float(cross[2])), angle)
        quat = rotation.GetQuat()
        orient_op = xform.AddOrientOp()
        orient_op.Set(Gf.Quatf(float(quat.GetReal()),
                                float(quat.GetImaginary()[0]),
                                float(quat.GetImaginary()[1]),
                                float(quat.GetImaginary()[2])))


def set_camera_view(stage, position, target):
    """Move the default viewport camera to look at the scene."""
    # Use the default perspective camera
    cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if not cam_prim.IsValid():
        # Create one if it doesn't exist
        cam_prim = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp").GetPrim()

    camera = UsdGeom.Camera(cam_prim)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(10.0, 200000.0))
    camera.GetFocalLengthAttr().Set(18.0)

    eye = Gf.Vec3d(*position)
    center = Gf.Vec3d(*target)
    up_vec = Gf.Vec3d(0, 0, 1)

    forward = (center - eye)
    forward_len = forward.GetLength()
    if forward_len > 0:
        forward /= forward_len

    right = Gf.Cross(forward, up_vec)
    right_len = right.GetLength()
    if right_len > 0:
        right /= right_len

    cam_up = Gf.Cross(right, forward)

    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(cam_up[0], cam_up[1], cam_up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
    mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    transform_op = xform.AddTransformOp()
    transform_op.Set(mat)


def capture_viewport_screenshot(output_path):
    """Capture screenshot from the active viewport."""
    from omni.kit.viewport.utility import get_viewport_from_window_name, capture_viewport_to_file
    import asyncio

    viewport_api = get_viewport_from_window_name("Viewport")
    if viewport_api is None:
        print(f"  WARNING: No viewport found")
        return False

    capture_viewport_to_file(viewport_api, output_path)
    # Need to wait for async capture to complete
    for _ in range(10):
        simulation_app.update()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"  Saved: {output_path} ({os.path.getsize(output_path) // 1024}KB)")
        return True
    else:
        print(f"  WARNING: Screenshot may be empty: {output_path}")
        return False


def main():
    print("=" * 60)
    print("Nav Node Verification - Floor 4 (with screenshots)")
    print("=" * 60)
    sys.stdout.flush()

    # Load nav graph
    print(f"\nLoading nav graph: {NAV_GRAPH_PATH}")
    with open(NAV_GRAPH_PATH) as f:
        nav = json.load(f)

    nodes = nav["nodes"]
    edges = nav["edges"]
    z_height = nav["z_height_cm"]
    print(f"  {len(nodes)} nodes, {len(edges)} edges, Z={z_height} cm")
    sys.stdout.flush()

    # Load USD
    print(f"\nLoading USD: {USD_PATH}")
    if not os.path.exists(USD_PATH):
        print(f"ERROR: USD file not found at {USD_PATH}")
        simulation_app.close()
        sys.exit(1)

    omni.usd.get_context().open_stage(USD_PATH)
    # Let stage fully load
    for i in range(20):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Failed to open stage")
        simulation_app.close()
        sys.exit(1)

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  Stage loaded, units={'cm' if meters_per_unit == 0.01 else 'm'}")
    sys.stdout.flush()

    # Create root prim for verification geometry
    stage.DefinePrim(Sdf.Path("/NavVerification"), "Xform")
    stage.DefinePrim(Sdf.Path("/NavVerification/nodes"), "Xform")
    stage.DefinePrim(Sdf.Path("/NavVerification/edges"), "Xform")

    # Build node position lookup
    node_positions = {}
    wx_vals, wy_vals = [], []
    for node in nodes:
        nid = node["id"]
        wx, wy = node["world_xy_cm"]
        pos = (wx, wy, z_height + 30)
        node_positions[nid] = pos
        wx_vals.append(wx)
        wy_vals.append(wy)

    center_x = (min(wx_vals) + max(wx_vals)) / 2
    center_y = (min(wy_vals) + max(wy_vals)) / 2
    span_x = max(wx_vals) - min(wx_vals)
    span_y = max(wy_vals) - min(wy_vals)

    # Place spheres at each node
    print(f"\nPlacing {len(nodes)} node spheres...")
    sys.stdout.flush()
    for i, node in enumerate(nodes):
        nid = node["id"]
        pos = node_positions[nid]
        sphere_path = f"/NavVerification/nodes/{nid}"
        create_sphere(stage, sphere_path, pos, 30.0, (0.0, 1.0, 0.0))

    # Highlight node_1 (blue) and node_55 (red)
    create_sphere(stage, "/NavVerification/nodes/node_1_highlight",
                  node_positions["node_1"], 50.0, (0.0, 0.3, 1.0))
    create_sphere(stage, "/NavVerification/nodes/node_55_highlight",
                  node_positions["node_55"], 50.0, (1.0, 0.0, 0.0))

    # Place edge lines
    print(f"Placing {len(edges)} edge lines...")
    sys.stdout.flush()
    for i, edge in enumerate(edges):
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in node_positions and to_id in node_positions:
            edge_path = f"/NavVerification/edges/edge_{i}"
            create_edge_line(stage, edge_path, node_positions[from_id],
                           node_positions[to_id], (1.0, 0.8, 0.0))

    # Let the scene render for many frames (needed for headless rendering)
    print("\nWarming up renderer...")
    sys.stdout.flush()
    for i in range(60):
        simulation_app.update()

    # Camera views to capture
    camera_views = [
        ("overhead_full",
         (center_x, center_y, z_height + 8000),
         (center_x, center_y, z_height)),

        ("overhead_left_wing",
         (min(wx_vals) + span_x * 0.2, center_y, z_height + 4000),
         (min(wx_vals) + span_x * 0.2, center_y, z_height)),

        ("overhead_right_wing",
         (max(wx_vals) - span_x * 0.2, center_y, z_height + 4000),
         (max(wx_vals) - span_x * 0.2, center_y, z_height)),

        ("perspective_high_NW",
         (center_x - 4000, center_y + 4000, z_height + 5000),
         (center_x, center_y, z_height)),

        ("perspective_high_SE",
         (center_x + 4000, center_y - 4000, z_height + 5000),
         (center_x, center_y, z_height)),
    ]

    # Capture screenshots from each camera angle
    print(f"\nCapturing {len(camera_views)} screenshots...")
    sys.stdout.flush()
    for view_name, cam_pos, cam_target in camera_views:
        print(f"\n  View: {view_name}")
        sys.stdout.flush()

        # Move camera
        set_camera_view(stage, cam_pos, cam_target)

        # Render many frames to fill the pipeline
        for _ in range(40):
            simulation_app.update()

        # Capture
        output_file = os.path.join(VERIFY_DIR, f"floor4_verify_{view_name}.png")
        capture_viewport_screenshot(output_file)

    # Save verification USD
    verify_usd_path = os.path.join(OUTPUT_DIR, "floor_4_nav_verification.usda")
    stage.GetRootLayer().Export(verify_usd_path)
    print(f"\nSaved verification USD: {verify_usd_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Screenshots: {VERIFY_DIR}")
    print(f"  USD: {verify_usd_path}")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"  X: {min(wx_vals):.1f} to {max(wx_vals):.1f} cm")
    print(f"  Y: {min(wy_vals):.1f} to {max(wy_vals):.1f} cm")
    print(f"  Z: {z_height} cm")
    sys.stdout.flush()

    simulation_app.close()


if __name__ == "__main__":
    main()
