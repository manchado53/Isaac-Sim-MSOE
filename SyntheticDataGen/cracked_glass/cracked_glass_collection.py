#!/usr/bin/env python3
"""
cracked_glass_collection.py -- per-panel dataset collector.

Renders the full 6-axis grid sweep for ONE panel from selected_panels.json.
Writes JPEGs + JSONL rows into a shared output dir so the 13-panel
orchestrator script can stack all runs into one dataset.

Per-panel frame count:
  6 lightings x 4 states x 3 arc_angles x 3 uv_tilings x 3 arc_dists x 3 heights
  = 1,944 frames / panel
  = 25,272 total across 13 panels

We render one panel per process rather than all 13 at once because Hydra's
render graph gets unhappy with hundreds of cameras; limiting to 27 cameras
per run (3 arc_angles x 3 dists x 3 heights) matches Tanner's pattern in
`origin/tanner_vlm:door_static_collection.py` where each SLURM job handled
one door at a time.

Usage (single panel -- called by run_cracked_glass_collection.sh):
  ~/isaacsim/python.sh SyntheticDataGen/cracked_glass/cracked_glass_collection.py \\
      --panel-idx 2 \\
      --output datasets/cracked_glass/20261204_1500

Orchestrator (all 13 panels):
  bash SyntheticDataGen/cracked_glass/run_cracked_glass_collection.sh
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)
sys.path.insert(0, _script_dir)


def _csv_floats(s):   return [float(x) for x in s.split(",") if x.strip()]
def _csv_ints(s):     return [int(x)   for x in s.split(",") if x.strip()]
def _csv_strs(s):     return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--selected-panels", default=os.path.join(
        _script_dir, "selected_panels.json"))
    p.add_argument("--usd", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--output", required=True,
                   help="Shared dataset output dir. Created if missing; "
                        "appended to if it already exists (so sequential "
                        "per-panel runs can share the same dataset).")
    p.add_argument("--panel-idx", type=int, required=True,
                   help="panel_idx from selected_panels.json. Only this "
                        "panel is rendered in this process.")
    p.add_argument("--variants", type=_csv_strs,
                   default=["edge_v1", "v4", "tempered_v5"])
    p.add_argument("--lightings", type=_csv_strs, default=[
        "bright_fluorescent", "daylight_flood", "warm_evening",
        "dim_night", "overcast_flat", "high_contrast"])
    p.add_argument("--arc-angles",    type=_csv_floats, default=[-0.4, 0.0, 0.4])
    p.add_argument("--arc-distances", type=_csv_floats, default=[1.5, 2.0, 2.5])
    p.add_argument("--uv-tilings",    type=_csv_ints,   default=[1, 2, 4])
    p.add_argument("--heights",       type=_csv_floats, default=[-0.5, 0.0, 0.5])
    p.add_argument("--resolution",    default="1280x720",
                   help="Render resolution WxH. 1280x720 = ~150 KB/JPEG "
                        "x 25k frames ~= 4 GB. 1920x1080 ~= ~9 GB.")
    p.add_argument("--jpeg-quality",  type=int, default=92)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps",  type=int, default=20)
    p.add_argument("--render-steps",  type=int, default=5,
                   help="rep.orchestrator.step() calls per frame.")
    return p.parse_args()


args = parse_args()


# ---------------------------------------------------------------------------
# Isaac Sim boot -- must come AFTER argparse and BEFORE pxr/omni imports
# ---------------------------------------------------------------------------

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene, _find_mesh_descendant,
)
from glass_anomaly import create_cracked_glass_material  # noqa: E402
from lighting_presets import (  # noqa: E402
    setup_lighting_rig, apply_lighting_preset, set_fill_position,
    LIGHTING_PRESETS,
)


TEXTURE_DIR = os.path.join(_project_dir, "assets", "materials", "cracked_glass")
W, H = (int(x) for x in args.resolution.lower().split("x"))
RESOLUTION = (W, H)

UV_TILING_TO_HITS = {1: 0, 2: 4, 4: 16}
STATES = ["intact"] + list(args.variants)   # 4 total when defaults
CRACKED_STATES = set(args.variants)

random.seed(args.seed)
rng = random.Random(args.seed + args.panel_idx)   # decorrelate per-panel jitter


# ---------------------------------------------------------------------------
# Load panel + validate
# ---------------------------------------------------------------------------

with open(args.selected_panels) as f:
    selected = json.load(f)
panel = next((p for p in selected["panels"]
              if p["panel_idx"] == args.panel_idx), None)
if panel is None:
    raise SystemExit(
        f"panel_idx={args.panel_idx} not found in {args.selected_panels}")
split = panel.get("split", "train")
print(f"[collect] panel_idx={args.panel_idx}  split={split}  "
      f"revit_id={panel['revit_id']}  floor={panel['floor']}", flush=True)


# ---------------------------------------------------------------------------
# Output layout (shared across per-panel runs)
# ---------------------------------------------------------------------------

os.makedirs(args.output, exist_ok=True)
frames_dir = os.path.join(args.output, "frames", split)
os.makedirs(frames_dir, exist_ok=True)
labels_all_path   = os.path.join(args.output, "labels.jsonl")
labels_split_path = os.path.join(args.output, f"labels_{split}.jsonl")
metadata_path     = os.path.join(args.output, "metadata.json")

print(f"[collect] output: {args.output}", flush=True)
print(f"[collect]   frames_dir:   {frames_dir}", flush=True)
print(f"[collect]   labels_all:   {labels_all_path}", flush=True)
print(f"[collect]   labels_split: {labels_split_path}", flush=True)


# ---------------------------------------------------------------------------
# Scene + physics + materials
# ---------------------------------------------------------------------------

world = World()
stage = omni.usd.get_context().get_stage()

building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(
    args.usd, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)

# Building bbox (for anchor_xyz + always-inside camera pick).
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
building_bbox = bbox_cache.ComputeWorldBound(
    stage.GetPrimAtPath("/World/Building")).ComputeAlignedRange()
bmn, bmx = building_bbox.GetMin(), building_bbox.GetMax()
building_centroid = (
    float((bmn[0] + bmx[0]) / 2),
    float((bmn[1] + bmx[1]) / 2),
    float((bmn[2] + bmx[2]) / 2),
)

# Lighting rig -- carb settings + dome + fill. Anchor the fill somewhere
# meaningful (building centroid); the main loop moves it above the camera
# each frame via set_fill_position.
light_handles = setup_lighting_rig(stage, anchor_xyz=building_centroid)


# Materials: 1 intact + 9 cracked (3 variants x 3 uv_tilings).
def _intact_material(path):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset("OmniGlass.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set((0.90, 0.95, 0.97))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.03)
    sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.003)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


UsdGeom.Scope.Define(stage, Sdf.Path("/World/Looks"))
intact_path = "/World/Looks/Intact"
_intact_material(intact_path)

cracked_paths = {}
for variant in args.variants:
    for uv_tiling in args.uv_tilings:
        path = f"/World/Looks/Crack_{variant}_uv{uv_tiling}"
        normal_tex    = os.path.join(TEXTURE_DIR, f"crack_normal_{variant}.png")
        opacity_tex   = os.path.join(TEXTURE_DIR, f"crack_opacity_{variant}.png")
        roughness_tex = os.path.join(TEXTURE_DIR, f"crack_roughness_{variant}.png")
        create_cracked_glass_material(
            stage, material_path=path,
            normal_texture=normal_tex,
            opacity_texture=opacity_tex   if os.path.isfile(opacity_tex)   else None,
            roughness_texture=roughness_tex if os.path.isfile(roughness_tex) else None,
            crack_hits=UV_TILING_TO_HITS[uv_tiling],
        )
        cracked_paths[(variant, uv_tiling)] = path

print(f"[collect] authored {1 + len(cracked_paths)} materials "
      f"(1 intact + {len(cracked_paths)} cracked)", flush=True)


# ---------------------------------------------------------------------------
# Panel pose + 27-camera cache (Tanner's pattern)
#
# `rep.orchestrator.step()` renders EVERY attached render_product per call.
# Our first attempt captured 1 camera per step, wasting the other 26 renders
# -- 0.3 fps. Isaac 5.1's orchestrator.step doesn't accept a render_products
# filter kwarg (TypeError confirmed), so we follow Tanner's pattern from
# `origin/tanner_vlm:simulations/robot/door_static_collection.py:826`: call
# orchestrator.step ONCE per world state, then read all 27 annotators.
# Restructured main loop: outer = (lighting, state, uv_tiling), inner = read
# all 27 cameras. 72 world states * 27 cameras = 1,944 frames per panel,
# same total, 27x less render waste.
# ---------------------------------------------------------------------------

def panel_pose(panel_meta, arc_angle, arc_distance, height_offset):
    """Camera position + look-at for `panel_meta`. Always picks the side
    whose centroid-distance is smaller -- i.e. inside the building."""
    prim = stage.GetPrimAtPath(panel_meta["prim_path"])
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2)
    dims = [mx[i]-mn[i] for i in range(3)]
    axis = min(range(3), key=lambda i: dims[i])   # panel normal = thinnest axis
    n = [0.0, 0.0, 0.0]; n[axis] = 1.0
    p_plus = (c[0]+n[0]*arc_distance, c[1]+n[1]*arc_distance, c[2]+n[2]*arc_distance)
    p_neg  = (c[0]-n[0]*arc_distance, c[1]-n[1]*arc_distance, c[2]-n[2]*arc_distance)
    d_plus = sum((p_plus[i]-building_centroid[i])**2 for i in range(3))
    d_neg  = sum((p_neg[i] -building_centroid[i])**2 for i in range(3))
    cam = p_plus if d_plus <= d_neg else p_neg
    if abs(arc_angle) > 1e-6:
        dx, dy = cam[0]-c[0], cam[1]-c[1]
        cos_a, sin_a = math.cos(arc_angle), math.sin(arc_angle)
        rx = cos_a*dx - sin_a*dy
        ry = sin_a*dx + cos_a*dy
        cam = (c[0]+rx, c[1]+ry, cam[2])
    if abs(height_offset) > 1e-6:
        cam = (cam[0], cam[1], cam[2] + height_offset)
    return cam, c


_cameras = []   # list of {arc_angle, arc_distance, height, cam_pos, annotator}
for arc_angle in args.arc_angles:
    for arc_distance in args.arc_distances:
        for height in args.heights:
            cam_pos, look_at = panel_pose(panel, arc_angle, arc_distance, height)
            cam_node = rep.create.camera(
                position=cam_pos, look_at=look_at,
                focal_length=12.0, horizontal_aperture=20.955,
                name=f"CGCam_pan{args.panel_idx:02d}"
                     f"_a{int(round(arc_angle*1000))}"
                     f"_d{int(round(arc_distance*1000))}"
                     f"_h{int(round(height*1000))}")
            rp = rep.create.render_product(cam_node, RESOLUTION)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            _cameras.append({
                "arc_angle":    arc_angle,
                "arc_distance": arc_distance,
                "height":       height,
                "cam_pos":      cam_pos,
                "annotator":    ann,
            })
print(f"[collect] pre-created {len(_cameras)} cameras", flush=True)

# Panel-centric static fill: place on interior side, 1.5m above the
# median-viewpoint camera. Matches Tanner's pattern where the fill sits
# above the door center, NOT above each camera (door_static_collection.py
# line 351). Intensity + color are still updated per cell by
# apply_lighting_preset; only the position is fixed.
_mid_arc  = args.arc_angles[len(args.arc_angles)  // 2]
_mid_dist = args.arc_distances[len(args.arc_distances) // 2]
_mid_cam_pos, _ = panel_pose(panel, _mid_arc, _mid_dist, 0.0)
_fill_pos = (_mid_cam_pos[0], _mid_cam_pos[1], _mid_cam_pos[2] + 1.5)
set_fill_position(light_handles, _fill_pos)
print(f"[collect] fill light fixed at {_fill_pos}", flush=True)


# ---------------------------------------------------------------------------
# Material binding (matches build_cracked_scene_usd.py:68-84)
# ---------------------------------------------------------------------------

def bind_panel_material(material_path):
    prim = stage.GetPrimAtPath(panel["prim_path"])
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    mesh = _find_mesh_descendant(prim, max_depth=4)
    for target in (prim, mesh):
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try: api.UnbindDirectBinding()
        except Exception: pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

world.reset()
for _ in range(args.warmup_steps):
    world.step(render=False)
print(f"[collect] warmup done ({args.warmup_steps} steps)", flush=True)


# ---------------------------------------------------------------------------
# Main loop -- 6 nested axes
# ---------------------------------------------------------------------------

def _frame_id(label_kind, lighting, state, arc_angle, arc_distance, uv_tiling,
              height, frame_num):
    deg = int(round(math.degrees(arc_angle)))
    return (f"{label_kind}_pan{args.panel_idx:02d}"
            f"_L-{lighting}"
            f"_M-{state}"
            f"_arc{deg:+03d}"
            f"_d{int(arc_distance*10):03d}"
            f"_uv{uv_tiling}"
            f"_h{int(height*10):+03d}"
            f"_f{frame_num:05d}")


total_frames = (len(args.lightings) * len(STATES) * len(args.arc_angles)
                * len(args.arc_distances) * len(args.uv_tilings)
                * len(args.heights))
print(f"[collect] panel {args.panel_idx}: {total_frames} frames to render",
      flush=True)

t0 = time.time()
frame_num = 0
labels_all_f   = open(labels_all_path,   "a")
labels_split_f = open(labels_split_path, "a")

n_cams = len(_cameras)
try:
    for lighting in args.lightings:
        for state in STATES:
            for uv_tiling in args.uv_tilings:
                # 1. Bind material (same material across the 27 captures).
                if state == "intact":
                    bind_panel_material(intact_path)
                    label_kind = "intact"
                    variant_or_clear = "clear"
                else:
                    bind_panel_material(cracked_paths[(state, uv_tiling)])
                    label_kind = "cracked"
                    variant_or_clear = state

                # 2. Apply lighting preset (fill position is static, set at
                #    panel startup; only intensity + color change here).
                dome_int, dome_temp, fill_int, fill_temp = \
                    apply_lighting_preset(light_handles, lighting, rng=rng)

                # 3. Render ONCE -- all 27 camera annotators update per step.
                for _ in range(args.render_steps):
                    rep.orchestrator.step(rt_subframes=4)

                # 4. Capture all 27 viewpoints from this single world state.
                for cam_rec in _cameras:
                    arc_angle    = cam_rec["arc_angle"]
                    arc_distance = cam_rec["arc_distance"]
                    height       = cam_rec["height"]
                    cam_pos      = cam_rec["cam_pos"]
                    data = cam_rec["annotator"].get_data()
                    img = np.asarray(data)
                    if img.ndim == 3 and img.shape[-1] == 4:
                        img = img[..., :3]
                    img = img.astype(np.uint8)

                    fid = _frame_id(
                        label_kind, lighting, variant_or_clear,
                        arc_angle, arc_distance, uv_tiling,
                        height, frame_num)
                    jpeg_path = os.path.join(frames_dir, f"{fid}.jpg")
                    Image.fromarray(img).save(
                        jpeg_path, "JPEG", quality=args.jpeg_quality)

                    row = {
                        "frame_id":        fid,
                        "label":           label_kind,
                        "split":           split,
                        "variant":         variant_or_clear,
                        "panel_idx":       args.panel_idx,
                        "panel_revit_id":  panel["revit_id"],
                        "panel_prim":      panel["prim_path"],
                        "panel_world_xyz": panel["world_xyz"],
                        "floor":           panel["floor"],
                        "lighting_preset": lighting,
                        "lighting_jitter": {
                            "dome_int":  dome_int,
                            "dome_temp": dome_temp,
                            "fill_int":  fill_int,
                            "fill_temp": fill_temp,
                        },
                        "arc_angle_rad":   arc_angle,
                        "arc_distance_m":  arc_distance,
                        "uv_tiling":       uv_tiling,
                        "height_offset_m": height,
                        "camera_xyz":      [float(c) for c in cam_pos],
                        "image":           os.path.relpath(jpeg_path, args.output),
                    }
                    line = json.dumps(row) + "\n"
                    labels_all_f.write(line)
                    labels_split_f.write(line)
                    frame_num += 1

                # Progress heartbeat every 10 world states (~270 frames).
                outer_idx = frame_num // n_cams
                if outer_idx % 10 == 0 or frame_num == total_frames:
                    labels_all_f.flush()
                    labels_split_f.flush()
                    elapsed = time.time() - t0
                    rate = frame_num / max(elapsed, 1e-6)
                    eta = (total_frames - frame_num) / max(rate, 1e-6)
                    print(f"[collect] pan{args.panel_idx:02d} "
                          f"[{frame_num:4d}/{total_frames}]  "
                          f"{rate:.1f} fps  "
                          f"elapsed={elapsed/60:.1f}m  "
                          f"eta={eta/60:.1f}m",
                          flush=True)

finally:
    labels_all_f.flush();   labels_all_f.close()
    labels_split_f.flush(); labels_split_f.close()


# ---------------------------------------------------------------------------
# Metadata -- read-modify-write so later panels' metadata merges cleanly
# ---------------------------------------------------------------------------

panel_record = {
    "panel_idx":    args.panel_idx,
    "split":        split,
    "revit_id":     panel["revit_id"],
    "floor":        panel["floor"],
    "frames":       frame_num,
    "seconds":      time.time() - t0,
    "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
}

if os.path.isfile(metadata_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
else:
    metadata = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cli": {
            "variants":      args.variants,
            "lightings":     args.lightings,
            "arc_angles":    args.arc_angles,
            "arc_distances": args.arc_distances,
            "uv_tilings":    args.uv_tilings,
            "heights":       args.heights,
            "resolution":    args.resolution,
            "jpeg_quality":  args.jpeg_quality,
            "seed":          args.seed,
        },
        "panels": [],
    }

metadata["panels"].append(panel_record)
metadata["total_frames"] = sum(p["frames"] for p in metadata["panels"])
metadata["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[collect] DONE panel {args.panel_idx}: "
      f"{frame_num} frames in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"[collect] running total across all panels: "
      f"{metadata['total_frames']} frames", flush=True)

simulation_app.close()
