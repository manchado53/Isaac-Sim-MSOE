#!/usr/bin/env python3
"""
Sweep preview renderer for the cracked-glass dataset.

Generates 7 contact sheets — one per sweep axis of the full dataset
collection — so the dataset designer can visually inspect what each
axis does BEFORE committing to the full run.

Axes previewed (from `DATASET_PLAN.md`):

    Grid:
      - Panel             13 curated panels from selected_panels.json
      - Lighting preset   6 (Tanner's canonical set)
      - Crack state       4 (intact + 3 variants)
      - Arc angle         3 (-0.4, 0, +0.4 rad)

    Per-frame random:
      - UV tiling         {1, 2, 4}
      - Arc distance      {1.5, 2.0, 2.5} m
      - Height offset     {-0.5, 0, +0.5} m  (Tanner's 3rd geometric axis)

Output:

    datasets/cracked_glass/sweep_preview_<timestamp>/
    ├── cells/                         full-res 1280×720 individual PNGs
    ├── contact_sheets/
    │   ├── panel_x_variant.png       13 panels × 4 variants        (52 cells)
    │   ├── lighting_x_variant.png    6 lightings × 4 variants      (24 cells)
    │   ├── uv_tiling.png             3 tilings × 3 variants        (9 cells)
    │   ├── arc_distance.png          3 distances × 3 variants      (9 cells)
    │   ├── arc_angle.png             3 arc angles × 1 variant      (3 cells)
    │   ├── color_temp.png            5 color temps × 1 cell        (5 cells)
    │   └── height_offset.png         3 heights × 3 variants        (9 cells)
    └── manifest.json

Usage:
    ~/isaacsim/python.sh SyntheticDataGen/cracked_glass/sweep_preview.py

    # Override canonical lighting / panel used for the axis sheets:
    ~/isaacsim/python.sh SyntheticDataGen/cracked_glass/sweep_preview.py \\
        --canonical-lighting bright_fluorescent \\
        --canonical-panel-idx 6
"""

import argparse
import datetime
import json
import math
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)
sys.path.insert(0, _script_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--selected-panels", default=os.path.join(
        _script_dir, "selected_panels.json"))
    p.add_argument("--usd", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--output", default=None)
    p.add_argument("--canonical-lighting", default="bright_fluorescent",
                   help="Lighting preset used for sheets that fix lighting.")
    p.add_argument("--canonical-panel-idx", type=int, default=2,
                   help="panel_idx value from selected_panels.json used "
                        "for sheets that fix the panel.")
    p.add_argument("--canonical-variant", default="tempered_v5",
                   help="Variant used for sheets that fix the variant.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cell-width", type=int, default=320,
                   help="Downscaled contact-sheet cell width (full-res cells "
                        "are kept under cells/). 320 px keeps the biggest "
                        "sheet (13 panels × 4 variants) under 1300×2400.")
    p.add_argument("--resolution", default="1280x720")
    return p.parse_args()


args = parse_args()


# ---------------------------------------------------------------------------
# Lighting presets — lifted from origin/tanner_vlm door_static_collection.py
# ---------------------------------------------------------------------------

LIGHTING_PRESETS = {
    "bright_fluorescent":
        dict(dome_int=25000, dome_temp=5500, fill_int=400000, fill_temp=4500),
    "daylight_flood":
        dict(dome_int=70000, dome_temp=6500, fill_int=150000, fill_temp=6000),
    "warm_evening":
        dict(dome_int=20000, dome_temp=3000, fill_int=200000, fill_temp=2800),
    "dim_night":
        dict(dome_int=5000,  dome_temp=4000, fill_int=50000,  fill_temp=3500),
    "overcast_flat":
        dict(dome_int=40000, dome_temp=5000, fill_int=100000, fill_temp=5000),
    "high_contrast":
        dict(dome_int=5000,  dome_temp=5000, fill_int=700000, fill_temp=5500),
}


# ---------------------------------------------------------------------------
# Launch SimulationApp FIRST (before any pxr / omni imports)
# ---------------------------------------------------------------------------

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene, _find_mesh_descendant,
)
from glass_anomaly import create_cracked_glass_material  # noqa: E402


TEXTURE_DIR = os.path.join(_project_dir, "assets", "materials", "cracked_glass")
W, H = (int(x) for x in args.resolution.lower().split("x"))
RESOLUTION = (W, H)

random.seed(args.seed)
rng = random.Random(args.seed)


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

with open(args.selected_panels) as f:
    selected = json.load(f)
panels_meta = selected["panels"]        # 13 records
panel_by_idx = {p["panel_idx"]: p for p in panels_meta}

output_root = args.output or os.path.join(
    _project_dir, "datasets", "cracked_glass",
    f"sweep_preview_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
cells_dir = os.path.join(output_root, "cells")
sheets_dir = os.path.join(output_root, "contact_sheets")
os.makedirs(cells_dir, exist_ok=True)
os.makedirs(sheets_dir, exist_ok=True)
print(f"[sweep] Output: {output_root}", flush=True)
print(f"[sweep] Panels to render: {len(panels_meta)}", flush=True)

world = World()
stage = omni.usd.get_context().get_stage()

building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(
    args.usd, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)

# Carb settings validated in cracked_glass_preview.py — let dome light
# contribute inside closed rooms (otherwise interiors go black under path
# tracing), and raise maxBounces so light from the bright curtain wall
# reaches deep into atria.
carb.settings.get_settings().set("/rtx/domeLight/upperLowerStrategy", 3)
carb.settings.get_settings().set("/rtx/pathTracing/maxBounces", 8)


# ---------------------------------------------------------------------------
# Material authoring — one per (variant, uv_tiling) combo + one intact
# ---------------------------------------------------------------------------

UV_TILING_TO_HITS = {1: 0, 2: 4, 4: 16}
VARIANTS = ("edge_v1", "v4", "tempered_v5")

def _glass_common_material(path):
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
_glass_common_material(intact_path)
print(f"[sweep] Authored intact material at {intact_path}", flush=True)

cracked_paths = {}              # (variant, uv_tiling) -> material path
for variant in VARIANTS:
    for uv_tiling in (1, 2, 4):
        path = f"/World/Looks/Crack_{variant}_uv{uv_tiling}"
        normal_tex = os.path.join(TEXTURE_DIR, f"crack_normal_{variant}.png")
        opacity_tex = os.path.join(TEXTURE_DIR, f"crack_opacity_{variant}.png")
        roughness_tex = os.path.join(TEXTURE_DIR, f"crack_roughness_{variant}.png")
        create_cracked_glass_material(
            stage, material_path=path,
            normal_texture=normal_tex,
            opacity_texture=opacity_tex if os.path.isfile(opacity_tex) else None,
            roughness_texture=roughness_tex if os.path.isfile(roughness_tex) else None,
            crack_hits=UV_TILING_TO_HITS[uv_tiling],
        )
        cracked_paths[(variant, uv_tiling)] = path
print(f"[sweep] Authored {len(cracked_paths)} crack materials "
      f"(3 variants × 3 UV tilings)", flush=True)


# ---------------------------------------------------------------------------
# Lighting rig — one DomeLight + one SphereLight fill (movable per cell)
# ---------------------------------------------------------------------------

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/PreviewDome"))
dome.CreateIntensityAttr(0.0)                               # set per cell
dome.CreateColorTemperatureAttr(5500.0)
dome.CreateEnableColorTemperatureAttr(True)

fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/PreviewFill"))
fill.CreateIntensityAttr(0.0)
fill.CreateRadiusAttr(0.1)
fill.CreateColorTemperatureAttr(4500.0)
fill.CreateEnableColorTemperatureAttr(True)
fill_xform = UsdGeom.XformCommonAPI(fill.GetPrim())
fill_xform.SetTranslate((0.0, 0.0, 0.0))


def apply_lighting(preset_name, jitter_rng=None,
                   dome_temp_override=None, fill_temp_override=None,
                   dome_int_override=None, fill_int_override=None):
    """Configure the pre-created dome + fill lights for a single cell.

    Jitter (if `jitter_rng` given) is Tanner's pattern: intensity × U(0.9,1.1)
    and temp + U(-200, +200) K. Overrides win over jitter, so the
    color-temp sweep sheet can pin temps exactly while still applying
    intensity jitter.
    """
    preset = LIGHTING_PRESETS[preset_name]
    di, dt = float(preset["dome_int"]), float(preset["dome_temp"])
    fi, ft = float(preset["fill_int"]), float(preset["fill_temp"])
    if jitter_rng is not None:
        di *= jitter_rng.uniform(0.9, 1.1)
        fi *= jitter_rng.uniform(0.9, 1.1)
        dt += jitter_rng.uniform(-200, 200)
        ft += jitter_rng.uniform(-200, 200)
    if dome_int_override is not None:  di = float(dome_int_override)
    if fill_int_override is not None:  fi = float(fill_int_override)
    if dome_temp_override is not None: dt = float(dome_temp_override)
    if fill_temp_override is not None: ft = float(fill_temp_override)
    dome.GetIntensityAttr().Set(di)
    dome.GetColorTemperatureAttr().Set(dt)
    fill.GetIntensityAttr().Set(fi)
    fill.GetColorTemperatureAttr().Set(ft)


def position_fill_above_camera(cam_pos, offset_z=1.5):
    fill_xform.SetTranslate((cam_pos[0], cam_pos[1], cam_pos[2] + offset_z))


# ---------------------------------------------------------------------------
# Panel poses — always-inside camera, per-panel arc
# ---------------------------------------------------------------------------

world.reset()
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
building_bbox = bbox_cache.ComputeWorldBound(building_prim).ComputeAlignedRange()


def panel_pose(panel_meta, arc_angle=0.0, arc_distance=1.5, height_offset=0.0):
    """Compute camera + look-at for a curated panel, inside-side, with
    rotation (``arc_angle`` rad, around world-Z) and vertical offset
    (``height_offset`` m, applied to both camera and target Z so the
    camera stays level with the shifted look-at point)."""
    prim = stage.GetPrimAtPath(panel_meta["prim_path"])
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2)
    dims = [mx[i]-mn[i] for i in range(3)]
    axis = min(range(3), key=lambda i: dims[i])
    n = [0.0, 0.0, 0.0]; n[axis] = 1.0
    p_plus = (c[0]+n[0]*arc_distance, c[1]+n[1]*arc_distance, c[2]+n[2]*arc_distance)
    p_neg  = (c[0]-n[0]*arc_distance, c[1]-n[1]*arc_distance, c[2]-n[2]*arc_distance)
    bmn, bmx = building_bbox.GetMin(), building_bbox.GetMax()
    bc = ((bmn[0]+bmx[0])/2, (bmn[1]+bmx[1])/2, (bmn[2]+bmx[2])/2)
    d_plus = sum((p_plus[i]-bc[i])**2 for i in range(3))
    d_neg  = sum((p_neg[i]-bc[i])**2  for i in range(3))
    cam = p_plus if d_plus <= d_neg else p_neg
    # Rotate camera around world-Z axis (panel up) by arc_angle.
    if abs(arc_angle) > 1e-6:
        dx, dy = cam[0] - c[0], cam[1] - c[1]
        cos_a, sin_a = math.cos(arc_angle), math.sin(arc_angle)
        rx = cos_a * dx - sin_a * dy
        ry = sin_a * dx + cos_a * dy
        cam = (c[0] + rx, c[1] + ry, cam[2])
    # Apply height offset — raise the camera above (or drop it below) the
    # panel's centerline while keeping the look-at point fixed at the
    # panel. This tilts the camera up/down rather than just sliding it.
    if abs(height_offset) > 1e-6:
        cam = (cam[0], cam[1], cam[2] + height_offset)
    return cam, c


# ---------------------------------------------------------------------------
# Material binding (matches bind_visual_material_deep from build script)
# ---------------------------------------------------------------------------

def bind_panel_material(panel_meta, material_path):
    prim = stage.GetPrimAtPath(panel_meta["prim_path"])
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
# Cell render: bind, light, camera, step, capture
# ---------------------------------------------------------------------------

# Cache one Replicator camera + render_product + annotator per unique
# (panel_idx, arc_angle, arc_distance) triplet. Creating a fresh camera
# PER CELL piles up Hydra graph state until it crashes (~60 cells in);
# trying to mutate one persistent camera's USD xform doesn't stick because
# Replicator's OmniGraph node reclaims the prim each tick. Caching is the
# middle ground: for 110 cells we end up with <20 cameras total (one per
# distinct viewpoint), each configured once at creation and reused across
# all lighting / material / jitter variations at that viewpoint.
_cam_cache = {}   # (panel_idx, arc_angle, arc_distance, height_offset) -> (cam_node, render_product, rgb_annotator, cam_pos)


def _get_cam(panel_meta, arc_angle, arc_distance, height_offset=0.0):
    key = (panel_meta["panel_idx"],
           round(arc_angle, 3), round(arc_distance, 3),
           round(height_offset, 3))
    if key in _cam_cache:
        return _cam_cache[key]
    cam_pos, look_at = panel_pose(
        panel_meta, arc_angle, arc_distance, height_offset)
    cam_node = rep.create.camera(
        position=cam_pos, look_at=look_at,
        focal_length=12.0, horizontal_aperture=20.955,
        name=f"SweepCam_p{panel_meta['panel_idx']:02d}"
             f"_a{int(round(arc_angle*1000))}"
             f"_d{int(round(arc_distance*1000))}"
             f"_h{int(round(height_offset*1000))}")
    rp = rep.create.render_product(cam_node, RESOLUTION)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    _cam_cache[key] = (cam_node, rp, ann, cam_pos)
    return _cam_cache[key]


def render_cell(panel_meta, variant, lighting,
                arc_angle=0.0, arc_distance=1.5, height_offset=0.0,
                uv_tiling=1, jitter=False,
                dome_temp=None, fill_temp=None, name=""):
    """Render one sweep cell to disk and return the file path + RGB array."""
    # 1. Material
    if variant == "intact":
        bind_panel_material(panel_meta, intact_path)
    else:
        bind_panel_material(panel_meta, cracked_paths[(variant, uv_tiling)])

    # 2. Camera pose — cached per unique viewpoint
    cam_node, rp, ann, cam_pos = _get_cam(
        panel_meta, arc_angle, arc_distance, height_offset)
    position_fill_above_camera(cam_pos)

    # 3. Lighting (jitter seeded deterministically from the cell name so
    #    jitter-sheet cells differ but re-running is reproducible)
    if jitter:
        cell_rng = random.Random(f"{name}-{args.seed}")
        apply_lighting(lighting, jitter_rng=cell_rng,
                       dome_temp_override=dome_temp,
                       fill_temp_override=fill_temp)
    else:
        apply_lighting(lighting,
                       dome_temp_override=dome_temp,
                       fill_temp_override=fill_temp)

    # 4. Step + capture from the cached camera/annotator for this viewpoint.
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=4)
    data = ann.get_data()
    img = np.asarray(data)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.uint8)
    # Cells are grouped by sheet via an "sN_" prefix in `name` — route
    # each cell into its own subfolder so `cells/` is browseable per axis
    # instead of being a flat pile of 110 files.
    _SHEET_DIRS = {
        "s1": "sheet1_panel_x_variant",
        "s2": "sheet2_lighting_x_variant",
        "s3": "sheet3_uv_tiling",
        "s4": "sheet4_arc_distance",
        "s5": "sheet5_arc_angle",
        "s6": "sheet6_color_temp",
        "s7": "sheet7_height_offset",
    }
    sub = _SHEET_DIRS.get(name[:2], "misc")
    sub_dir = os.path.join(cells_dir, sub)
    os.makedirs(sub_dir, exist_ok=True)
    out_path = os.path.join(sub_dir, f"{name}.png")
    Image.fromarray(img).save(out_path)
    return out_path, img


# ---------------------------------------------------------------------------
# Contact sheet assembly
# ---------------------------------------------------------------------------

def _load_cell(path, cell_w):
    im = Image.open(path).convert("RGB")
    aspect = im.width / im.height
    cell_h = int(cell_w / aspect)
    return im.resize((cell_w, cell_h), Image.LANCZOS), cell_h


def build_sheet(out_path, rows, title="", col_labels=None, row_labels=None):
    """rows is a list of lists of cell-png paths. All rows must have the
    same length. Out path is the contact sheet PNG."""
    cell_w = args.cell_width
    # Probe one cell for height
    _, cell_h = _load_cell(rows[0][0], cell_w)
    pad = 8
    header_h = 40 if col_labels else 0
    title_h = 30 if title else 0
    label_w = 100 if row_labels else 0

    n_rows = len(rows)
    n_cols = len(rows[0])
    total_w = label_w + n_cols * (cell_w + pad) + pad
    total_h = title_h + header_h + n_rows * (cell_h + pad) + pad

    sheet = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_small = ImageFont.load_default()
        font_title = font_small

    if title:
        draw.text((pad, 6), title, fill=(235, 235, 240), font=font_title)
    if col_labels:
        for ci, lbl in enumerate(col_labels):
            x = label_w + pad + ci * (cell_w + pad)
            draw.text((x, title_h + 10), lbl, fill=(200, 200, 210),
                      font=font_small)
    for ri, row in enumerate(rows):
        y = title_h + header_h + ri * (cell_h + pad) + pad
        if row_labels:
            draw.text((6, y + cell_h // 2 - 8), row_labels[ri],
                      fill=(200, 200, 210), font=font_small)
        for ci, cell_path in enumerate(row):
            x = label_w + pad + ci * (cell_w + pad)
            im, _ = _load_cell(cell_path, cell_w)
            sheet.paste(im, (x, y))
    sheet.save(out_path)
    print(f"[sweep] Wrote {out_path} ({total_w}×{total_h})", flush=True)


# ---------------------------------------------------------------------------
# Sheet plans (what to render)
# ---------------------------------------------------------------------------

STATES = ["intact", "edge_v1", "v4", "tempered_v5"]
t0 = datetime.datetime.now()
manifest_rows = []

canonical_panel = panel_by_idx[args.canonical_panel_idx]


def log(msg, start):
    dt = (datetime.datetime.now() - start).total_seconds()
    print(f"[sweep] {msg}  (+{dt:.1f}s)", flush=True)


# --- Sheet 1: panel × variant -------------------------------------------------
log("Sheet 1/7 panel_x_variant (52 cells)…", t0)
grid1 = []
for pm in panels_meta:
    row = []
    for state in STATES:
        variant = "intact" if state == "intact" else state
        name = f"s1_p{pm['panel_idx']:02d}_{variant}"
        path, _ = render_cell(
            pm, variant, args.canonical_lighting,
            arc_angle=0.0, arc_distance=1.5, uv_tiling=1, name=name)
        row.append(path)
        manifest_rows.append({
            "sheet": "panel_x_variant", "panel_idx": pm["panel_idx"],
            "variant": variant, "lighting": args.canonical_lighting,
            "arc_angle": 0.0, "arc_distance": 1.5, "uv_tiling": 1,
            "file": os.path.relpath(path, cells_dir),
        })
    grid1.append(row)
build_sheet(
    os.path.join(sheets_dir, "panel_x_variant.png"), grid1,
    title=f"Panel × Variant  (lighting={args.canonical_lighting}, arc=0, "
          f"d=1.5m, uv=1)",
    col_labels=STATES,
    row_labels=[f"#{pm['panel_idx']:02d}" for pm in panels_meta])


# --- Sheet 2: lighting × variant ---------------------------------------------
log("Sheet 2/7 lighting_x_variant (24 cells)…", t0)
grid2 = []
for lighting in LIGHTING_PRESETS.keys():
    row = []
    for state in STATES:
        variant = "intact" if state == "intact" else state
        name = f"s2_p{canonical_panel['panel_idx']:02d}_{lighting}_{variant}"
        path, _ = render_cell(
            canonical_panel, variant, lighting,
            arc_angle=0.0, arc_distance=1.5, uv_tiling=1, name=name)
        row.append(path)
        manifest_rows.append({
            "sheet": "lighting_x_variant",
            "panel_idx": canonical_panel["panel_idx"],
            "variant": variant, "lighting": lighting,
            "arc_angle": 0.0, "arc_distance": 1.5, "uv_tiling": 1,
            "file": os.path.relpath(path, cells_dir),
        })
    grid2.append(row)
build_sheet(
    os.path.join(sheets_dir, "lighting_x_variant.png"), grid2,
    title=f"Lighting × Variant  (panel #{canonical_panel['panel_idx']:02d}, "
          f"arc=0, d=1.5m, uv=1)",
    col_labels=STATES, row_labels=list(LIGHTING_PRESETS.keys()))


# --- Sheet 3: UV tiling × variant --------------------------------------------
log("Sheet 3/7 uv_tiling (9 cells)…", t0)
grid3 = []
for uv in (1, 2, 4):
    row = []
    for variant in VARIANTS:
        name = f"s3_p{canonical_panel['panel_idx']:02d}_{variant}_uv{uv}"
        path, _ = render_cell(
            canonical_panel, variant, args.canonical_lighting,
            arc_angle=0.0, arc_distance=1.5, uv_tiling=uv, name=name)
        row.append(path)
        manifest_rows.append({
            "sheet": "uv_tiling",
            "panel_idx": canonical_panel["panel_idx"],
            "variant": variant, "lighting": args.canonical_lighting,
            "arc_angle": 0.0, "arc_distance": 1.5, "uv_tiling": uv,
            "file": os.path.relpath(path, cells_dir),
        })
    grid3.append(row)
build_sheet(
    os.path.join(sheets_dir, "uv_tiling.png"), grid3,
    title=f"UV tiling  (panel #{canonical_panel['panel_idx']:02d}, "
          f"lighting={args.canonical_lighting})",
    col_labels=list(VARIANTS),
    row_labels=[f"uv={uv}×" for uv in (1, 2, 4)])


# --- Sheet 4: arc distance × variant -----------------------------------------
log("Sheet 4/7 arc_distance (9 cells)…", t0)
grid4 = []
for dist in (1.5, 2.0, 2.5):
    row = []
    for variant in VARIANTS:
        name = (f"s4_p{canonical_panel['panel_idx']:02d}_"
                f"{variant}_d{dist:.1f}")
        path, _ = render_cell(
            canonical_panel, variant, args.canonical_lighting,
            arc_angle=0.0, arc_distance=dist, uv_tiling=1, name=name)
        row.append(path)
        manifest_rows.append({
            "sheet": "arc_distance",
            "panel_idx": canonical_panel["panel_idx"],
            "variant": variant, "lighting": args.canonical_lighting,
            "arc_angle": 0.0, "arc_distance": dist, "uv_tiling": 1,
            "file": os.path.relpath(path, cells_dir),
        })
    grid4.append(row)
build_sheet(
    os.path.join(sheets_dir, "arc_distance.png"), grid4,
    title=f"Arc distance  (panel #{canonical_panel['panel_idx']:02d}, "
          f"lighting={args.canonical_lighting})",
    col_labels=list(VARIANTS),
    row_labels=[f"d={d}m" for d in (1.5, 2.0, 2.5)])


# --- Sheet 5: arc angle ------------------------------------------------------
log("Sheet 5/7 arc_angle (3 cells)…", t0)
row5 = []
for ang in (-0.4, 0.0, 0.4):
    deg = int(round(math.degrees(ang)))
    name = (f"s5_p{canonical_panel['panel_idx']:02d}_"
            f"{args.canonical_variant}_arc{deg:+d}")
    path, _ = render_cell(
        canonical_panel, args.canonical_variant, args.canonical_lighting,
        arc_angle=ang, arc_distance=1.5, uv_tiling=1, name=name)
    row5.append(path)
    manifest_rows.append({
        "sheet": "arc_angle",
        "panel_idx": canonical_panel["panel_idx"],
        "variant": args.canonical_variant,
        "lighting": args.canonical_lighting,
        "arc_angle": ang, "arc_distance": 1.5, "uv_tiling": 1,
        "file": os.path.relpath(path, cells_dir),
    })
build_sheet(
    os.path.join(sheets_dir, "arc_angle.png"), [row5],
    title=f"Arc angle  (panel #{canonical_panel['panel_idx']:02d}, "
          f"lighting={args.canonical_lighting}, "
          f"variant={args.canonical_variant})",
    col_labels=[f"{math.degrees(a):+.0f}°" for a in (-0.4, 0.0, 0.4)])


# --- Sheet 6: color temperature ----------------------------------------------
log("Sheet 6/7 color_temp (5 cells)…", t0)
row6 = []
for temp_k in (3000, 4000, 5000, 6500, 8000):
    name = (f"s6_p{canonical_panel['panel_idx']:02d}_"
            f"{args.canonical_variant}_T{temp_k}")
    path, _ = render_cell(
        canonical_panel, args.canonical_variant, args.canonical_lighting,
        arc_angle=0.0, arc_distance=1.5, uv_tiling=1,
        dome_temp=temp_k, fill_temp=temp_k, name=name)
    row6.append(path)
    manifest_rows.append({
        "sheet": "color_temp",
        "panel_idx": canonical_panel["panel_idx"],
        "variant": args.canonical_variant,
        "lighting": args.canonical_lighting,
        "color_temperature_k": temp_k,
        "arc_angle": 0.0, "arc_distance": 1.5, "uv_tiling": 1,
        "file": os.path.relpath(path, cells_dir),
    })
build_sheet(
    os.path.join(sheets_dir, "color_temp.png"), [row6],
    title=f"Color temperature  (panel #{canonical_panel['panel_idx']:02d}, "
          f"variant={args.canonical_variant}, base preset "
          f"intensities from {args.canonical_lighting})",
    col_labels=[f"{t} K" for t in (3000, 4000, 5000, 6500, 8000)])


# --- Sheet 7: camera height offset × variant (Tanner's 3rd geometric axis) ---
log("Sheet 7/7 height_offset (9 cells)…", t0)
grid7 = []
for h in (-0.5, 0.0, 0.5):
    row = []
    for variant in VARIANTS:
        name = (f"s7_p{canonical_panel['panel_idx']:02d}_"
                f"{variant}_h{h:+.1f}")
        path, _ = render_cell(
            canonical_panel, variant, args.canonical_lighting,
            arc_angle=0.0, arc_distance=1.5, height_offset=h,
            uv_tiling=1, name=name)
        row.append(path)
        manifest_rows.append({
            "sheet": "height_offset",
            "panel_idx": canonical_panel["panel_idx"],
            "variant": variant, "lighting": args.canonical_lighting,
            "arc_angle": 0.0, "arc_distance": 1.5,
            "height_offset": h, "uv_tiling": 1,
            "file": os.path.relpath(path, cells_dir),
        })
    grid7.append(row)
build_sheet(
    os.path.join(sheets_dir, "height_offset.png"), grid7,
    title=f"Camera height offset  (panel #{canonical_panel['panel_idx']:02d}, "
          f"lighting={args.canonical_lighting})",
    col_labels=list(VARIANTS),
    row_labels=[f"h={h:+.1f}m" for h in (-0.5, 0.0, 0.5)])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

manifest = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "usd": args.usd,
    "selected_panels_source": args.selected_panels,
    "canonical_lighting": args.canonical_lighting,
    "canonical_panel_idx": args.canonical_panel_idx,
    "canonical_variant": args.canonical_variant,
    "seed": args.seed,
    "cells_dir": "cells",
    "sheets": {
        "panel_x_variant": "contact_sheets/panel_x_variant.png",
        "lighting_x_variant": "contact_sheets/lighting_x_variant.png",
        "uv_tiling": "contact_sheets/uv_tiling.png",
        "arc_distance": "contact_sheets/arc_distance.png",
        "arc_angle": "contact_sheets/arc_angle.png",
        "color_temp": "contact_sheets/color_temp.png",
        "height_offset": "contact_sheets/height_offset.png",
    },
    "n_cells": len(manifest_rows),
    "wallclock_s": (datetime.datetime.now() - t0).total_seconds(),
    "cells": manifest_rows,
}
with open(os.path.join(output_root, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[sweep] DONE. {len(manifest_rows)} cells, "
      f"{manifest['wallclock_s']:.0f}s total. Manifest: "
      f"{output_root}/manifest.json", flush=True)

simulation_app.close()
