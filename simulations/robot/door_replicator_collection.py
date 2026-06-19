#!/usr/bin/env python3
"""
Door State Data Collection — Replicator Edition

Byte-compatible NPZ output with door_static_collection.py, but uses the
canonical omni.replicator.core rendering path:

    render_product  +  rgb annotator  +  scene-state label in custom loop

Migration scope for this first port (tracked in the 2026-04-12 SDG research
spike):

  IN — replaces with Replicator primitives:
    * Frame capture: rep.AnnotatorRegistry.get_annotator("rgb") attached to
      a per-camera rep.create.render_product, instead of Camera.get_rgba().
      This is the same readout path used successfully in
      simulations/robot/material_comparison.py after Camera.get_rgba() was
      observed to miss material-binding updates.

  KEPT — still custom (Replicator can't cleanly express these):
    * Direct 4×4 xformOp:transform rewrite for Revit-exported door meshes.
      This exists because Revit USD sublayer composition makes parent-xform
      rotation unreliable — it is a USD composition problem, not a
      Replicator limitation.
    * Physics raycast occlusion filter for candidate camera positions.
    * Discrete lighting preset application (DomeLight + ceiling RectLight
      array). Ported to rep.randomizer.light in a follow-up port.
    * CLOSED/OPEN threshold label at 5°.
    * NPZ schema (consumed unchanged by prepare_finetune_data.py).

  INTENTIONALLY NOT PORTED in this pass (kept in door_static_collection.py):
    * Legacy random mode
    * --test-mode / --sanity-grid / --flip-swing
    * Legacy sanity JPEG block at 0°/45°/90°

  Why this is "using SDG" even without rep.orchestrator.step / writer graph:
    The render_product + annotator path IS the Replicator rendering path.
    rep.orchestrator.step() is the driver for declarative graphs with
    writers and randomizer nodes; when you are reading RGBA in a custom
    loop (as we are, to pair with scene-state labels), the documented
    pattern is world.step(render=True) + annotator.get_data(). This matches
    material_comparison.py which is the only script in this repo already
    using the Replicator path. A later port can add rep.randomizer.* nodes
    and a custom Writer subclass to emit JSONL directly, replacing the
    hand-rolled randomization + NPZ loop.

Usage (drop-in for the grid-mode flow of door_static_collection.py):
    sbatch submit_isaac_job.sh simulations/robot/door_replicator_collection.py \\
        --door-prim "/World/Building/.../Doors_XXXXXXXX" \\
        --hinge-x -2047.7 --hinge-y 7969.1 \\
        --camera-center-x -20.48 --camera-center-y 80.15 \\
        --camera-z 27.8 \\
        --camera-arc-offsets -60,-45,-30,-15,0,15,30,45,60 \\
        --camera-distances 1.5,2.0,3.5 \\
        --camera-height-offsets -0.5,0.0,1.0 \\
        --output door_labels_XXXXXXXX_rep
"""

import os
import sys
import math
import random
import argparse
import functools
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# CLI (parsed before SimulationApp)
# ---------------------------------------------------------------------------

_DEFAULT_DOOR_PATH = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New/Doors_13606713"
)
_DEFAULT_HINGE_X   = -2047.728271484375
_DEFAULT_HINGE_Y   =  7969.07
_DEFAULT_CAM_CX    = -20.48
_DEFAULT_CAM_CY    =  80.15
_DEFAULT_CAM_Z     =  27.8
_DEFAULT_ARC       = "0"            # single head-on camera for visual inspection
_DEFAULT_DISTS     = "2.0"
_DEFAULT_HEIGHTS   = "0.0"
_DEFAULT_OUTPUT    = "door_labels_wood_patternsV2"

parser = argparse.ArgumentParser(
    description="Replicator-path door-state data collection (grid mode)."
)
parser.add_argument("--door-prim", default=_DEFAULT_DOOR_PATH)
parser.add_argument("--hinge-x", type=float, default=_DEFAULT_HINGE_X)
parser.add_argument("--hinge-y", type=float, default=_DEFAULT_HINGE_Y)
parser.add_argument("--camera-center-x", type=float, default=_DEFAULT_CAM_CX)
parser.add_argument("--camera-center-y", type=float, default=_DEFAULT_CAM_CY)
parser.add_argument("--camera-z", type=float, default=_DEFAULT_CAM_Z)
parser.add_argument("--camera-arc-offsets", default=_DEFAULT_ARC)
parser.add_argument("--camera-distances", default=_DEFAULT_DISTS)
parser.add_argument("--camera-height-offsets", default=_DEFAULT_HEIGHTS)
parser.add_argument("--output", default=_DEFAULT_OUTPUT)
parser.add_argument("--no-occlusion-check", action="store_true")
parser.add_argument("--no-material-variation", action="store_true")
parser.add_argument("--no-camera-jitter", action="store_true")
parser.add_argument("--grid-frames-per-cell", type=int, default=1)
parser.add_argument("--baseline-only", action="store_true",
                    help="bright_fluorescent + original material only — "
                         "matches door_static_collection.py for generalization-test doors")
parser.add_argument("--dump-jpegs", action="store_true")

args, _unknown = parser.parse_known_args()

DOOR_PATH       = args.door_prim
HINGE_X_CM      = args.hinge_x
HINGE_Y_CM      = args.hinge_y
DOOR_CENTER_XY  = np.array([args.camera_center_x, args.camera_center_y])
CAMERA_Z_BASE   = args.camera_z
CAM_ARC_OFFSETS = [float(x.strip()) for x in args.camera_arc_offsets.split(",")]
CAM_DISTANCES   = [float(x.strip()) for x in args.camera_distances.split(",")]
CAM_HEIGHT_OFFS = [float(x.strip()) for x in args.camera_height_offsets.split(",")]
OUTPUT_NAME     = args.output
DO_OCCLUSION    = not args.no_occlusion_check
MATERIAL_VAR    = not args.no_material_variation
CAMERA_JITTER   = not args.no_camera_jitter
GRID_FRAMES     = args.grid_frames_per_cell
BASELINE_ONLY   = args.baseline_only
DUMP_JPEGS      = args.dump_jpegs

CAM_RESOLUTION  = (1024, 768)
WARMUP_STEPS    = 10
CLOSED_MAX_DEG  = 5.0

LIGHTING_CONDITIONS = [
    {"name": "bright_fluorescent", "dome_int": 15_000, "dome_temp": 5500,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 400_000, "ceiling_temp": 5000},
    {"name": "daylight_flood",     "dome_int": 50_000, "dome_temp": 6500,
     "ceiling_pattern": (1, 0, 1, 0), "ceiling_int": 230_000, "ceiling_temp": 6500},
    {"name": "warm_evening",       "dome_int":  8_000, "dome_temp": 3000,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 200_000, "ceiling_temp": 2700},
    {"name": "dim_night",          "dome_int":  2_000, "dome_temp": 4000,
     "ceiling_pattern": (1, 0, 1, 0), "ceiling_int": 150_000, "ceiling_temp": 3500},
    {"name": "overcast_flat",      "dome_int": 25_000, "dome_temp": 5000,
     "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 220_000, "ceiling_temp": 5000},
    {"name": "high_contrast",      "dome_int":  2_000, "dome_temp": 5000,
     "ceiling_pattern": (0, 1, 0, 0), "ceiling_int": 700_000, "ceiling_temp": 5500},
]

DOOR_MATERIALS = [
    # A1 + A4 combined, wood tint only — holds colour/roughness/metallic
    # constant across all three presets so the pattern is the ONLY variable.
    # `diffuse_tint` multiplies the grayscale texture to give a wood-ish
    # brown surface, then brushed / grain / speckle give three different
    # surface-detail patterns to compare.
    {"name": "fill_tex_wood_brushed", "technique": "fill_textured", "texture": "brushed", "tint": (0.70, 0.45, 0.25), "roughness": 0.75, "metallic": 0.02},
    {"name": "fill_tex_wood_grain",   "technique": "fill_textured", "texture": "grain",   "tint": (0.70, 0.45, 0.25), "roughness": 0.75, "metallic": 0.02},
    {"name": "fill_tex_wood_speckle", "technique": "fill_textured", "texture": "speckle", "tint": (0.70, 0.45, 0.25), "roughness": 0.75, "metallic": 0.02},
]

ANGLE_GRID = [0.0, 45.0]   # inspection grid: CLOSED + clearly-OPEN, for visual audit

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

# ---------------------------------------------------------------------------
# Isaac Sim + Replicator startup (AFTER argparse)
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[ReplCollect] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
_ts("Creating SimulationApp(headless=True) ...")
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

# Match door_static_collection.py DLSS disable: render at native camera
# resolution to avoid the 320×240 DLSS upscale smear.
import carb
_carb_settings = carb.settings.get_settings()
_carb_settings.set("/rtx/post/aa/op", 4)            # RTXAA (native-res DL AA)
_carb_settings.set("/rtx/post/dlss/execMode", 0)
_ts("DLSS disabled, AA set to RTXAA")

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# The Replicator import is what earns this script the "Replicator edition"
# label. Must come AFTER SimulationApp — the extension is loaded on app init.
import omni.replicator.core as rep  # noqa: E402
_ts("omni.replicator.core imported")

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False
    print("[WARN] PIL unavailable — A4 textured technique will be skipped")

def _generate_procedural_textures(scratch_dir):
    """Write grayscale pattern PNGs (brushed / grain / speckle) for the
    textured and fill_textured techniques. Returns {name: abs_path}.

    The patterns are grayscale and bright (mean ~0.8) so OmniPBR's
    diffuse_tint input can colourise them via multiply without going
    too dark. That separates colour from texture pattern — the same
    brushed image can be tinted red, white, or near-black.
    """
    textures = {}
    if not _pil_ok:
        return textures
    os.makedirs(scratch_dir, exist_ok=True)
    h = w = 512
    rng = np.random.default_rng(42)

    def _save(name, gray):
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(scratch_dir, f"{name}.png")
        _PIL.fromarray(rgb).save(p)
        textures[name] = p

    # Brushed — horizontal streaks, narrowed to ~0.84-0.92 range so
    # diffuse_tint keeps a uniform hue (wide range caused multi-color shift).
    base = np.full((h, w), 0.88, dtype=np.float32)
    base += rng.uniform(-0.04, 0.04, (h, 1)).astype(np.float32)
    base += rng.uniform(-0.01, 0.01, (h, w)).astype(np.float32)
    _save("brushed", base)

    # Grain — vertical bands (wood-grain shaped), narrowed to ~0.81-0.95
    xs = np.arange(w, dtype=np.float32)
    band = 0.5 * (np.sin(xs * 0.04) + np.sin(xs * 0.11 + 1.7))   # -1..1
    base = 0.88 + 0.05 * band[None, :]
    base = np.broadcast_to(base, (h, w)).copy()
    base += rng.uniform(-0.02, 0.02, (h, w)).astype(np.float32)
    _save("grain", base)

    # Speckle — flat mid-bright with fine high-frequency noise, ~0.85-0.91
    base = np.full((h, w), 0.88, dtype=np.float32)
    base += rng.uniform(-0.03, 0.03, (h, w)).astype(np.float32)
    _save("speckle", base)

    print(f"[ReplCollect] Grayscale pattern textures → {scratch_dir}")
    return textures

_tex_dir = os.path.join(_project_dir, "data", f"{OUTPUT_NAME}_textures")
_procedural_textures = _generate_procedural_textures(_tex_dir)

# ---------------------------------------------------------------------------
# Candidate camera grid
# ---------------------------------------------------------------------------

def _generate_candidate_positions():
    candidates = []
    idx = 0
    for dist in CAM_DISTANCES:
        for h_off in CAM_HEIGHT_OFFS:
            cam_z = CAMERA_Z_BASE + h_off
            for offset_deg in CAM_ARC_OFFSETS:
                θ = math.radians(offset_deg)
                cam_pos = np.array([
                    DOOR_CENTER_XY[0] + dist * math.cos(θ),
                    DOOR_CENTER_XY[1] + dist * math.sin(θ),
                    cam_z,
                ])
                yaw_deg = offset_deg + 180.0
                cam_ori = rot_utils.euler_angles_to_quats(
                    np.array([0.0, 0.0, yaw_deg]), degrees=True
                )
                d_label = f"d{dist:.1f}".replace(".", "p")
                h_label = f"h{h_off:+.1f}".replace("+", "p").replace("-", "n").replace(".", "p")
                a_label = f"{int(offset_deg):+d}".replace("+", "p").replace("-", "n")
                prim_name = f"/World/ReplCam_{a_label}_{d_label}_{h_label}"
                candidates.append({
                    "prim_path":  prim_name,
                    "pos":        cam_pos,
                    "ori":        cam_ori,
                    "offset_deg": offset_deg,
                    "distance":   dist,
                    "height_off": h_off,
                    "idx":        idx,
                })
                idx += 1
    return candidates

ALL_CANDIDATES = _generate_candidate_positions()
print(f"[ReplCollect] Generated {len(ALL_CANDIDATES)} candidate camera positions")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(building_usd):
    print(f"ERROR: building USD not found: {building_usd}")
    simulation_app.close()
    sys.exit(1)

_ts("Creating World ...")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts(f"Inserting building USD as sublayer: {building_usd}")
stage.GetRootLayer().subLayerPaths.insert(0, building_usd)
_ts("Building USD sublayer inserted")

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)

# 4-slot ceiling RectLight array (identical layout to door_static_collection.py)
_CEILING_OFFSET_Z  = 1.2
_CEILING_SPREAD_XY = 1.2
_CEILING_SIZE      = 0.6
_ceiling_offsets = [
    (-_CEILING_SPREAD_XY, -_CEILING_SPREAD_XY),
    (+_CEILING_SPREAD_XY, -_CEILING_SPREAD_XY),
    (-_CEILING_SPREAD_XY, +_CEILING_SPREAD_XY),
    (+_CEILING_SPREAD_XY, +_CEILING_SPREAD_XY),
]
ceiling_lights = []
for i, (dx, dy) in enumerate(_ceiling_offsets):
    light_path = f"/World/CeilingLight_{i}"
    rl = UsdLux.RectLight.Define(stage, Sdf.Path(light_path))
    rl.CreateIntensityAttr(0.0)
    rl.CreateWidthAttr(_CEILING_SIZE)
    rl.CreateHeightAttr(_CEILING_SIZE)
    prim = rl.GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(
        float(DOOR_CENTER_XY[0]) + dx,
        float(DOOR_CENTER_XY[1]) + dy,
        float(CAMERA_Z_BASE) + _CEILING_OFFSET_Z,
    ))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    ceiling_lights.append(rl)

# Pre-create one material per preset (per-cell loop only flips bindings,
# never mutates shader inputs — OmniPBR caches the first-set color).
def _make_omnipbr_material(preset, tex_path=None):
    name = preset["name"]
    mp = f"/World/Looks/DoorMat_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if tex_path is not None:
        # Texture path — diffuse_texture provides the pattern, and
        # diffuse_tint is multiplied on top to colourise it (see
        # OmniPBRBase.mdl:386 where tinted_diffuse_color =
        # diffuse_reflection_color * diffuse_tint). project_uvw=True
        # generates UVs on-the-fly because Revit-exported meshes have no
        # UV coordinates. world_or_object=False keeps projection in object
        # space so the texture moves with the door mesh.
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(tex_path))
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
        sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(2.0, 2.0))
        tint = preset.get("tint", (1.0, 1.0, 1.0))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*tint))
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*preset["color"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["roughness"]))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(
        float(preset["metallic"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat

def _make_omniglass_material(preset):
    name = preset["name"]
    mp = f"/World/Looks/DoorMat_{name}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*preset["glass_color"]))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(preset["ior"]))
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(
        float(preset["frosting"]))
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(
        bool(preset["thin_walled"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat

_door_preset_materials = {}
for _preset in DOOR_MATERIALS:
    _tech = _preset["technique"]
    if _tech == "baseline" and _preset.get("color") is None:
        continue  # "baseline_original" sentinel — uses Revit-original binding
    if _tech in ("baseline", "fill_glass"):
        _door_preset_materials[_preset["name"]] = _make_omnipbr_material(_preset)
    elif _tech == "glass_variant":
        _door_preset_materials[_preset["name"]] = _make_omniglass_material(_preset)
    elif _tech in ("textured", "fill_textured"):
        _tex_path = _procedural_textures.get(_preset["texture"])
        if _tex_path is None:
            print(f"[WARN] texture '{_preset['texture']}' missing — "
                  f"skipping preset {_preset['name']}")
            continue
        _door_preset_materials[_preset["name"]] = _make_omnipbr_material(
            _preset, tex_path=_tex_path)
print(f"[ReplCollect] Pre-created {len(_door_preset_materials)} door preset materials")

# ---------------------------------------------------------------------------
# Load scene + physics raycast occlusion filter
# ---------------------------------------------------------------------------

_ts("world.reset() — loading building USD")
world.reset()
_ts("world.reset() done")
simulation_app.update()
for _ in range(5):
    world.step(render=True)
_ts("Pre-raycast warmup done")

def _check_line_of_sight(candidates):
    if not DO_OCCLUSION:
        return candidates
    try:
        from omni.physx import get_physx_scene_query_interface
        physx_query = get_physx_scene_query_interface()
    except Exception as e:
        print(f"[WARN] PhysX query unavailable ({e}) — skipping occlusion check")
        return candidates

    valid = []
    blocked = 0
    for cand in candidates:
        origin = cand["pos"]
        target = np.array([DOOR_CENTER_XY[0], DOOR_CENTER_XY[1], origin[2]])
        direction = target - origin
        max_dist = float(np.linalg.norm(direction))
        if max_dist < 0.01:
            valid.append(cand)
            continue
        direction = direction / max_dist
        hit = physx_query.raycast_closest(
            tuple(float(v) for v in origin),
            tuple(float(v) for v in direction),
            max_dist,
        )
        if hit["hit"]:
            hit_path = hit.get("rigidBody", "") or ""
            if DOOR_PATH in hit_path:
                valid.append(cand)
            else:
                blocked += 1
                print(f"  [Occluded] arc={cand['offset_deg']:+.0f}° "
                      f"dist={cand['distance']:.1f}m "
                      f"height={cand['height_off']:+.1f}m → {hit_path}")
        else:
            valid.append(cand)
    print(f"[ReplCollect] Occlusion check: {len(valid)}/{len(candidates)} clear, "
          f"{blocked} blocked")
    return valid

CAMERA_CONFIGS = _check_line_of_sight(ALL_CANDIDATES)
if not CAMERA_CONFIGS:
    print("ERROR: all camera positions occluded")
    simulation_app.close()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Cameras + Replicator render products + rgb annotators
#
# This is the Replicator-path section. Cameras are created via the Isaac
# Sim high-level wrapper so set_world_pose() still works for camera jitter,
# but frame readout goes through a per-camera rgb annotator bound to a
# render_product — not Camera.get_rgba().
# ---------------------------------------------------------------------------

cameras = []           # isaacsim.sensors.camera.Camera (for jitter + metadata)
render_products = []   # rep.create.render_product handles
rgb_annotators = []    # rep.AnnotatorRegistry.get_annotator("rgb") per cam

for cfg in CAMERA_CONFIGS:
    cam = Camera(
        prim_path=cfg["prim_path"],
        position=cfg["pos"],
        frequency=30,
        resolution=CAM_RESOLUTION,
        orientation=cfg["ori"],
    )
    cameras.append(cam)

simulation_app.update()

for cam, cfg in zip(cameras, CAMERA_CONFIGS):
    cam.initialize()
    usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(cfg["prim_path"]))
    usd_cam.GetFocalLengthAttr().Set(5.0)
    usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

    # Bind this camera into the Replicator render graph. One render product
    # per camera keeps the readout loop trivial — for N>>10 cameras a single
    # shared annotator attached to the list may be cheaper; revisit if the
    # arc grid grows past ~30 cameras.
    rp = rep.create.render_product(cfg["prim_path"], CAM_RESOLUTION)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    render_products.append(rp)
    rgb_annotators.append(ann)

print(f"[ReplCollect] {len(cameras)} cameras + render_products + rgb annotators ready")

_ts("Renderer warmup (30 steps) ...")
for _ in range(30):
    world.step(render=True)
_ts("Renderer warmup done")

# ---------------------------------------------------------------------------
# Door mesh discovery + 4×4 matrix rotation (KEPT from door_static_collection)
# ---------------------------------------------------------------------------

door_prim = stage.GetPrimAtPath(DOOR_PATH)
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)

if door_prim.IsInstance() or door_prim.IsInstanceable():
    print(f"[ReplCollect] Door is instanced — SetInstanceable(False) to expose subtree")
    door_prim.SetInstanceable(False)
    simulation_app.update()

def _is_glass_mesh(mesh_prim) -> bool:
    n = mesh_prim.GetPath().name.lower()
    return (n.startswith("glass_") or "_gl_" in n or "glazing" in n or "pane_" in n)

_mesh_ops = []
for child in door_prim.GetChildren():
    if not child.IsA(UsdGeom.Mesh):
        continue
    xformable = UsdGeom.Xformable(child)
    op = None
    for xop in xformable.GetOrderedXformOps():
        if xop.GetOpName() == "xformOp:transform":
            op = xop
            break
    if op is None:
        print(f"[WARN] skipping {child.GetPath()} — no xformOp:transform")
        continue
    mat = op.Get()
    _mesh_ops.append(({
        "path": str(child.GetPath()),
        "ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2],
    }, op))

if not _mesh_ops:
    print(f"ERROR: no child meshes with xformOp:transform under {DOOR_PATH}")
    simulation_app.close()
    sys.exit(1)
print(f"[ReplCollect] {len(_mesh_ops)} door child meshes discovered")

# Cache original material bindings separately for frame vs glass meshes.
# A1 (fill_glass) and A3 (glass_variant) both mutate the glass side, so
# we need to be able to restore either half independently between cells.
_original_frame_bindings = []
_original_glass_bindings = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh):
        continue
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    targets = list(rel.GetTargets()) if rel else []
    if _is_glass_mesh(p):
        _original_glass_bindings.append((p, targets))
    else:
        _original_frame_bindings.append((p, targets))
print(f"[ReplCollect] Cached bindings: {len(_original_frame_bindings)} frame, "
      f"{len(_original_glass_bindings)} glass")

def _restore_bindings(cache):
    for mesh_prim, orig_targets in cache:
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        binding_api.UnbindAllBindings()
        if not orig_targets:
            continue
        rel = binding_api.GetDirectBindingRel() or binding_api.CreateDirectBindingRel()
        for tgt in orig_targets:
            rel.AddTarget(tgt)

def _restore_original_door_materials(include_frame=True, include_glass=True):
    if include_frame:
        _restore_bindings(_original_frame_bindings)
    if include_glass:
        _restore_bindings(_original_glass_bindings)

def _door_matrix(ox_cm, oy_cm, oz_cm, angle_deg):
    θ = math.radians(angle_deg)
    c, s = math.cos(θ), math.sin(θ)
    dx = ox_cm - HINGE_X_CM
    dy = oy_cm - HINGE_Y_CM
    tx = c * dx - s * dy + HINGE_X_CM
    ty = s * dx + c * dy + HINGE_Y_CM
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        tx, ty, oz_cm, 1,
    )

def set_door_angle(angle_deg):
    for m, op in _mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg))

# ---------------------------------------------------------------------------
# Lighting + material cell helpers
# ---------------------------------------------------------------------------

def _color_temp_to_rgb(kelvin: float) -> Gf.Vec3f:
    temp = kelvin / 100.0
    if temp <= 66:
        r = 255
    else:
        r = max(0, min(255, 329.698727446 * ((temp - 60) ** -0.1332047592)))
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = max(0, min(255, g))
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = max(0, min(255, 138.5177312231 * math.log(temp - 10) - 305.0447927307))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)

def apply_lighting_condition(cond: dict):
    jitter_i = lambda v: v * random.uniform(0.90, 1.10)
    jitter_t = lambda v: v + random.uniform(-200, 200)
    dome_int  = jitter_i(cond["dome_int"])
    dome_temp = jitter_t(cond["dome_temp"])
    dome.GetIntensityAttr().Set(dome_int)
    dome.GetColorAttr().Set(_color_temp_to_rgb(dome_temp))
    ceiling_int  = jitter_i(cond["ceiling_int"])
    ceiling_temp = jitter_t(cond["ceiling_temp"])
    ceiling_rgb  = _color_temp_to_rgb(ceiling_temp)
    for slot, rl in enumerate(ceiling_lights):
        on = cond["ceiling_pattern"][slot]
        rl.GetIntensityAttr().Set(ceiling_int if on else 0.0)
        rl.GetColorAttr().Set(ceiling_rgb)

def _bind_door_meshes(mat, include_frame: bool, include_glass: bool):
    strong = UsdShade.Tokens.strongerThanDescendants
    for p in Usd.PrimRange(door_prim):
        if not p.IsA(UsdGeom.Mesh):
            continue
        is_glass = _is_glass_mesh(p)
        if is_glass and not include_glass:
            continue
        if (not is_glass) and not include_frame:
            continue
        UsdShade.MaterialBindingAPI.Apply(p)
        UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=strong)

def apply_door_material(spec: dict):
    """Technique dispatcher.

    Always restores both frame and glass to their Revit-original bindings
    first, so no prior cell bleeds into this one. Then binds the preset's
    material to whichever side(s) the technique targets:

        baseline       → frame only (glass left untouched)
        fill_glass     → frame + glass (curtain-wall → solid panel)
        glass_variant  → glass only (OmniGlass swap, frame untouched)
        textured       → frame only (procedural OmniPBR texture)
        fill_textured  → frame + glass (A1 + A4 combined: solid-panel
                         appearance with a tinted grayscale pattern)
    """
    _restore_original_door_materials(include_frame=True, include_glass=True)

    tech = spec["technique"]
    if tech == "baseline":
        if spec.get("color") is None:
            return  # "baseline_original" — nothing to bind
        mat = _door_preset_materials[spec["name"]]
        _bind_door_meshes(mat, include_frame=True, include_glass=False)
    elif tech == "fill_glass":
        mat = _door_preset_materials[spec["name"]]
        _bind_door_meshes(mat, include_frame=True, include_glass=True)
    elif tech == "glass_variant":
        mat = _door_preset_materials[spec["name"]]
        _bind_door_meshes(mat, include_frame=False, include_glass=True)
    elif tech == "textured":
        mat = _door_preset_materials.get(spec["name"])
        if mat is None:
            return  # texture unavailable (PIL missing) — stay at original
        _bind_door_meshes(mat, include_frame=True, include_glass=False)
    elif tech == "fill_textured":
        mat = _door_preset_materials.get(spec["name"])
        if mat is None:
            return
        _bind_door_meshes(mat, include_frame=True, include_glass=True)

def apply_camera_jitter():
    for cam_obj, cfg in zip(cameras, CAMERA_CONFIGS):
        base_pos = cfg["pos"]
        jittered_pos = np.array([
            base_pos[0] + random.uniform(-0.15, 0.15),
            base_pos[1] + random.uniform(-0.15, 0.15),
            base_pos[2] + random.uniform(-0.10, 0.10),
        ])
        pitch = random.uniform(-2.0, 2.0)
        roll  = random.uniform(-1.5, 1.5)
        base_yaw = cfg["offset_deg"] + 180.0
        jittered_ori = rot_utils.euler_angles_to_quats(
            np.array([roll, pitch, base_yaw]), degrees=True
        )
        try:
            cam_obj.set_world_pose(position=jittered_pos, orientation=jittered_ori)
        except Exception:
            cam_prim = stage.GetPrimAtPath(cfg["prim_path"])
            UsdGeom.Xformable(cam_prim).MakeMatrixXform().Set(
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(*jittered_pos.tolist()))
            )

# ---------------------------------------------------------------------------
# Collection buffers + capture helper (Replicator annotator readout)
# ---------------------------------------------------------------------------

_jpeg_dir = None
if DUMP_JPEGS:
    _jpeg_dir = os.path.join(_project_dir, "data", f"{OUTPUT_NAME}_jpegs")
    os.makedirs(_jpeg_dir, exist_ok=True)
    print(f"[ReplCollect] JPEG dump → {_jpeg_dir}")

all_frames, all_labels, all_angles = [], [], []
all_episodes, all_cam_offsets = [], []
all_light_idx, all_material_idx = [], []

def _capture_frame(episode_idx, angle_deg, light_idx, material_idx,
                   light_name=None, material_name=None, frame_idx=0):
    """Step once, then read every rgb annotator. This is the Replicator
    readout path — the per-camera rep.AnnotatorRegistry("rgb") returns an
    HxWx4 uint8 array after the renderer settles."""
    world.step(render=True)
    captured = 0
    for ann, cfg in zip(rgb_annotators, CAMERA_CONFIGS):
        data = ann.get_data()
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            print(f"[WARN] unexpected rgb shape {arr.shape} — skipping")
            continue
        frame = arr[:, :, :3].astype(np.uint8)
        all_frames.append(frame)
        all_labels.append(angle_deg / 90.0)
        all_angles.append(angle_deg)
        all_episodes.append(episode_idx)
        all_cam_offsets.append(int(cfg["offset_deg"]))
        all_light_idx.append(light_idx)
        all_material_idx.append(material_idx)
        captured += 1

        if DUMP_JPEGS and _pil_ok and _jpeg_dir is not None:
            cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
            a_lab = f"a{int(cfg['offset_deg']):+d}".replace("+", "p").replace("-", "n")
            fname = (f"ep{episode_idx:04d}_{cls}_ang{angle_deg:05.1f}"
                     f"_L-{light_name or 'NA'}_M-{material_name or 'NA'}"
                     f"_cam{a_lab}_f{frame_idx}.jpg")
            try:
                _PIL.fromarray(frame).save(os.path.join(_jpeg_dir, fname),
                                           "JPEG", quality=88)
            except Exception as e:
                print(f"[WARN] JPEG save failed {fname}: {e}")
    return captured

# ---------------------------------------------------------------------------
# Grid loop: lighting × material × angle
# ---------------------------------------------------------------------------

if BASELINE_ONLY:
    lightings = [LIGHTING_CONDITIONS[0]]
    materials = [DOOR_MATERIALS[0]]      # "baseline_original"
else:
    # Technique-inspection mode: single neutral lighting so the ONLY
    # visual difference between frames is the material technique.
    # `overcast_flat` gives an even white-balanced render that makes
    # colour/roughness/glass differences easy to compare by eye.
    _neutral = [c for c in LIGHTING_CONDITIONS if c["name"] == "overcast_flat"]
    lightings = _neutral or [LIGHTING_CONDITIONS[0]]
    materials = DOOR_MATERIALS if MATERIAL_VAR else [DOOR_MATERIALS[0]]
angles = ANGLE_GRID

n_cams = len(cameras)
total_cells = len(lightings) * len(materials) * len(angles)
print(f"[ReplCollect] Grid: {len(lightings)} lightings × {len(materials)} materials "
      f"× {len(angles)} angles = {total_cells} cells")
print(f"[ReplCollect] Cameras: {n_cams}  frames/cell: {GRID_FRAMES}  "
      f"jitter: {'ON' if CAMERA_JITTER else 'OFF'}")

# Prime the RTX temporal accumulator with the first lighting preset.
# This must happen after apply_lighting_condition() is defined (it's a
# top-level function, so the call can't live up in the warmup block).
# Without this, the accumulator builds history against the default 50k
# dome, then has to re-converge when the first cell changes lighting —
# which is what makes the first frames darker than the rest.
_first_lighting = lightings[0]
apply_lighting_condition(_first_lighting)
_ts("Lighting settle (20 steps with first preset active) ...")
for _ in range(20):
    world.step(render=True)
_ts("Lighting settle done — starting collection loop")
episode = 0
for lighting in lightings:
    g_light_idx = LIGHTING_CONDITIONS.index(lighting)
    for material in materials:
        g_mat_idx = DOOR_MATERIALS.index(material)
        for angle_deg in angles:
            apply_lighting_condition(lighting)
            apply_door_material(material)
            set_door_angle(angle_deg)
            if CAMERA_JITTER:
                apply_camera_jitter()
            for _ in range(WARMUP_STEPS):
                world.step(render=True)

            ep_captured = 0
            for frame_i in range(GRID_FRAMES):
                ep_captured += _capture_frame(
                    episode, angle_deg, g_light_idx, g_mat_idx,
                    light_name=lighting["name"],
                    material_name=material["name"],
                    frame_idx=frame_i,
                )

            cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
            print(f"[ReplCollect] Ep {episode+1:4d}/{total_cells}  "
                  f"[{cls:6s} {angle_deg:5.1f}°]  "
                  f"light={lighting['name']:<20s} mat={material['name']:<20s}  "
                  f"frames={ep_captured}")
            episode += 1

# ---------------------------------------------------------------------------
# Save — byte-compatible NPZ schema with door_static_collection.py
# ---------------------------------------------------------------------------

data_dir = os.path.join(_project_dir, "data")
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, f"{OUTPUT_NAME}.npz")

print(f"\n[ReplCollect] Saving {len(all_frames)} frames → {out_path}")
np.savez_compressed(
    out_path,
    frames         = np.array(all_frames,       dtype=np.uint8),
    labels         = np.array(all_labels,       dtype=np.float32),
    angles         = np.array(all_angles,       dtype=np.float32),
    episode        = np.array(all_episodes,     dtype=np.int32),
    cam_offsets    = np.array(all_cam_offsets,  dtype=np.int32),
    light_idx      = np.array(all_light_idx,    dtype=np.int32),
    material_idx   = np.array(all_material_idx, dtype=np.int32),
    light_names    = np.array([c["name"] for c in LIGHTING_CONDITIONS]),
    material_names = np.array([m["name"] for m in DOOR_MATERIALS]),
)
size_mb = os.path.getsize(out_path) // 1024 // 1024
angle_arr = np.array(all_angles, dtype=np.float32)
print(f"[ReplCollect] Saved ({size_mb} MB) — "
      f"CLOSED: {(angle_arr <= CLOSED_MAX_DEG).sum()}  "
      f"OPEN: {(angle_arr > CLOSED_MAX_DEG).sum()}  "
      f"cameras: {n_cams}")

world.stop()
simulation_app.close()
