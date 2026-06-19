#!/usr/bin/env python3
"""
C1 — Wall material variation test on Doors_10649710 (RS___Type_FG_Door).

Tests C1 (surrounding wall material swap) on a solid glass-panel door
embedded in a real wall — unlike curtain-wall doors, this family has
actual Walls category geometry (drywall/CMU) surrounding the frame.

Door: Doors_10649710, floor 3, Type_FG_Door (instanced — B2 path)
    Uses SetInstanceable(False) + local-space rotation.

Wall discovery:
    Walks /Geometry/Walls/ reading leaf mesh xformOp:transform row-3
    (world-space translation baked in by Revit, same as door meshes)
    to find wall prims within WALL_RADIUS_CM of the door on floor 3.

Grid:
    6 material presets × 2 angles (0° CLOSED, 45° OPEN) × 1 lighting = 12 frames
    Door stays at Revit-original. Only surrounding wall meshes are swapped.

Output:
    data/c1_test_jpegs_Doors_10649710/c1_<material>_<CLOSED|OPEN>.jpg

Run:
    sbatch submit_isaac_job.sh simulations/robot/c1_wall_material_test.py
"""

import os
import sys
import math
import random
import functools
import numpy as np

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOOR_PATH    = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649710"
)
# Instanced door — B2 rotation (local-space). No world-space hinge available.
# Hinge estimated as min-edge of wide axis in local space (approach_axis=X
# → wide axis=Y in local → hinge_ly = -(width/2) cm = -50.8 cm).
IS_INSTANCED   = True
HINGE_LX_CM    =   0.0     # local x hinge offset
HINGE_LY_CM    = -50.8     # local y hinge offset (min Y edge)
DOOR_CENTER_XY = np.array([-12.30, 75.47])   # metres
CAMERA_Z_BASE  = 19.44                         # metres (floor 3 level + 1.2 m)
CAM_ARC        = 0.0                           # head-on
CAM_DIST       = 2.0                           # metres

# Wall search — all in USD cm units (leaf mesh xformOp world translation)
WALL_CENTRE_CM = (-1229.53, 7547.26)           # door parent matrix translation
WALL_FLOOR_Z   = (1750.0, 2150.0)             # floor 3: ~18–21 m in cm
WALL_RADIUS_CM = 500.0                         # 5 m search radius

CAM_RESOLUTION = (1024, 768)
ANGLE_GRID     = [0.0, 45.0]
WARMUP_STEPS   = 10
STARTUP_WARMUP = 30

# Materials applied to both door + mullions simultaneously
MATERIALS = [
    {"name": "original",         "color": None},
    {"name": "brushed_aluminum", "color": (0.75, 0.75, 0.78), "roughness": 0.35, "metallic": 0.85},
    {"name": "painted_white",    "color": (0.90, 0.90, 0.88), "roughness": 0.70, "metallic": 0.05},
    {"name": "dark_anodized",    "color": (0.12, 0.12, 0.15), "roughness": 0.25, "metallic": 0.80},
    {"name": "painted_red",      "color": (0.65, 0.10, 0.08), "roughness": 0.60, "metallic": 0.10},
    {"name": "wood_veneer",      "color": (0.55, 0.35, 0.20), "roughness": 0.75, "metallic": 0.02},
]

LIGHTING = {"name": "overcast_flat", "dome_int": 25_000, "dome_temp": 5000,
            "ceiling_pattern": (1, 1, 1, 1), "ceiling_int": 220_000, "ceiling_temp": 5000}

BUILDING_ROOT = ("/World/Building"
                 "/_22_129___MSOE_Engineering_Building___A___2_25_2026")

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR  = os.path.join(_project_dir, "data", "c1_test_jpegs_Doors_10649710")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[C1Test] [+{_time.time()-_t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Gf, Sdf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
_ts("Loading building USD ...")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(LIGHTING["dome_int"])

_CEILING_OFFSETS = [(-1.2,-1.2),(+1.2,-1.2),(-1.2,+1.2),(+1.2,+1.2)]
ceiling_lights = []
for i,(dx,dy) in enumerate(_CEILING_OFFSETS):
    rl = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/CeilingLight_{i}"))
    rl.CreateIntensityAttr(0.0)
    rl.CreateWidthAttr(0.6)
    rl.CreateHeightAttr(0.6)
    xf = UsdGeom.Xformable(rl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(
        float(DOOR_CENTER_XY[0])+dx, float(DOOR_CENTER_XY[1])+dy,
        float(CAMERA_Z_BASE)+1.2))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0,0.0,0.0))
    ceiling_lights.append(rl)

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

θ = math.radians(CAM_ARC)
cam_pos = np.array([
    DOOR_CENTER_XY[0] + CAM_DIST * math.cos(θ),
    DOOR_CENTER_XY[1] + CAM_DIST * math.sin(θ),
    CAMERA_Z_BASE,
])
cam_ori = rot_utils.euler_angles_to_quats(
    np.array([0.0, 0.0, CAM_ARC + 180.0]), degrees=True)

cam = Camera(prim_path="/World/C1Cam", position=cam_pos,
             frequency=30, resolution=CAM_RESOLUTION, orientation=cam_ori)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/C1Cam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/C1Cam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

# ---------------------------------------------------------------------------
# Lighting apply
# ---------------------------------------------------------------------------

def _kelvin_to_rgb(k):
    t = k / 100.0
    r = 255 if t <= 66 else max(0, min(255, 329.698727446 * ((t-60)**-0.1332047592)))
    g = (99.4708025861*math.log(t)-161.1195681661) if t<=66 else max(0,min(255,288.1221695283*((t-60)**-0.0755148492)))
    g = max(0, min(255, g))
    b = 255 if t>=66 else (0 if t<=19 else max(0,min(255,138.5177312231*math.log(t-10)-305.0447927307)))
    return Gf.Vec3f(r/255., g/255., b/255.)

def apply_lighting():
    ji = lambda v: v * random.uniform(0.90, 1.10)
    jt = lambda v: v + random.uniform(-200, 200)
    dome.GetIntensityAttr().Set(ji(LIGHTING["dome_int"]))
    dome.GetColorAttr().Set(_kelvin_to_rgb(jt(LIGHTING["dome_temp"])))
    ci = ji(LIGHTING["ceiling_int"]); ct = jt(LIGHTING["ceiling_temp"])
    for slot, rl in enumerate(ceiling_lights):
        on = LIGHTING["ceiling_pattern"][slot]
        rl.GetIntensityAttr().Set(ci if on else 0.0)
        rl.GetColorAttr().Set(_kelvin_to_rgb(ct))

# ---------------------------------------------------------------------------
# Door mesh discovery + rotation
# ---------------------------------------------------------------------------

door_prim = stage.GetPrimAtPath(DOOR_PATH)
if not door_prim.IsValid():
    print(f"ERROR: door prim not found: {DOOR_PATH}")
    simulation_app.close(); sys.exit(1)

# Instanced door — concretize subtree before walking
if door_prim.IsInstance() or door_prim.IsInstanceable():
    print("[C1Test] SetInstanceable(False) ...")
    door_prim.SetInstanceable(False)
    simulation_app.update()

def _is_glass(p):
    n = p.GetPath().name.lower()
    return n.startswith("glass_") or "_gl_" in n or "glazing" in n or "pane_" in n

# B2 path: walk full subtree for xformOp:transform meshes (local-space coords)
_mesh_ops = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh):
        continue
    xf = UsdGeom.Xformable(p)
    op = next((o for o in xf.GetOrderedXformOps()
               if o.GetOpName() == "xformOp:transform"), None)
    if op is None:
        continue
    m = op.Get()
    if m is None:
        continue
    _mesh_ops.append(({"lx": m[3][0], "ly": m[3][1], "lz": m[3][2]}, op))

print(f"[C1Test] {len(_mesh_ops)} door meshes (local-space B2 rotation)")

def set_door_angle(angle_deg):
    """Rotate in LOCAL space around (HINGE_LX_CM, HINGE_LY_CM)."""
    θ = math.radians(angle_deg)
    c, s = math.cos(θ), math.sin(θ)
    for m, op in _mesh_ops:
        dx, dy = m["lx"] - HINGE_LX_CM, m["ly"] - HINGE_LY_CM
        op.Set(Gf.Matrix4d(c,s,0,0,-s,c,0,0,0,0,1,0,
                           c*dx-s*dy+HINGE_LX_CM,
                           s*dx+c*dy+HINGE_LY_CM,
                           m["lz"], 1))

# ---------------------------------------------------------------------------
# Mullion discovery (xformOp translation, not BBoxCache)
# ---------------------------------------------------------------------------

def _xform_translation(prim):
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:transform":
            m = op.Get()
            if m: return m[3][0], m[3][1], m[3][2]
        if op.GetOpName() == "xformOp:translate":
            v = op.Get()
            if v: return v[0], v[1], v[2]
    return None

# ---------------------------------------------------------------------------
# Wall mesh discovery — leaf mesh xformOp:transform row-3 gives world pos
# (Revit bakes world-space translations into each leaf mesh matrix, same
# as for door meshes where this approach confirmed correct positions).
# ---------------------------------------------------------------------------

walls_root = stage.GetPrimAtPath(f"{BUILDING_ROOT}/Geometry/Walls")
wall_meshes = []

if walls_root.IsValid():
    for prim in Usd.PrimRange(walls_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        t = _xform_translation(prim)
        if t is None:
            continue
        tx, ty, tz = t
        if not (WALL_FLOOR_Z[0] <= tz <= WALL_FLOOR_Z[1]):
            continue
        d = math.sqrt((tx-WALL_CENTRE_CM[0])**2 + (ty-WALL_CENTRE_CM[1])**2)
        if d > WALL_RADIUS_CM:
            continue
        wall_meshes.append(prim)
else:
    print("[C1Test] WARNING: /Geometry/Walls not valid")

print(f"[C1Test] {len(wall_meshes)} wall meshes within {WALL_RADIUS_CM} cm of door")

# ---------------------------------------------------------------------------
# Pre-create OmniPBR materials (door + mullion share the same preset dict)
# ---------------------------------------------------------------------------

_preset_mats = {}   # name → UsdShade.Material
for p in MATERIALS:
    if p["color"] is None:
        continue
    mp = f"/World/Looks/C1Mat_{p['name']}"
    mat = UsdShade.Material.Define(stage, Sdf.Path(mp))
    sh  = UsdShade.Shader.Define(stage, Sdf.Path(mp + "/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["color"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float  ).Set(float(p["roughness"]))
    sh.CreateInput("metallic_constant",             Sdf.ValueTypeNames.Float  ).Set(float(p["metallic"]))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    _preset_mats[p["name"]] = mat

print(f"[C1Test] Pre-created {len(_preset_mats)} materials")

# Cache original bindings for door (frame only) and mullions
_orig_door_bindings = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh) or _is_glass(p):
        continue
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    _orig_door_bindings.append((p, list(rel.GetTargets()) if rel else []))

_orig_wall_bindings = []
for p in wall_meshes:
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    _orig_wall_bindings.append((p, list(rel.GetTargets()) if rel else []))

def _restore(cache):
    for mesh_prim, targets in cache:
        api = UsdShade.MaterialBindingAPI(mesh_prim)
        api.UnbindAllBindings()
        if targets:
            rel = api.GetDirectBindingRel() or api.CreateDirectBindingRel()
            for t in targets: rel.AddTarget(t)

def _bind_context(mat):
    """Option B: swap surrounding wall — door stays at Revit-original."""
    strong = UsdShade.Tokens.strongerThanDescendants
    for mesh_prim in wall_meshes:
        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat, bindingStrength=strong)

# ---------------------------------------------------------------------------
# Lighting settle + warmup
# ---------------------------------------------------------------------------

apply_lighting()
_ts(f"Startup warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)

apply_lighting()
_ts("Lighting settle (20 steps) ...")
for _ in range(20):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Grid loop
# ---------------------------------------------------------------------------

def capture(mat_name, angle_deg):
    world.step(render=True)
    data = ann.get_data()
    if data is None: return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3: return False
    rgb = arr[:,:,:3].astype(np.uint8)
    cls = "CLOSED" if angle_deg <= 5.0 else "OPEN"
    fname = f"c1_{mat_name}_{cls}.jpg"
    if _pil_ok:
        _PIL.fromarray(rgb).save(os.path.join(OUT_DIR, fname), "JPEG", quality=92)
    else:
        np.save(os.path.join(OUT_DIR, fname.replace(".jpg",".npy")), rgb)
    return True

total = len(MATERIALS) * len(ANGLE_GRID)
cell  = 0
_ts("Starting grid loop ...")

for mat_spec in MATERIALS:
    mat_name = mat_spec["name"]

    # Door always stays at Revit-original. Only surrounding walls are swapped.
    _restore(_orig_wall_bindings)

    if mat_spec["color"] is not None:
        _bind_context(_preset_mats[mat_name])

    for angle_deg in ANGLE_GRID:
        set_door_angle(angle_deg)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)

        ok  = capture(mat_name, angle_deg)
        cls = "CLOSED" if angle_deg <= 5.0 else "OPEN"
        cell += 1
        print(f"[C1Test] {cell:2d}/{total}  mat={mat_name:<20}  "
              f"{cls:6s}  {'saved' if ok else 'FAILED'}")

    # Reset door to closed
    set_door_angle(0.0)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_ts("Done")
print(f"[C1Test] JPEGs → {OUT_DIR}/")
world.stop()
simulation_app.close()
