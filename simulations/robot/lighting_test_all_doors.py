#!/usr/bin/env python3
"""
Lighting condition verification across all 10 selected training doors.

Renders each door at 0° (CLOSED) and 45° (OPEN) under all 6 lighting
presets from LIGHTING_AND_MATERIALS.md so the full lighting × door
combination can be visually signed off before production data collection.

10 doors × 6 lighting presets × 2 angles = 120 JPEGs.

Door config is taken verbatim from door_hinge_test.py (same hinge, camera,
angle_sign, mesh_count fields). Ceiling lights are created with mutable
translate ops and repositioned to each door's centre per-door.

Output
------
  data/lighting_test/<family>_<door_id>_floor<N>_<lighting>_<CLOSED|OPEN>.jpg
  data/lighting_test_summary.json

Run
---
  sbatch submit_isaac_job.sh simulations/robot/lighting_test_all_doors.py
"""

import os
import sys
import math
import json
import random
import functools
import numpy as np

print = functools.partial(print, flush=True)   # noqa: A001

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANGLE_GRID     = [0.0]           # CLOSED only — just need to see lighting, not door state
CAM_RESOLUTION = (1024, 768)
WARMUP_STEPS   = 15             # per lighting preset settle (bumped from 10 — marginal for large jumps)
DOOR_WARMUP    = 20             # steps after repositioning camera + lights (first lighting active)
STARTUP_WARMUP = 30
CLOSED_MAX_DEG = 5.0
CAM_DIST_M     = 2.0            # metres — matches candidate + hinge test scripts

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
OUT_DIR      = os.path.join(_project_dir, "data", "lighting_test")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Lighting conditions — from LIGHTING_AND_MATERIALS.md
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Door registry — identical to door_hinge_test.py
# door_center_m added here so ceiling lights can be anchored correctly.
# Derived as cam_pos + 2 m back along camera look direction.
# ---------------------------------------------------------------------------

_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors"
)

DOORS = [
    # ── TRAIN ────────────────────────────────────────────────────────────────
    {   # CW_F_NEW floor 4 — replaces Doors_13607620 (floor 2) per user review
        "id":             "Doors_13606466",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466",
        "family_short":   "CW_F_NEW",
        "floor":          4,
        "role":           "train",
        "mesh_count":     1,
        "hinge_x_cm":     -4836.01,
        "hinge_y_cm":     6170.58,    # translation_y + extent_min_y
        "cam_pos":        [-46.360, 62.163, 23.704],
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-48.360, 62.163, 23.704],  # cam_pos − 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {   # Type_FG promoted to train (was holdout) so Type_FG is represented
        # in training. Inward swing (angle_sign=-1 + parent matrix; see
        # hinge test job 243071).
        "id":             "Doors_10649156",
        "prim_path":      f"{_BASE}/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156",
        "family_short":   "Type_FG",
        "floor":          3,
        "role":           "train",
        "mesh_count":     0,
        "hinge_x_cm":     +50.80,
        "hinge_y_cm":     0.0,
        "cam_pos":        [-14.295, 80.200, 19.437],
        "cam_yaw_deg":    0.0,
        "door_center_m":  [-12.295, 80.200, 19.437],  # cam_pos + 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     -1,
    },
    {
        "id":             "Doors_8286057",
        "prim_path":      f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057",
        "family_short":   "Type_F",
        "floor":          2,
        "role":           "train",
        "mesh_count":     0,
        "hinge_x_cm":     +50.80,
        "hinge_y_cm":     0.0,
        "cam_pos":        [17.443, 79.341, 14.764],
        "cam_yaw_deg":    270.0,
        "door_center_m":  [17.443, 77.341, 14.764],   # cam_pos − 2 m in +Y
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {
        "id":             "Doors_3135827",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827",
        "family_short":   "CW_FG_New",
        "floor":          3,
        "role":           "train",
        "mesh_count":     2,
        "hinge_x_cm":     1724.33,
        "hinge_y_cm":     5878.48,
        "cam_pos":        [17.701, 56.807, 19.437],
        "cam_yaw_deg":    90.0,
        "door_center_m":  [17.701, 58.807, 19.437],   # cam_pos + 2 m in +Y
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {   # Type_F floor 5 42" wide — OUTWARD-SWINGING. Promoted to train so the
        # outward-swing visual pattern is represented in the training
        # distribution (previously the only outward door was holdout-only,
        # which meant training could never learn the outward-swing pattern).
        "id":             "Doors_7171063",
        "prim_path":      f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063",
        "family_short":   "Type_F",
        "floor":          5,
        "role":           "train",
        "mesh_count":     0,
        "hinge_x_cm":     +58.42,
        "hinge_y_cm":     0.0,
        "cam_pos":        [-49.316, 49.892, 27.972],
        "cam_yaw_deg":    0.0,
        "door_center_m":  [-47.316, 49.892, 27.972],  # cam_pos + 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     -1,  # outward swing (toward camera / into corridor)
    },
    {
        "id":             "Doors_12590968",
        "prim_path":      f"{_BASE}/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968",
        "family_short":   "Type_N",
        "floor":          3,
        "role":           "train",
        "mesh_count":     0,
        "hinge_x_cm":     +50.80,
        "hinge_y_cm":     0.0,
        "cam_pos":        [17.878, 50.941, 19.437],
        "cam_yaw_deg":    90.0,
        "door_center_m":  [17.878, 52.941, 19.437],   # cam_pos + 2 m in +Y
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {   # CW_FG_New floor 2 sideP — new addition
        "id":             "Doors_12302795",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795",
        "family_short":   "CW_FG_New",
        "floor":          2,
        "role":           "train",
        "mesh_count":     2,
        "hinge_x_cm":     -5925.83,
        "hinge_y_cm":     5774.34,    # translation_y + extent_min_y
        "cam_pos":        [-57.258, 58.251, 14.561],
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-59.258, 58.251, 14.561],  # cam_pos − 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {   # CW_F_NEW floor 3 sideP — new addition
        "id":             "Doors_13606056",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056",
        "family_short":   "CW_F_NEW",
        "floor":          3,
        "role":           "train",
        "mesh_count":     1,
        "hinge_x_cm":     -4747.75,
        "hinge_y_cm":     6958.30,    # translation_y + extent_min_y
        "cam_pos":        [-45.477, 70.040, 19.437],
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-47.477, 70.040, 19.437],  # cam_pos − 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    # ── HOLDOUT ──────────────────────────────────────────────────────────────
    {   # CW_FG_New demoted from train → holdout. Tests clean instance-level
        # generalization within a seen family (3 → 2 CW_FG_New trainers now).
        "id":             "Doors_13182085",
        "prim_path":      f"{_BASE}/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085",
        "family_short":   "CW_FG_New",
        "floor":          4,
        "role":           "holdout",
        "mesh_count":     2,
        "hinge_x_cm":     -4142.56,
        "hinge_y_cm":     8214.32,
        "cam_pos":        [-39.426, 82.600, 23.704],
        "cam_yaw_deg":    180.0,
        "door_center_m":  [-41.426, 82.600, 23.704],  # cam_pos − 2 m in +X
        "leaf_mesh_names": None,
        "angle_sign":     1,
    },
    {   # Type_F floor 5 sideP — demoted from train → holdout. Clean
        # instance-level generalization probe within the seen Type_F family
        # (inward swing matches the other 2 remaining Type_F training doors).
        "id":             "Doors_1915613",
        "prim_path":      f"{_BASE}/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613",
        "family_short":   "Type_F",
        "floor":          5,
        "role":           "holdout",
        "mesh_count":     0,
        "hinge_x_cm":     +50.80,
        "hinge_y_cm":     0.0,
        "cam_pos":        [-58.753, 65.219, 27.972],
        "cam_yaw_deg":    270.0,
        "door_center_m":  [-58.753, 63.219, 27.972],  # cam_pos − 2 m in +Y
        "leaf_mesh_names": None,
        "angle_sign":     1,   # parent [-1,-1] rotation; positive opens into room
    },
]

# ---------------------------------------------------------------------------
# Isaac Sim startup
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import time as _time
_t0 = _time.time()
def _ts(msg):
    print(f"[LightTest] [+{_time.time() - _t0:6.1f}s] {msg}")

_ts("Importing SimulationApp ...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
_ts("SimulationApp ready")

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 4)
_s.set("/rtx/post/dlss/execMode", 0)

import omni.usd
from pxr import Usd, UsdLux, UsdGeom, Gf, Sdf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

try:
    from PIL import Image as _PIL
    _pil_ok = True
except ImportError:
    _pil_ok = False
    print("[WARN] PIL unavailable — saving .npy instead of .jpg")

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

_ts("Inserting building USD ...")
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)

# Dome light
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(25_000)
dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# 4-slot ceiling array — mutable translate ops so lights follow each door.
# Offsets match door_replicator_collection.py layout.
_CEILING_SPREAD = 1.2
_CEILING_RISE   = 1.4     # metres above camera Z
_CEILING_OFFSETS = [
    (-_CEILING_SPREAD, -_CEILING_SPREAD),
    (+_CEILING_SPREAD, -_CEILING_SPREAD),
    (-_CEILING_SPREAD, +_CEILING_SPREAD),
    (+_CEILING_SPREAD, +_CEILING_SPREAD),
]

ceiling_lights   = []
ceiling_t_ops    = []   # mutable translate ops, one per slot
ceiling_int_attrs = []
ceiling_col_attrs = []

for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
    path = Sdf.Path(f"/World/CeilingLight_{i}")
    rl   = UsdLux.RectLight.Define(stage, path)
    rl.CreateIntensityAttr(0.0)
    rl.CreateWidthAttr(0.6)
    rl.CreateHeightAttr(0.6)
    xf   = UsdGeom.Xformable(rl.GetPrim())
    t_op = xf.AddTranslateOp()       # stored — we call Set() per door
    r_op = xf.AddRotateXYZOp()
    r_op.Set(Gf.Vec3f(180.0, 0.0, 0.0))   # face downward
    t_op.Set(Gf.Vec3d(dx, dy, 5.0))       # placeholder; updated per door
    ceiling_lights.append(rl)
    ceiling_t_ops.append(t_op)
    ceiling_int_attrs.append(rl.GetIntensityAttr())
    ceiling_col_attrs.append(rl.GetColorAttr())

_ts("world.reset() ...")
world.reset()
_ts("world.reset() done")
for _ in range(5):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

cam = Camera(
    prim_path="/World/LightTestCam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=CAM_RESOLUTION,
    orientation=rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, 180.0]), degrees=True),
)
simulation_app.update()
cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/LightTestCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))

rp  = rep.create.render_product("/World/LightTestCam", CAM_RESOLUTION)
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])

_ts(f"Renderer cold-start warmup ({STARTUP_WARMUP} steps) ...")
for _ in range(STARTUP_WARMUP):
    world.step(render=True)
_ts("Warmup done")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kelvin_to_rgb(k: float) -> Gf.Vec3f:
    t = k / 100.0
    r = 255 if t <= 66 else max(0, min(255, 329.698727446 * ((t - 60) ** -0.1332047592)))
    if t <= 66:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    g = max(0, min(255, g))
    b = 255 if t >= 66 else (0 if t <= 19 else
        max(0, min(255, 138.5177312231 * math.log(t - 10) - 305.0447927307)))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)


def apply_lighting(cond: dict):
    """Write dome + ceiling attributes with ±10%/±200 K jitter."""
    ji = lambda v: v * random.uniform(0.90, 1.10)
    jt = lambda v: v + random.uniform(-200, 200)
    dome.GetIntensityAttr().Set(ji(cond["dome_int"]))
    dome.GetColorAttr().Set(_kelvin_to_rgb(jt(cond["dome_temp"])))
    ci  = ji(cond["ceiling_int"])
    ct  = jt(cond["ceiling_temp"])
    rgb = _kelvin_to_rgb(ct)
    for slot, (int_attr, col_attr) in enumerate(
            zip(ceiling_int_attrs, ceiling_col_attrs)):
        on = cond["ceiling_pattern"][slot]
        int_attr.Set(ci if on else 0.0)
        col_attr.Set(rgb)


def reposition_lights(door_center_m: list, cam_z: float):
    """Move the 4-slot ceiling array to hover above the current door."""
    cx, cy = door_center_m[0], door_center_m[1]
    cz     = cam_z + _CEILING_RISE
    for i, (dx, dy) in enumerate(_CEILING_OFFSETS):
        ceiling_t_ops[i].Set(Gf.Vec3d(cx + dx, cy + dy, cz))


def reposition_camera(pos, yaw_deg):
    ori = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, yaw_deg]), degrees=True)
    cam.set_world_pose(position=np.array(pos), orientation=ori)


def _door_matrix(ox, oy, oz, angle_deg, hx, hy) -> Gf.Matrix4d:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx, dy = ox - hx, oy - hy
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c * dx - s * dy + hx,
        s * dx + c * dy + hy,
        oz, 1,
    )


def set_angle(mesh_ops, angle_deg, hx, hy):
    for m, op in mesh_ops:
        op.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg, hx, hy))


def discover_meshes(door_prim, mesh_count, leaf_mesh_names):
    mesh_ops = []
    prim_iter = (door_prim.GetChildren() if mesh_count > 0
                 else Usd.PrimRange(door_prim))
    for p in prim_iter:
        if not p.IsA(UsdGeom.Mesh):
            continue
        if leaf_mesh_names is not None and p.GetName() not in leaf_mesh_names:
            continue
        xf = UsdGeom.Xformable(p)
        op = next((o for o in xf.GetOrderedXformOps()
                   if o.GetOpName() == "xformOp:transform"), None)
        if op is None:
            continue
        mat = op.Get()
        if mat is None:
            continue
        mesh_ops.append(({"ox": mat[3][0], "oy": mat[3][1], "oz": mat[3][2]}, op))
    return mesh_ops


def capture(door, lighting_name, angle_deg) -> bool:
    world.step(render=True)
    data = ann.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    rgb  = arr[:, :, :3].astype(np.uint8)
    cls  = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
    role_tag = "_HOLDOUT" if door["role"] == "holdout" else ""
    fname = (f"{door['family_short']}_{door['id']}_floor{door['floor']}"
             f"{role_tag}_{lighting_name}_{cls}.jpg")
    path = os.path.join(OUT_DIR, fname)
    if _pil_ok:
        _PIL.fromarray(rgb).save(path, "JPEG", quality=92)
    else:
        np.save(path.replace(".jpg", ".npy"), rgb)
    return True

# ---------------------------------------------------------------------------
# Prime renderer with first door's lighting before the loop
# ---------------------------------------------------------------------------

apply_lighting(LIGHTING_CONDITIONS[0])
_ts("Lighting settle (20 steps, first preset) ...")
for _ in range(20):
    world.step(render=True)
_ts("Settle done — starting main loop")

# ---------------------------------------------------------------------------
# Main loop: door × lighting × angle
# ---------------------------------------------------------------------------

summary = []
total_expected = len(DOORS) * len(LIGHTING_CONDITIONS) * len(ANGLE_GRID)

for di, door in enumerate(DOORS):
    door_id   = door["id"]
    fam_short = door["family_short"]
    role_tag  = "[HOLDOUT]" if door["role"] == "holdout" else "[train  ]"

    print(f"\n{'─'*65}")
    _ts(f"[{di+1:2d}/{len(DOORS)}] {role_tag} {fam_short:<12} {door_id}  floor={door['floor']}")

    entry = {"door_id": door_id, "family": fam_short, "floor": door["floor"],
             "role": door["role"], "status": "ok", "results": {}, "note": ""}

    # -- Validate + concretize --
    door_prim = stage.GetPrimAtPath(door["prim_path"])
    if not door_prim.IsValid():
        _ts(f"  SKIP — prim not found")
        entry.update({"status": "skipped", "note": "prim not valid"})
        summary.append(entry)
        continue

    if door_prim.IsInstance() or door_prim.IsInstanceable():
        door_prim.SetInstanceable(False)
        simulation_app.update()

    # -- Discover meshes --
    mesh_ops = discover_meshes(
        door_prim, door["mesh_count"], door["leaf_mesh_names"])
    if not mesh_ops:
        _ts(f"  SKIP — no xformOp:transform meshes")
        entry.update({"status": "skipped", "note": "no xform meshes"})
        summary.append(entry)
        continue

    _ts(f"  {len(mesh_ops)} mesh(es) — "
        f"{'world' if door['mesh_count'] > 0 else 'local'}-space rotation  "
        f"angle_sign={door['angle_sign']:+d}")

    # -- Reposition camera + ceiling lights to this door --
    reposition_camera(door["cam_pos"], door["cam_yaw_deg"])
    reposition_lights(door["door_center_m"], door["cam_pos"][2])

    # Apply the first lighting preset BEFORE the door-change warmup so
    # the RTX temporal accumulator primes under the correct lighting for
    # this door. Without this, the accumulator arrives at the first
    # capture still converging from the previous door's last preset
    # (often high_contrast — a very different lighting state).
    apply_lighting(LIGHTING_CONDITIONS[0])
    _ts(f"  Door-change warmup ({DOOR_WARMUP} steps, first lighting active) ...")
    for _ in range(DOOR_WARMUP):
        world.step(render=True)

    # -- Lighting × angle inner loop --
    door_ok = True
    for lc in LIGHTING_CONDITIONS:
        apply_lighting(lc)
        for _ in range(WARMUP_STEPS):
            world.step(render=True)

        key = lc["name"]
        entry["results"][key] = {}

        for angle_deg in ANGLE_GRID:
            signed = door["angle_sign"] * angle_deg
            set_angle(mesh_ops, signed, door["hinge_x_cm"], door["hinge_y_cm"])
            ok  = capture(door, lc["name"], angle_deg)
            cls = "CLOSED" if angle_deg <= CLOSED_MAX_DEG else "OPEN"
            entry["results"][key][cls] = "ok" if ok else "failed"
            if not ok:
                door_ok = False
            print(f"  {lc['name']:<22}  {cls:6s}  {'saved' if ok else 'FAILED'}")

        # Reset door to closed after each lighting pass
        set_angle(mesh_ops, 0.0, door["hinge_x_cm"], door["hinge_y_cm"])

    if not door_ok:
        entry["status"] = "partial"
    summary.append(entry)

    saved_so_far = sum(
        1 for e in summary
        for lv in e.get("results", {}).values()
        for v in lv.values() if v == "ok"
    )
    _ts(f"  → {di+1}/{len(DOORS)} doors done  {saved_so_far}/{total_expected} JPEGs saved")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ok_count   = sum(1 for e in summary if e["status"] == "ok")
part_count = sum(1 for e in summary if e["status"] == "partial")
skip_count = sum(1 for e in summary if e["status"] == "skipped")
total_saved = sum(
    1 for e in summary
    for lv in e.get("results", {}).values()
    for v in lv.values() if v == "ok"
)

summary_path = os.path.join(_project_dir, "data", "lighting_test_summary.json")
with open(summary_path, "w") as f:
    json.dump({"ok": ok_count, "partial": part_count, "skipped": skip_count,
               "total_saved": total_saved, "total_expected": total_expected,
               "doors": summary}, f, indent=2)
    f.flush(); os.fsync(f.fileno())

print(f"\n{'='*65}")
_ts(f"Done — {ok_count} doors OK  {part_count} partial  {skip_count} skipped")
_ts(f"JPEGs saved: {total_saved} / {total_expected} expected")
_ts(f"Output  → {OUT_DIR}/")
_ts(f"Summary → {summary_path}")

world.stop()
simulation_app.close()
