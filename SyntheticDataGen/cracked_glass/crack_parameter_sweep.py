#!/usr/bin/env python3
"""
Cracked-glass parameter sweep — renders each parameter in isolation
on 5 glass panels so you can evaluate what each knob actually does.

Output structure:
  datasets/cracked_glass/sweep_test/
  ├── by_seed/        — different crack patterns (same panel, lighting, hits)
  ├── by_hits/        — different density (same panel, seed, lighting)
  ├── by_lighting/    — different lighting presets (same panel, crack)
  ├── by_bump/        — different bump intensity (same panel, crack)
  ├── by_panel/       — different panel shapes (same crack, lighting)
  └── intact_baseline/ — all 5 panels with no cracks

Usage:
    ~/isaacsim/python.sh SyntheticDataGen/cracked_glass/crack_parameter_sweep.py
"""

import os, sys, random, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _SCRIPT_DIR)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image
import omni.usd
import omni.replicator.core as rep
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdLux, Gf

from isaacsim.core.api import World
from sim_utils.building_physics import setup_physics_scene, _find_mesh_descendant
from glass_anomaly import enumerate_glass_prims
from sim_utils.test_cameras import CAMERA_FOCAL_LENGTH

USD = f"{_PROJ}/assets/environments/InnovationCenter/InnovationCenter.usd"
TEX = f"{_PROJ}/assets/materials/cracked_glass"
OUT = f"{_PROJ}/datasets/cracked_glass/sweep_test"

# --- Sweep dimensions ---

SEEDS = {
    "s1337": f"{TEX}/crack_normal_v4.png",
    "s100":  f"{TEX}/crack_normal_s100.png",
    "s2025": f"{TEX}/crack_normal_s2025.png",
    "s9999": f"{TEX}/crack_normal_s9999.png",
}

HITS_VALUES = [0, 1, 4, 9]

BUMP_VALUES = [0.5, 1.0, 2.0, 3.0]

LIGHTING_PRESETS = [
    {"name": "bright_fluorescent", "dome_int": 7500,  "dome_temp": 5500,
     "fill_int": 45000, "fill_temp": 4500},
    {"name": "daylight_flood",     "dome_int": 10000, "dome_temp": 6500,
     "fill_int": 20000, "fill_temp": 6000},
    {"name": "warm_evening",       "dome_int": 3500,  "dome_temp": 3000,
     "fill_int": 25000, "fill_temp": 2800},
    {"name": "dim_night",          "dome_int": 1500,  "dome_temp": 4000,
     "fill_int": 8000,  "fill_temp": 3500},
    {"name": "high_contrast",      "dome_int": 1500,  "dome_temp": 5000,
     "fill_int": 80000, "fill_temp": 5500},
]

N_PANELS = 5
RESOLUTION = (1280, 720)

# --- Helpers ---

OMNIGLASS_MDL = "OmniGlass.mdl"

def _color_temp_to_rgb(temp_k):
    """Approximate blackbody color temperature to RGB."""
    t = temp_k / 100.0
    if t <= 66:
        r = 1.0
        g = max(0, min(1, (99.4708025861 * math.log(t) - 161.1195681661) / 255.0))
    else:
        r = max(0, min(1, (329.698727446 * ((t - 60) ** -0.1332047592)) / 255.0))
        g = max(0, min(1, (288.1221695283 * ((t - 60) ** -0.0755148492)) / 255.0))
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = max(0, min(1, (138.5177312231 * math.log(t - 10) - 305.0447927307) / 255.0))
    return Gf.Vec3f(r, g, b)


def make_cracked_material(stage, path, normal_tex, crack_hits=0, bump=1.0,
                          frosting=0.45, glass_color=(0.82, 0.90, 0.94)):
    """Create one OmniGlass cracked material with specified params."""
    parent = str(Sdf.Path(path).GetParentPath())
    if not stage.GetPrimAtPath(parent):
        UsdGeom.Scope.Define(stage, parent)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset(OMNIGLASS_MDL, "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*glass_color))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(frosting)
    sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.005)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)

    if crack_hits > 0:
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
        if crack_hits > 1:
            n = math.ceil(math.sqrt(crack_hits))
            sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(1.0 / n, 1.0 / n))

    if normal_tex and os.path.isfile(normal_tex):
        sh.CreateInput("normal_map_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(normal_tex))
        sh.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(bump)
        sh.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(True)

    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return path


def make_intact_material(stage, path="/World/Looks/Intact"):
    parent = str(Sdf.Path(path).GetParentPath())
    if not stage.GetPrimAtPath(parent):
        UsdGeom.Scope.Define(stage, parent)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset(OMNIGLASS_MDL, "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.90, 0.95, 0.97))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.003)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return path


def bind_deep(stage, prim, mat_path):
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    for target in [prim, _find_mesh_descendant(prim, max_depth=4)]:
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try:
            api.UnbindDirectBinding()
        except Exception:
            pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def panel_camera_pose(prim, distance, bbox_cache):
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = tuple((mn[i] + mx[i]) / 2 for i in range(3))
    dims = [mx[i] - mn[i] for i in range(3)]
    axis = min(range(3), key=lambda i: dims[i])
    n = [0, 0, 0]
    n[axis] = 1.0
    cam = tuple(c[i] + n[i] * distance for i in range(3))
    return cam, c


# --- Scene setup ---

print("[sweep] Loading scene...", flush=True)
world = World()
stage = omni.usd.get_context().get_stage()
bp = stage.DefinePrim("/World/Building", "Xform")
bp.GetReferences().AddReference(USD, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)

# Create controllable lights
dome_path = "/World/SweepDome"
dome = UsdLux.DomeLight.Define(stage, dome_path)
dome.GetIntensityAttr().Set(7500)
dome.GetColorAttr().Set(_color_temp_to_rgb(5500))

fill_path = "/World/SweepFill"
fill = UsdLux.SphereLight.Define(stage, fill_path)
fill.GetIntensityAttr().Set(45000)
fill.GetRadiusAttr().Set(0.5)

sun_path = "/World/SweepSun"
sun = UsdLux.DistantLight.Define(stage, sun_path)
sun.GetIntensityAttr().Set(3000)
sun.GetAngleAttr().Set(0.53)


def apply_lighting(preset):
    dome.GetIntensityAttr().Set(preset["dome_int"])
    dome.GetColorAttr().Set(_color_temp_to_rgb(preset["dome_temp"]))
    fill.GetIntensityAttr().Set(preset["fill_int"])
    fill.GetColorAttr().Set(_color_temp_to_rgb(preset["fill_temp"]))


# Default lighting
DEFAULT_LIGHTING = LIGHTING_PRESETS[0]
apply_lighting(DEFAULT_LIGHTING)

# Default crack texture
DEFAULT_SEED = "s1337"
DEFAULT_NORMAL = SEEDS[DEFAULT_SEED]
DEFAULT_HITS = 0
DEFAULT_BUMP = 1.0

glass = enumerate_glass_prims(stage)
print(f"[sweep] Glass prims: {len(glass)}", flush=True)

world.reset()

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)

# Pick 5 panels with decent size
random.seed(7)
candidates = []
for p in glass:
    bb = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    dims = [mx[i] - mn[i] for i in range(3)]
    if sorted(dims)[1] * sorted(dims)[2] > 1.0:
        candidates.append(p)
random.shuffle(candidates)
panels = candidates[:N_PANELS]
print(f"[sweep] Selected {len(panels)} panels", flush=True)

# Use panel 0 as the default for single-parameter sweeps
default_panel = panels[0]
cam_pos, look_at = panel_camera_pose(default_panel, 2.0, bbox_cache)

# Intact material
intact_mat = make_intact_material(stage)


def render_frame(cam_pos, look_at, out_path, cam_name="SweepCam"):
    cam_node = rep.create.camera(
        position=cam_pos, look_at=look_at,
        focal_length=CAMERA_FOCAL_LENGTH, horizontal_aperture=20.955,
        name=cam_name,
    )
    rp = rep.create.render_product(cam_node, RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])
    for _ in range(5):
        rep.orchestrator.step(rt_subframes=4)
    data = rgb_ann.get_data()
    if data is not None:
        img = np.asarray(data)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(img.astype(np.uint8)).save(out_path)
    try:
        rgb_ann.detach([rp])
    except Exception:
        pass
    return data is not None


mat_counter = [0]
def next_mat_path():
    mat_counter[0] += 1
    return f"/World/Looks/Sweep_{mat_counter[0]:04d}"


# =========================================================================
# SWEEP 1: by_seed — vary crack pattern, fix everything else
# =========================================================================
print("\n[sweep] === by_seed ===", flush=True)
apply_lighting(DEFAULT_LIGHTING)
for seed_name, normal_path in SEEDS.items():
    mat = make_cracked_material(stage, next_mat_path(), normal_path,
                                crack_hits=DEFAULT_HITS, bump=DEFAULT_BUMP)
    bind_deep(stage, default_panel, mat)
    out = f"{OUT}/by_seed/{seed_name}.png"
    ok = render_frame(cam_pos, look_at, out, f"seed_{seed_name}")
    print(f"  {'saved' if ok else 'FAILED'} {seed_name}", flush=True)

# =========================================================================
# SWEEP 2: by_hits — vary density, fix everything else
# =========================================================================
print("\n[sweep] === by_hits ===", flush=True)
for hits in HITS_VALUES:
    mat = make_cracked_material(stage, next_mat_path(), DEFAULT_NORMAL,
                                crack_hits=hits, bump=DEFAULT_BUMP)
    bind_deep(stage, default_panel, mat)
    out = f"{OUT}/by_hits/hits_{hits}.png"
    ok = render_frame(cam_pos, look_at, out, f"hits_{hits}")
    print(f"  {'saved' if ok else 'FAILED'} hits={hits}", flush=True)

# =========================================================================
# SWEEP 3: by_lighting — vary lighting, fix everything else
# =========================================================================
print("\n[sweep] === by_lighting ===", flush=True)
mat = make_cracked_material(stage, next_mat_path(), DEFAULT_NORMAL,
                            crack_hits=DEFAULT_HITS, bump=DEFAULT_BUMP)
bind_deep(stage, default_panel, mat)
for preset in LIGHTING_PRESETS:
    apply_lighting(preset)
    out = f"{OUT}/by_lighting/{preset['name']}.png"
    ok = render_frame(cam_pos, look_at, out, f"light_{preset['name']}")
    print(f"  {'saved' if ok else 'FAILED'} {preset['name']}", flush=True)
apply_lighting(DEFAULT_LIGHTING)  # reset

# =========================================================================
# SWEEP 4: by_bump — vary bump intensity, fix everything else
# =========================================================================
print("\n[sweep] === by_bump ===", flush=True)
for bump_val in BUMP_VALUES:
    mat = make_cracked_material(stage, next_mat_path(), DEFAULT_NORMAL,
                                crack_hits=DEFAULT_HITS, bump=bump_val)
    bind_deep(stage, default_panel, mat)
    out = f"{OUT}/by_bump/bump_{bump_val:.1f}.png"
    ok = render_frame(cam_pos, look_at, out, f"bump_{bump_val}")
    print(f"  {'saved' if ok else 'FAILED'} bump={bump_val}", flush=True)

# =========================================================================
# SWEEP 5: by_panel — same crack on different panels
# =========================================================================
print("\n[sweep] === by_panel ===", flush=True)
mat = make_cracked_material(stage, next_mat_path(), DEFAULT_NORMAL,
                            crack_hits=DEFAULT_HITS, bump=DEFAULT_BUMP)
for i, panel in enumerate(panels):
    bind_deep(stage, panel, mat)
    p_cam, p_look = panel_camera_pose(panel, 2.0, bbox_cache)
    out = f"{OUT}/by_panel/panel_{i}_{panel.GetName()}.png"
    ok = render_frame(p_cam, p_look, out, f"panel_{i}")
    # Unbind after (restore intact for other panels)
    bind_deep(stage, panel, intact_mat)
    print(f"  {'saved' if ok else 'FAILED'} panel_{i}: {panel.GetName()}", flush=True)

# =========================================================================
# SWEEP 6: intact_baseline — all 5 panels with no cracks
# =========================================================================
print("\n[sweep] === intact_baseline ===", flush=True)
for i, panel in enumerate(panels):
    bind_deep(stage, panel, intact_mat)
    p_cam, p_look = panel_camera_pose(panel, 2.0, bbox_cache)
    out = f"{OUT}/intact_baseline/panel_{i}_{panel.GetName()}.png"
    ok = render_frame(p_cam, p_look, out, f"intact_{i}")
    print(f"  {'saved' if ok else 'FAILED'} panel_{i}: {panel.GetName()}", flush=True)

print(f"\n[sweep] DONE. Output at {OUT}/", flush=True)
app.close()
