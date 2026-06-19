"""
lighting_presets.py -- Discrete indoor lighting presets for cracked-glass
dataset collection.

Port of origin/tanner_vlm:simulations/robot/door_static_collection.py
(LIGHTING_CONDITIONS at lines 213-229, _color_temp_to_rgb at 603-632,
apply_lighting_condition at 651-678). Two structural changes:
  - Presets live on a `handles` dict instead of module globals, so the
    collector can own the dome/fill prims (the door collector had them
    at fixed global paths).
  - Optional `rng` arg in `apply_lighting_preset` instead of importing
    `random` -- makes the full grid run reproducible per `--seed`.

The 7th Tanner preset (camera_headlight) is intentionally absent -- we
dropped it after the sweep review.

Public API:
  LIGHTING_PRESETS                            -- dict of 6 preset configs
  setup_lighting_rig(stage, anchor_xyz)       -- creates dome + fill,
                                                 sets carb settings,
                                                 returns handles dict
  apply_lighting_preset(handles, name, rng)   -- mutates lights per preset
  set_fill_position(handles, xyz)             -- moves fill prim
"""

import math
import os
import random

import carb.settings
from pxr import Gf, Sdf, UsdGeom, UsdLux

# ---------------------------------------------------------------------------
# HDRI environment map -- bundled with Isaac Sim 5.1. Used by the two
# presets where strong natural/window light is expected. If the file is
# missing, apply_lighting_preset falls back to the flat dome color.
# ---------------------------------------------------------------------------
_HDRI_STUDIO = os.path.expanduser(
    "~/isaacsim/extscache/omni.kit.widget.material_preview-1.0.16/data/"
    "photo_studio_01_4k.hdr"
)


# ---------------------------------------------------------------------------
# 6 realistic indoor lighting scenarios for the Innovation Center.
# Verbatim from Tanner's LIGHTING_CONDITIONS (door_static_collection.py
# lines 213-229), minus camera_headlight.
# ---------------------------------------------------------------------------
LIGHTING_PRESETS = {
    "bright_fluorescent": {
        "dome_int":  25_000, "dome_temp": 5500,
        "fill_int": 400_000, "fill_temp": 4500,
        "hdri": None,
        "note": "artificial ceiling light (our validated default)",
    },
    "daylight_flood": {
        "dome_int":  70_000, "dome_temp": 6500,
        "fill_int": 150_000, "fill_temp": 6000,
        "hdri": _HDRI_STUDIO,
        "note": "window-driven natural light",
    },
    "warm_evening": {
        "dome_int":  20_000, "dome_temp": 3000,
        "fill_int": 200_000, "fill_temp": 2800,
        "hdri": None,
        "note": "warm tungsten fixtures",
    },
    "dim_night": {
        "dome_int":   5_000, "dome_temp": 4000,
        "fill_int":  50_000, "fill_temp": 3500,
        "hdri": None,
        "note": "minimal overnight lighting",
    },
    "overcast_flat": {
        "dome_int":  40_000, "dome_temp": 5000,
        "fill_int": 100_000, "fill_temp": 5000,
        "hdri": _HDRI_STUDIO,
        "note": "diffuse sky through windows",
    },
    "high_contrast": {
        "dome_int":   5_000, "dome_temp": 5000,
        "fill_int": 700_000, "fill_temp": 5500,
        "hdri": None,
        "note": "single bright overhead only",
    },
}


def _color_temp_to_rgb(kelvin: float) -> Gf.Vec3f:
    """
    Approximate color temperature (K) to normalized RGB. Tanner Helland's
    algorithm, verbatim from door_static_collection.py:603-632.
    """
    temp = kelvin / 100.0
    # Red
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = max(0.0, min(255.0, r))
    # Green
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = max(0.0, min(255.0, g))
    # Blue
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(temp - 10) - 305.0447927307
        b = max(0.0, min(255.0, b))
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)


def setup_lighting_rig(stage, anchor_xyz, dome_path="/World/CGDome",
                       fill_path="/World/CGFill"):
    """
    Create a DomeLight + SphereLight fill pair and set the two carb
    settings validated in cracked_glass_preview.py.

    Matches the lighting topology of sweep_preview.py:224-235: dome for
    global image-based or flat-colored ambient; movable sphere fill
    positioned near the camera (call `set_fill_position` per panel).

    Returns a handles dict consumed by apply_lighting_preset.
    """
    carb.settings.get_settings().set("/rtx/domeLight/upperLowerStrategy", 3)
    carb.settings.get_settings().set("/rtx/pathTracing/maxBounces", 8)

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_path))
    dome.CreateIntensityAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    # Pre-create the texture attribute so apply_lighting_preset can .Set
    # it each call without re-Creating.
    dome_tex = dome.CreateTextureFileAttr()
    dome_tex.Set(Sdf.AssetPath(""))

    fill = UsdLux.SphereLight.Define(stage, Sdf.Path(fill_path))
    fill.CreateIntensityAttr(0.0)
    fill.CreateRadiusAttr(0.1)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    fill_xform = UsdGeom.XformCommonAPI(fill.GetPrim())
    fill_xform.SetTranslate(tuple(float(c) for c in anchor_xyz))

    return {
        "dome":       dome,
        "dome_tex":   dome_tex,
        "fill":       fill,
        "fill_xform": fill_xform,
    }


def apply_lighting_preset(handles, name, rng=None):
    """
    Apply a discrete preset from LIGHTING_PRESETS, with small jitter
    (±10% intensity, ±200K temperature) within the cell. Lifted from
    door_static_collection.py:651-678 with two changes: takes a handles
    dict instead of module globals, and uses an optional `rng` instead
    of the `random` module.

    Returns the actual (jittered) dome_int, dome_temp, fill_int,
    fill_temp for logging into labels.jsonl.
    """
    if name not in LIGHTING_PRESETS:
        raise KeyError(f"Unknown lighting preset: {name!r}. "
                       f"Known: {sorted(LIGHTING_PRESETS)}")
    preset = LIGHTING_PRESETS[name]
    r = rng if rng is not None else random

    dome_int  = preset["dome_int"]  * r.uniform(0.90, 1.10)
    dome_temp = preset["dome_temp"] + r.uniform(-200.0, 200.0)
    fill_int  = preset["fill_int"]  * r.uniform(0.90, 1.10)
    fill_temp = preset["fill_temp"] + r.uniform(-200.0, 200.0)

    dome = handles["dome"]
    fill = handles["fill"]
    dome.GetIntensityAttr().Set(float(dome_int))
    dome.GetColorAttr().Set(_color_temp_to_rgb(dome_temp))
    fill.GetIntensityAttr().Set(float(fill_int))
    fill.GetColorAttr().Set(_color_temp_to_rgb(fill_temp))

    # HDRI swap: preset has a path -> enable image-based lighting;
    # preset is None or file missing -> clear texture so dome uses flat color.
    hdri = preset.get("hdri")
    if hdri and os.path.isfile(hdri):
        handles["dome_tex"].Set(Sdf.AssetPath(hdri))
    else:
        handles["dome_tex"].Set(Sdf.AssetPath(""))

    return dome_int, dome_temp, fill_int, fill_temp


def set_fill_position(handles, xyz):
    """
    Move the fill SphereLight to `xyz`. Collector calls this once per
    panel to put the fill ~1.5m above the current camera (pattern from
    sweep_preview.py and door_static_collection.py:351).
    """
    handles["fill_xform"].SetTranslate(tuple(float(c) for c in xyz))
