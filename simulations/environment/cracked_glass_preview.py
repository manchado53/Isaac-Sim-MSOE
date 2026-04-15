#!/usr/bin/env python3
"""
Cracked-glass single-panel preview renderer.

Picks N random glass panels from a building USD and renders each one
multiple times — once with an intact OmniGlass material and once per
crack variant — so you can visually inspect how each variant looks
when bound to real glass through Isaac Sim's RTX path tracer.

Used to generate the Layer 4 images in docs/cracked-glass-pipeline.md.
Bypasses Replicator's BasicWriter (which doesn't flush its async queue
before detach in our pipeline) and instead reads RGB pixels directly
from a `rep` annotator into PIL.

Usage:
    ./python.sh simulations/environment/cracked_glass_preview.py
    ./python.sh simulations/environment/cracked_glass_preview.py \\
        --variants v4,v5 --n-panels 5 --distance 3.0
"""

import argparse
import json
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJ)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cracked-glass single-panel preview renderer"
    )
    p.add_argument("--usd", default=os.path.join(
        _PROJ, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--variants", default="v1,v2,v3,v4",
                   help="Comma-separated crack variant names to render. "
                        "Each must have crack_normal_<v>.png in the texture "
                        "dir. (default: v1,v2,v3,v4)")
    p.add_argument("--n-panels", type=int, default=3,
                   help="Number of random glass panels to sample (default 3)")
    p.add_argument("--distance", type=float, default=2.5,
                   help="Camera distance from each panel in stage units "
                        "(default 2.5 — meters for InnovationCenter.usd)")
    p.add_argument("--seed", type=int, default=7,
                   help="Random seed for panel selection (default 7 matches "
                        "the doc's reference renders)")
    p.add_argument("--output", default=None,
                   help="Output dir for PNGs (default: "
                        "datasets/cracked_glass/preview_<variants>)")
    p.add_argument("--resolution", default="1280x720",
                   help="Render resolution as WxH (default 1280x720)")
    return p.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim before any pxr/omni imports
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene,
    _find_mesh_descendant,
)
from sim_utils.glass_anomaly import (  # noqa: E402
    create_cracked_glass_materials,
    enumerate_glass_prims,
)
from sim_utils.test_cameras import (  # noqa: E402
    CAMERA_FOCAL_LENGTH,
    add_interior_lighting,
)

TEXTURE_DIR = os.path.join(_PROJ, "assets", "materials", "cracked_glass")

variant_list = tuple(v.strip() for v in args.variants.split(",") if v.strip())
if not variant_list:
    print("[preview] ERROR: --variants must list at least one variant",
          flush=True)
    sys.exit(1)

w, h = (int(x) for x in args.resolution.lower().split("x"))
RESOLUTION = (w, h)

output_dir = args.output or os.path.join(
    _PROJ, "datasets", "cracked_glass",
    f"preview_{'_'.join(variant_list)}",
)
os.makedirs(output_dir, exist_ok=True)
print(f"[preview] USD: {args.usd}", flush=True)
print(f"[preview] Variants: {variant_list}", flush=True)
print(f"[preview] N panels: {args.n_panels}", flush=True)
print(f"[preview] Output: {output_dir}", flush=True)

random.seed(args.seed)


def bind_visual_material_deep(stage, prim, material_path):
    """Bind at both the Xform wrapper AND any Mesh descendant.

    Some panels in InnovationCenter.usd have a material bound directly on
    the inner Mesh prim; binding 'strongerThanDescendants' at the wrapper
    Xform doesn't override a child Mesh binding, so the swap silently
    no-ops. Walking to the Mesh descendant + unbinding any pre-existing
    default-purpose binding fixes it. Documented as Gotcha 1 in
    docs/cracked-glass-pipeline.md.
    """
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    targets = [prim, _find_mesh_descendant(prim, max_depth=4)]
    for target in targets:
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try:
            api.UnbindDirectBinding()
        except Exception:
            pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def panel_pose(prim, distance, bbox_cache):
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = (
        (mn[0] + mx[0]) / 2,
        (mn[1] + mx[1]) / 2,
        (mn[2] + mx[2]) / 2,
    )
    dims = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    axis = min(range(3), key=lambda i: dims[i])  # thinnest = panel normal
    n = [0.0, 0.0, 0.0]
    n[axis] = 1.0
    cam = (
        c[0] + n[0] * distance,
        c[1] + n[1] * distance,
        c[2] + n[2] * distance,
    )
    return cam, c, dims


def make_intact_material(stage, path="/World/Looks/Intact"):
    """Author a clean-glass OmniGlass baseline with low frosting."""
    UsdGeom.Scope.Define(stage, Sdf.Path(path).GetParentPath())
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset("OmniGlass.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        (0.90, 0.95, 0.97))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.003)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(
        sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return path


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------
if not os.path.isfile(args.usd):
    print(f"[preview] ERROR: USD not found: {args.usd}", flush=True)
    sys.exit(2)

world = World()
stage = omni.usd.get_context().get_stage()

bp = stage.DefinePrim("/World/Building", "Xform")
bp.GetReferences().AddReference(args.usd, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)
add_interior_lighting(stage)

# Author the intact baseline + each requested cracked variant
intact_path = make_intact_material(stage)
cracked_paths = {}
for v in variant_list:
    paths = create_cracked_glass_materials(
        stage,
        texture_dir=TEXTURE_DIR,
        root_path=f"/World/Looks/Crack_{v}",
        variants=(v,),
    )
    if not paths:
        print(f"[preview] WARN: variant '{v}' produced no material "
              f"(missing crack_normal_{v}.png?)", flush=True)
        continue
    cracked_paths[v] = paths[0]

if not cracked_paths:
    print("[preview] ERROR: no valid variants — aborting", flush=True)
    sys.exit(3)

glass = enumerate_glass_prims(stage)
print(f"[preview] Glass prims discovered: {len(glass)}", flush=True)
world.reset()

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)

# Filter to panels with reasonable area (>1 m^2 in stage units) so we don't
# pick tiny fragments. Then sample N at random.
candidates = []
for p in glass:
    _, _, d = panel_pose(p, args.distance, bbox_cache)
    if sorted(d)[1] * sorted(d)[2] > 1.0:
        candidates.append(p)
random.shuffle(candidates)
panels = candidates[: args.n_panels]
print(f"[preview] Selected {len(panels)} panels", flush=True)

# Build the variant list in render order: intact first, then each cracked
render_variants = [("intact", intact_path)]
for v in variant_list:
    if v in cracked_paths:
        render_variants.append((v, cracked_paths[v]))

# -----------------------------------------------------------------------------
# Per-panel render loop
# -----------------------------------------------------------------------------
for idx, panel in enumerate(panels):
    cam_pos, look_at, dims = panel_pose(panel, args.distance, bbox_cache)
    mesh = _find_mesh_descendant(panel, max_depth=4)
    print(f"[preview] panel {idx}: {panel.GetName()} "
          f"dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}) "
          f"mesh={mesh.GetName() if mesh else None}", flush=True)

    cam_node = rep.create.camera(
        position=cam_pos, look_at=look_at,
        focal_length=CAMERA_FOCAL_LENGTH, horizontal_aperture=20.955,
        name=f"PreviewCam_{idx}",
    )
    rp = rep.create.render_product(cam_node, RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])

    for vname, mpath in render_variants:
        bind_visual_material_deep(stage, panel, mpath)
        # Step a few times to let materials + shaders settle
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4)
        data = rgb_ann.get_data()
        if data is None or not hasattr(data, "shape"):
            print(f"  [WARN] no RGB data for {vname}", flush=True)
            continue
        img = np.asarray(data)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        dst = os.path.join(output_dir, f"panel{idx:02d}_{vname}.png")
        Image.fromarray(img.astype(np.uint8)).save(dst)
        print(f"  saved {os.path.basename(dst)}", flush=True)

    # Detaching the annotator throws a benign HydraTexture warning on
    # Isaac Sim 5.1; swallow it so the loop continues.
    try:
        rgb_ann.detach([rp])
    except Exception as e:
        print(f"  [detach warn] {e}", flush=True)

# -----------------------------------------------------------------------------
# Sidecar manifest
# -----------------------------------------------------------------------------
meta = {
    "usd": args.usd,
    "panels": [str(p.GetPath()) for p in panels],
    "variants": [v[0] for v in render_variants],
    "binding": "bind_visual_material_deep (Xform + Mesh descendant)",
    "seed": args.seed,
    "distance": args.distance,
    "resolution": RESOLUTION,
}
with open(os.path.join(output_dir, "preview_manifest.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[preview] DONE. Manifest: {output_dir}/preview_manifest.json",
      flush=True)
simulation_app.close()
