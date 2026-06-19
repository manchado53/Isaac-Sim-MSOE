#!/usr/bin/env python3
"""Inspect the subtree of a Type_F_W_SIDELITE_Door after SetInstanceable(False)
to identify which child prims are the rotating door leaf vs the fixed sidelite."""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdShade
from isaacsim.core.api import World

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USD_PATH = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

print(f"Loading {USD_PATH} as sublayer ...")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, USD_PATH)
world.reset()
for _ in range(5):
    simulation_app.update()

# Diagnostic: print top-level prims so we can confirm the path structure
print("Root-level prims:")
for p in stage.GetPseudoRoot().GetChildren():
    print(f"  /{p.GetName()}")
    for c in p.GetChildren():
        print(f"    /{p.GetName()}/{c.GetName()}")

PRIM_PATH = ("/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
             "/Geometry/Doors/RS___Type_F_W_SIDELITE_Door"
             "/NEW___HM_Frame___36_x84_w_Sidelite/Doors_9563128")

prim = stage.GetPrimAtPath(PRIM_PATH)
print(f"\nPrim path tried: {PRIM_PATH}")
print(f"Valid: {prim.IsValid()}")
print(f"IsInstance: {prim.IsInstance()}  IsInstanceable: {prim.IsInstanceable()}")

if prim.IsInstance() or prim.IsInstanceable():
    print("Calling SetInstanceable(False) ...")
    prim.SetInstanceable(False)
    simulation_app.update()

print("\n=== Subtree after concretizing ===")
for p in Usd.PrimRange(prim):
    depth  = str(p.GetPath()).replace(str(prim.GetPath()), "").count("/")
    indent = "  " * depth
    is_mesh = p.IsA(UsdGeom.Mesh)
    xf = UsdGeom.Xformable(p)
    ops = [o.GetOpName() for o in xf.GetOrderedXformOps()]
    has_xform = "xformOp:transform" in ops

    mat_path = ""
    if is_mesh:
        binding = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
        mat_path = str(binding.GetMaterialPath()) if binding.GetMaterialPath() else "none"

    mesh_tag  = " <Mesh>"   if is_mesh   else ""
    xform_tag = " <xform>"  if has_xform else ""
    print(f"{indent}{p.GetName()}  [{p.GetTypeName()}]{mesh_tag}{xform_tag}")
    if is_mesh:
        print(f"{indent}  mat={mat_path}")
        print(f"{indent}  xform_ops={ops}")
        # Print local translation from xformOp:transform if present
        if has_xform:
            for o in xf.GetOrderedXformOps():
                if o.GetOpName() == "xformOp:transform":
                    m = o.Get()
                    if m:
                        print(f"{indent}  local_translation_cm=({m[3][0]:.2f}, {m[3][1]:.2f}, {m[3][2]:.2f})")

print("\nDone.")
simulation_app.close()
