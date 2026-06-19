#!/usr/bin/env python3
"""
Inspect one Type_F_Door and one Type_F_W_SIDELITE_Door to find where their
actual geometry lives. The door_catalog.json reports mesh_count=0 for all
non-curtain-wall door families, but the doors obviously have geometry since
they have bounding boxes. This script walks the full prim subtree beneath
a representative door to locate any Mesh prims and report their depth/path.
"""

import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
USD_PATH     = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")

# Pick one door of each interesting family from the catalog
with open(os.path.join(_script_dir, "door_catalog.json")) as f:
    catalog = json.load(f)

TARGET_FAMILIES = [
    "RS___Type_F_Door",
    "RS___Type_F_W_SIDELITE_Door",
    "RS___Type_FG_Door",
    "RS___Type_2F_Door",
]

samples = {}
for d in catalog["doors"]:
    fam = d["door_type_family"]
    if fam in TARGET_FAMILIES and fam not in samples:
        samples[fam] = d
    if len(samples) == len(TARGET_FAMILIES):
        break

print(f"Opening stage {USD_PATH} ...")
omni.usd.get_context().open_stage(USD_PATH)
simulation_app.update()
stage = omni.usd.get_context().get_stage()

def walk(prim, depth=0, max_depth=12):
    if depth > max_depth:
        return
    type_name = prim.GetTypeName()
    indent = "  " * depth
    marker = " <Mesh>" if prim.IsA(UsdGeom.Mesh) else ""
    print(f"{indent}{prim.GetName()}  [{type_name}]{marker}")
    for child in prim.GetChildren():
        walk(child, depth + 1, max_depth)

for fam, d in samples.items():
    path = d["prim_path"]
    prim = stage.GetPrimAtPath(path)
    print("=" * 80)
    print(f"FAMILY: {fam}")
    print(f"PATH:   {path}")
    print(f"SIZE:   {d['dimensions']['width']:.2f} x {d['dimensions']['thickness']:.2f} x {d['dimensions']['height']:.2f} m")
    print(f"FLOOR:  {d['floor']}")
    if not prim.IsValid():
        print("  ERROR: prim not valid")
        continue
    print(f"PRIM TYPE: {prim.GetTypeName()}")
    # References / payloads?
    refs = prim.GetMetadata("references")
    payloads = prim.GetMetadata("payload")
    print(f"REFERENCES: {refs}")
    print(f"PAYLOAD:    {payloads}")
    print("SUBTREE:")
    walk(prim)
    # Count Mesh prims in subtree
    mesh_count = 0
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            mesh_count += 1
    print(f"TOTAL MESH DESCENDANTS: {mesh_count}")
    print()

simulation_app.close()
