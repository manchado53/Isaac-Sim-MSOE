#!/usr/bin/env python3
"""Walk every descendant (de-instancing as we go) of non-hinged door groups
and print full path + typename + collider state for every Mesh."""
import os
from pxr import Usd, UsdGeom, UsdPhysics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
USD = os.path.join(_ROOT, "assets/environments/msoe_building/MSOEBuildingV4_physics.usd")
stage = Usd.Stage.Open(USD)

def walk_and_find_meshes(prim, meshes=None, depth=0):
    if meshes is None:
        meshes = []
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
    for c in prim.GetChildren():
        walk_and_find_meshes(c, meshes, depth + 1)
    return meshes

GEO = "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026/Geometry/Doors"
doors_prim = stage.GetPrimAtPath(GEO)

for grp in doors_prim.GetChildren():
    name = grp.GetName()
    if not ("Dormakaba" in name or "SIDELITE" in name or "Type_2FG_Door" == name.split('/')[-1] or "2FG" in name):
        continue
    meshes = walk_and_find_meshes(grp)
    n_total = len(meshes)
    n_coll = sum(1 for m in meshes if m.HasAPI(UsdPhysics.CollisionAPI))
    print(f"\n{name}: {n_total} meshes, {n_coll} with colliders")
    for m in meshes[:8]:
        c = "C" if m.HasAPI(UsdPhysics.CollisionAPI) else "-"
        print(f"  [{c}] {m.GetPath()}")
    if n_total > 8:
        print(f"  ... ({n_total - 8} more)")
