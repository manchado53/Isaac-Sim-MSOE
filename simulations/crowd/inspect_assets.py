"""Quick USD asset inspector — run with Isaac Sim python.sh to check prim hierarchy."""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdSkel, UsdGeom
import omni.usd

BASE = os.path.expanduser("~/isaacsim/extscache")

CHARACTER_USD = os.path.join(BASE,
    "omni.metropolis.utils-0.1.20+107.3.3.lx64.cp311/test_data/characters/test/test.usd")

WALK_USD = os.path.join(BASE,
    "omni.anim.retarget.core-107.3.3+107.3.3.lx64.r.cp311.u353"
    "/data/rigs/Human/assets/walk.skelanim.usd")

SKELETON_USD = os.path.join(BASE,
    "omni.anim.retarget.core-107.3.3+107.3.3.lx64.r.cp311.u353"
    "/data/rigs/Human/assets/human_skeleton.usd")

def inspect(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND")
        return
    stage = Usd.Stage.Open(path)
    for prim in stage.Traverse():
        indent = "  " * len(prim.GetPath().pathString.split("/"))
        type_name = prim.GetTypeName()
        print(f"{indent}{prim.GetPath()}  [{type_name}]")
        # Extra info for key types
        if type_name == "SkelAnimation":
            anim = UsdSkel.Animation(prim)
            joints = anim.GetJointsAttr().Get()
            times = anim.GetTranslationsAttr().GetTimeSamples() if anim.GetTranslationsAttr() else []
            print(f"{indent}  joints={len(joints) if joints else 0}, time_samples={len(times)}")
        elif type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            print(f"{indent}  points={len(pts) if pts else 0}")
        elif type_name == "Skeleton":
            skel = UsdSkel.Skeleton(prim)
            joints = skel.GetJointsAttr().Get()
            print(f"{indent}  joints={len(joints) if joints else 0}")

inspect(CHARACTER_USD, "Metropolis test.usd (character)")
inspect(WALK_USD,      "walk.skelanim.usd")
inspect(SKELETON_USD,  "human_skeleton.usd")

simulation_app.close()
