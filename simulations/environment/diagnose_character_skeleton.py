#!/usr/bin/env python3
"""
Diagnose the skeleton structure of an NVIDIA biped character.

Loads F_Business_02 and dumps:
  - The SkelRoot prim path
  - The Skeleton prim path
  - Full joint list (with topology order)
  - Rest transforms per joint
  - Bind transforms per joint
  - Whether a SkelAnimation is bound

Output: logs/character_skeleton.json + stdout summary.

Used to plan UsdSkel.Animation joint manipulation for realistic poses
(e.g., lying down with arms at sides, instead of rotating a T-pose body).
"""
import os
import sys
import json

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdSkel, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.utils import prims as prims_utils
from isaacsim.storage.native import get_assets_root_path

CHARACTER_SUBPATH = "/Isaac/People/Characters/F_Business_02/F_Business_02.usd"


def vec3_to_list(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def quat_to_list(q):
    """Convert Gf.Quatf/Quatd to [w, x, y, z]."""
    try:
        r = q.GetReal()
        i = q.GetImaginary()
        return [float(r), float(i[0]), float(i[1]), float(i[2])]
    except Exception:
        return None


def matrix_to_list(m):
    """Convert Gf.Matrix4d to a 4x4 list of floats."""
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]


def decompose_xform(m):
    """Extract translation + rotation quaternion from a 4x4 matrix."""
    try:
        translation = vec3_to_list(m.ExtractTranslation())
    except Exception:
        translation = None
    try:
        rotation = quat_to_list(m.ExtractRotationQuat())
    except Exception:
        try:
            rotation = quat_to_list(m.ExtractRotation().GetQuat())
        except Exception:
            rotation = None
    return {"translation": translation, "rotation_quat_wxyz": rotation}


def main():
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[diag_skel] ERROR: get_assets_root_path() returned None", flush=True)
        sys.exit(1)
    character_usd = assets_root + CHARACTER_SUBPATH
    print(f"[diag_skel] Loading: {character_usd}", flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Load character into minimal stage (no building needed for skeleton inspection)
    person = prims_utils.create_prim(
        "/World/Person_01",
        "Xform",
        usd_path=character_usd,
    )
    if not person or not person.IsValid():
        print("[diag_skel] ERROR: could not load character", flush=True)
        sys.exit(2)

    world.reset()
    for _ in range(5):
        world.step(render=False)

    # Find SkelRoot
    skel_roots = []
    skeletons = []
    skel_anims = []
    anim_graphs = []
    all_types = {}
    for prim in Usd.PrimRange(person):
        t = prim.GetTypeName()
        all_types[t] = all_types.get(t, 0) + 1
        if t == "SkelRoot":
            skel_roots.append(str(prim.GetPath()))
        elif t == "Skeleton":
            skeletons.append(str(prim.GetPath()))
        elif t == "SkelAnimation":
            skel_anims.append(str(prim.GetPath()))
        elif "AnimationGraph" in t or "AnimGraph" in t:
            anim_graphs.append(str(prim.GetPath()))

    print(f"[diag_skel] Prim types: {all_types}", flush=True)
    print(f"[diag_skel] SkelRoots: {skel_roots}", flush=True)
    print(f"[diag_skel] Skeletons: {skeletons}", flush=True)
    print(f"[diag_skel] SkelAnimations: {skel_anims}", flush=True)
    print(f"[diag_skel] AnimationGraphs: {anim_graphs}", flush=True)

    result = {
        "character_usd": character_usd,
        "prim_type_counts": all_types,
        "skel_roots": skel_roots,
        "skeletons": skeletons,
        "skel_animations": skel_anims,
        "animation_graphs": anim_graphs,
        "skeleton_details": [],
    }

    # For each Skeleton, dump joint list, rest transforms, bind transforms
    for skel_path in skeletons:
        sk_prim = stage.GetPrimAtPath(skel_path)
        sk = UsdSkel.Skeleton(sk_prim)
        joints = sk.GetJointsAttr().Get() or []
        rest_xforms = sk.GetRestTransformsAttr().Get() or []
        bind_xforms = sk.GetBindTransformsAttr().Get() or []

        joint_data = []
        for i, jpath in enumerate(joints):
            entry = {
                "index": i,
                "path": str(jpath),
                "leaf_name": str(jpath).split("/")[-1],
                "depth": str(jpath).count("/"),
            }
            if i < len(rest_xforms):
                try:
                    entry["rest"] = decompose_xform(rest_xforms[i])
                except Exception as e:
                    entry["rest_error"] = str(e)
            if i < len(bind_xforms):
                try:
                    entry["bind"] = decompose_xform(bind_xforms[i])
                except Exception as e:
                    entry["bind_error"] = str(e)
            joint_data.append(entry)

        # Check binding
        binding_api = UsdSkel.BindingAPI(sk_prim)
        anim_targets = []
        try:
            rel = binding_api.GetAnimationSourceRel()
            if rel:
                anim_targets = [str(t) for t in (rel.GetTargets() or [])]
        except Exception:
            pass

        result["skeleton_details"].append({
            "path": skel_path,
            "n_joints": len(joints),
            "animation_source_targets": anim_targets,
            "joints": joint_data,
        })

    # Dump
    out_path = os.path.join(_project_dir, "logs", "character_skeleton.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[diag_skel] Wrote {out_path} ({os.path.getsize(out_path)} bytes)",
          flush=True)

    # Print a compact joint list summary
    for skel in result["skeleton_details"]:
        print(f"\n[diag_skel] Skeleton {skel['path']}: {skel['n_joints']} joints",
              flush=True)
        # List first 60 for visibility
        for j in skel["joints"][:80]:
            rest_t = j.get("rest", {}).get("translation")
            t_str = f"t={rest_t}" if rest_t else ""
            print(f"  [{j['index']:3d}] {j['path']}  {t_str}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
