#!/usr/bin/env python3
"""Remove all Planting (tree) prims from a Revit-exported USD file."""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from pxr import Usd, Sdf

DEFAULT_USD = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "assets", "environments", "msoe_building", "MSOEBuildingApril.usd"
)
USD_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USD

def find_planting_roots(stage):
    """Find top-level Planting category folders."""
    roots = []
    for prim in stage.TraverseAll():
        if prim.GetName() == "Planting" and prim.GetChildren():
            roots.append(prim.GetPath())
    return roots

def main():
    usd_path = os.path.abspath(USD_PATH)
    print(f"Opening: {usd_path}")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("ERROR: Could not open stage")
        sys.exit(1)

    roots = find_planting_roots(stage)
    print(f"Found {len(roots)} Planting folder(s):")

    total_removed = 0
    for root_path in roots:
        prim = stage.GetPrimAtPath(root_path)
        children = prim.GetChildren()
        count = len(children)
        print(f"  {root_path} — {count} tree prim(s)")
        total_removed += count

    if total_removed == 0:
        print("No planting prims found. Nothing to do.")
        sys.exit(0)

    # Remove each Planting folder entirely
    for root_path in roots:
        print(f"Removing {root_path} ...")
        stage.RemovePrim(root_path)

    stage.GetRootLayer().Save()
    print(f"Saved. Removed {total_removed} tree prims from {len(roots)} Planting folder(s).")

if __name__ == "__main__":
    main()
