"""Create a wrapper USDA for roscoe_robot.usdz with defaultPrim set.

cd ~/isaacsim && ./python.sh ~/isaac-sim-msoe/simulations/crowd/fix_robot_usd.py
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

USDZ = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/robots/roscoe_robot.usdz"
OUT  = "/home/ad.msoe.edu/somed/isaac-sim-msoe/assets/robots/roscoe_robot.usda"

stage = Usd.Stage.Open(USDZ)
children = list(stage.GetPseudoRoot().GetChildren())
print(f"Root prims in USDZ: {[str(p.GetPath()) for p in children]}")

if not children:
    print("ERROR: no prims found in USDZ")
    simulation_app.close()
    exit(1)

root_prim_name = children[0].GetName()
root_prim_path = str(children[0].GetPath())
print(f"Using root prim: {root_prim_path}")

# Write wrapper USDA
usda = f'''#usda 1.0
(
    defaultPrim = "{root_prim_name}"
    upAxis = "Z"
)

def Xform "{root_prim_name}" (
    references = @{USDZ}@<{root_prim_path}>
)
{{
}}
'''

with open(OUT, "w") as f:
    f.write(usda)

print(f"Written: {OUT}")
print(f"Now reference this file instead of the .usdz")
simulation_app.close()
