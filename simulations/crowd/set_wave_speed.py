"""Boost Warp ocean wave animation speed via OmniGraph time scale.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/set_wave_speed.py').read())
"""
import omni.usd
import omni.graph.core as og
from pxr import Usd

stage = omni.usd.get_context().get_stage()

WAVE_SPEED = 3.0  # multiplier — 1.0=default, 3.0=3x faster

found = False
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "ActionGraph" not in path:
        continue
    for attr in prim.GetAttributes():
        name = attr.GetName().lower()
        if any(k in name for k in ["speed", "time_scale", "timescale", "frequency", "freq"]):
            print(f"Found: {path}.{attr.GetName()} = {attr.Get()}")
            try:
                attr.Set(WAVE_SPEED)
                print(f"  -> set to {WAVE_SPEED}")
                found = True
            except Exception as e:
                print(f"  -> failed: {e}")

# Also try via OmniGraph API
try:
    graph = og.get_graph_by_path("/World/ActionGraph")
    if graph:
        for node in graph.get_nodes():
            for attr in node.get_attributes():
                n = attr.get_name().lower()
                if any(k in n for k in ["speed", "time_scale", "frequency"]):
                    print(f"OmniGraph attr: {node.get_prim_path()}.{attr.get_name()} = {attr.get()}")
                    try:
                        attr.set(WAVE_SPEED)
                        print(f"  -> set to {WAVE_SPEED}")
                        found = True
                    except Exception as e:
                        print(f"  -> failed: {e}")
except Exception as e:
    print(f"OmniGraph API error: {e}")

if not found:
    print("No speed attribute found — dumping all ActionGraph node attrs:")
    try:
        graph = og.get_graph_by_path("/World/ActionGraph")
        if graph:
            for node in graph.get_nodes():
                print(f"\nNode: {node.get_prim_path()} [{node.get_type_name()}]")
                for attr in node.get_attributes():
                    try:
                        print(f"  {attr.get_name()} = {attr.get()}")
                    except:
                        pass
    except Exception as e:
        print(f"Error: {e}")
