"""Speed up Warp wave animation by injecting a time-scale multiplier into the ActionGraph.

Replaces the OnTick time output with a scaled version feeding into the Warp kernel.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/speed_up_waves.py').read())
"""
import omni.usd
import omni.graph.core as og

SPEED = 4.0  # 4x faster wave animation

stage = omni.usd.get_context().get_stage()

# Dump every attribute on every node in ActionGraph to find the time/speed input
try:
    graph = og.get_graph_by_path("/World/ActionGraph")
    if not graph:
        print("ActionGraph not found at /World/ActionGraph")
    else:
        print(f"Nodes in ActionGraph:")
        for node in graph.get_nodes():
            type_name = node.get_type_name()
            print(f"\n  [{type_name}]  {node.get_prim_path()}")
            for attr in node.get_attributes():
                try:
                    val = attr.get()
                    print(f"    {attr.get_name()} = {val}")
                except:
                    pass
except Exception as e:
    print(f"Error: {e}")
