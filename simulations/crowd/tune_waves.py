"""Inspect and tune Warp ocean wave parameters via OmniGraph attributes.

exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/tune_waves.py').read())
"""
import omni.usd
import omni.graph.core as og

stage = omni.usd.get_context().get_stage()

# Dump all OmniGraph node attributes in /World/ActionGraph
graph = og.get_graph_by_path("/World/ActionGraph")
if graph is None:
    # Try alternate path used by ocean.usda
    for prim in stage.Traverse():
        if "ActionGraph" in str(prim.GetPath()) or "OmniGraph" in str(prim.GetPath()):
            print(f"Found graph prim: {prim.GetPath()} [{prim.GetTypeName()}]")

    print("\nSearching for Warp node attributes...")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "warp" in path.lower() or "ocean" in path.lower() or "wave" in path.lower():
            print(f"\n{path} [{prim.GetTypeName()}]")
            for attr in prim.GetAttributes():
                val = attr.Get()
                if val is not None:
                    print(f"  {attr.GetName()} = {val}")
else:
    print(f"Graph: {graph.get_path_to_graph()}")
    for node in graph.get_nodes():
        node_type = node.get_type_name()
        if "warp" in node_type.lower() or "ocean" in node_type.lower() or "wave" in node_type.lower():
            print(f"\nNode: {node.get_prim_path()}  type={node_type}")
            for attr in node.get_attributes():
                try:
                    val = attr.get()
                    print(f"  {attr.get_name()} = {val}")
                except:
                    pass

# Also check MeshOut scale as a proxy for wave height
mesh_out = stage.GetPrimAtPath("/World/MeshOut")
if mesh_out.IsValid():
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(mesh_out)
    print(f"\nMeshOut xform ops:")
    for op in xf.GetOrderedXformOps():
        print(f"  {op.GetOpName()} = {op.Get()}")
