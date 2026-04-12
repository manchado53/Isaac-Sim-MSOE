"""
Helpers for loading building metadata from assets/environments/<Building>/building_info.json.

Usage:
    from sim_utils.building_info import get_floor, lookup_node

    floor = get_floor("InnovationCenter", 4)
    print(floor["floor_deck_z"], floor["map_metadata_yaml"])

    x, y = lookup_node(floor["nav_graph_json"], node_id=77)
"""

import json
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_ROOT = REPO_ROOT / "assets" / "environments"


def load_building(name: str) -> dict:
    """Load building_info.json for a building.

    Args:
        name: Building directory name under assets/environments/ (e.g. "InnovationCenter").

    Returns:
        Parsed JSON dict with an added `_root` key pointing to the building directory.
    """
    root = ENVS_ROOT / name
    info_path = root / "building_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No building_info.json at {info_path}")
    info = json.loads(info_path.read_text())
    info["_root"] = root
    return info


def get_floor(building_name: str, floor_num: int) -> dict:
    """Get one floor's metadata with all paths resolved to absolute filesystem paths.

    Args:
        building_name: e.g. "InnovationCenter".
        floor_num: e.g. 4.

    Returns:
        A copy of the floor dict from building_info.json with:
        - `occupancy_png`, `map_metadata_yaml`, `nav_graph_json` resolved to absolute paths
        - `_building_root` — absolute path to the building directory
        - `_usd_physics` — absolute path to the physics USD file
    """
    building = load_building(building_name)
    floor_key = str(floor_num)
    if floor_key not in building["floors"]:
        raise KeyError(
            f"Floor {floor_num} not found in {building_name}. "
            f"Available: {sorted(building['floors'].keys())}"
        )

    floor = dict(building["floors"][floor_key])
    root = building["_root"]

    for key in ("occupancy_png", "map_metadata_yaml", "nav_graph_json"):
        if key in floor and floor[key]:
            floor[key] = str((root / floor[key]).resolve())

    floor["_building_root"] = str(root)
    floor["_usd_physics"] = str((root / building["usd"]["physics"]).resolve())
    return floor


def lookup_node(nav_graph_path: str, node_id: int) -> Tuple[float, float]:
    """Look up a node's world (x, y) coordinates in an auto_navgraph JSON file.

    The JSON stores world coordinates that already match the USD frame, as of
    the `auto_graph_generator.py` fix that aligned `pixel_to_world` with
    `isaacsim.asset.gen.omap.utils::compute_coordinates` (col 0 = x_max,
    row 0 = y_min). No read-time compensation needed.

    Args:
        nav_graph_path: Absolute path to a floor_N_auto_navgraph.json file.
        node_id: Integer node ID.

    Returns:
        Tuple of (x, y) in meters, matching the USD world frame.

    Raises:
        ValueError: If the node ID is not present in the graph.
    """
    graph = json.loads(Path(nav_graph_path).read_text())
    for node in graph["nodes"]:
        if node["id"] == node_id:
            x, y = node["world_xy_m"]
            return (float(x), float(y))
    raise ValueError(f"Node id {node_id} not found in {nav_graph_path}")
