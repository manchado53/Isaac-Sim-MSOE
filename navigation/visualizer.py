"""
Debug draw rendering for the navigation graph in Isaac Sim.

Draws nodes as colored points and edges as lines using Isaac Sim's
debug draw overlay. Everything drawn here is:
    - Visible on screen and in video recordings (the viewer sees the routes)
    - NOT physical objects (the robot doesn't collide with them)
    - NOT detected by robot cameras/sensors (rendering overlay only)

The visualizer has no knowledge of the building or pathfinding —
it just takes a Graph and draws whatever is in it.

Color scheme by NodeType:
    ROOM         -> Blue    (0, 0, 1, 1)
    HALLWAY      -> White   (1, 1, 1, 1)
    INTERSECTION -> Yellow  (1, 1, 0, 1)
    STAIRWELL    -> Red     (1, 0, 0, 1)
    ELEVATOR     -> Purple  (0.5, 0, 1, 1)
    ENTRANCE     -> Green   (0, 1, 0, 1)
"""

from typing import List, Optional

import numpy as np

from .graph import Graph, NodeType


# Color mapping: NodeType -> RGBA tuple
NODE_COLORS = {
    NodeType.ROOM: (0.0, 0.0, 1.0, 1.0),         # Blue
    NodeType.HALLWAY: (1.0, 1.0, 1.0, 1.0),       # White
    NodeType.INTERSECTION: (1.0, 1.0, 0.0, 1.0),  # Yellow
    NodeType.STAIRWELL: (1.0, 0.0, 0.0, 1.0),     # Red
    NodeType.ELEVATOR: (0.5, 0.0, 1.0, 1.0),       # Purple
    NodeType.ENTRANCE: (0.0, 1.0, 0.0, 1.0),       # Green
}

# Point sizes: larger for important nodes, smaller for hallways
NODE_SIZES = {
    NodeType.ROOM: 15.0,
    NodeType.HALLWAY: 10.0,
    NodeType.INTERSECTION: 12.0,
    NodeType.STAIRWELL: 18.0,
    NodeType.ELEVATOR: 18.0,
    NodeType.ENTRANCE: 20.0,
}

# Edge appearance
EDGE_COLOR = (0.5, 0.5, 0.5, 1.0)  # Gray
EDGE_THICKNESS = 2.0

# Path appearance (for highlighted Dijkstra result)
PATH_COLOR = (0.0, 1.0, 0.0, 1.0)   # Bright green
PATH_THICKNESS = 5.0
START_COLOR = (0.0, 1.0, 0.0, 1.0)  # Green
END_COLOR = (1.0, 0.0, 0.0, 1.0)    # Red
PATH_NODE_SIZE = 25.0


def _get_draw_interface():
    """Acquire the Isaac Sim debug draw interface."""
    from isaacsim.util.debug_draw import _debug_draw
    return _debug_draw.acquire_debug_draw_interface()


def draw_nodes(graph: Graph, floor: Optional[int] = None):
    """
    Draw every node as a colored point at its position.

    Color is determined by the node's NodeType. Useful on its own
    for checking that coordinates are in the right spots before
    connecting edges.

    Args:
        graph: The navigation graph to draw nodes from.
        floor: If specified, only draw nodes on this floor.
            None draws all floors.
    """
    draw = _get_draw_interface()

    nodes = graph.get_nodes_on_floor(floor) if floor is not None else graph.get_all_nodes()

    points = []
    colors = []
    sizes = []

    for node in nodes:
        points.append(node.position.tolist())
        colors.append(NODE_COLORS.get(node.node_type, (1.0, 1.0, 1.0, 1.0)))
        sizes.append(NODE_SIZES.get(node.node_type, 10.0))

    if points:
        draw.draw_points(points, colors, sizes)


def draw_edges(graph: Graph, floor: Optional[int] = None):
    """
    Draw every edge as a thin gray line between connected nodes.

    If floor is specified, only draws edges where at least one node
    is on that floor (so cross-floor stairwell edges still show up).

    Args:
        graph: The navigation graph to draw edges from.
        floor: If specified, only draw edges involving this floor.
            None draws all edges.
    """
    draw = _get_draw_interface()

    starts = []
    ends = []
    colors = []
    sizes = []

    for edge in graph.get_all_edges():
        start_node = graph.get_node(edge.start_node_id)
        end_node = graph.get_node(edge.end_node_id)

        if start_node is None or end_node is None:
            continue

        # Filter by floor if specified
        if floor is not None:
            if start_node.floor != floor and end_node.floor != floor:
                continue

        starts.append(start_node.position.tolist())
        ends.append(end_node.position.tolist())
        colors.append(EDGE_COLOR)
        sizes.append(EDGE_THICKNESS)

    if starts:
        draw.draw_lines(starts, ends, colors, sizes)


def draw_graph(graph: Graph, floor: Optional[int] = None):
    """
    Draw the full navigation graph — edges first, then nodes on top.

    This shows the complete building map: gray lines for all walkable
    connections, colored dots for each location.

    Args:
        graph: The navigation graph to draw.
        floor: If specified, only draw this floor. None draws everything.
    """
    draw_edges(graph, floor)
    draw_nodes(graph, floor)


def draw_path(graph: Graph, path: List[str]):
    """
    Draw a Dijkstra result as a highlighted route.

    Draws thick bright green lines along the path, with a green dot
    at the start and a red dot at the destination. Call this AFTER
    draw_graph() so the route appears on top of the gray map.

    Args:
        graph: The navigation graph (for looking up node positions).
        path: List of node IDs from dijkstra() (e.g.,
            ["f1_entrance", "f1_lobby", "f1_room_101"]).
    """
    if not path or len(path) < 2:
        return

    draw = _get_draw_interface()

    # Draw path lines
    starts = []
    ends = []
    colors = []
    sizes = []

    for i in range(len(path) - 1):
        start_node = graph.get_node(path[i])
        end_node = graph.get_node(path[i + 1])

        if start_node is None or end_node is None:
            continue

        starts.append(start_node.position.tolist())
        ends.append(end_node.position.tolist())
        colors.append(PATH_COLOR)
        sizes.append(PATH_THICKNESS)

    if starts:
        draw.draw_lines(starts, ends, colors, sizes)

    # Draw start point (green) and end point (red)
    start_node = graph.get_node(path[0])
    end_node = graph.get_node(path[-1])

    points = []
    point_colors = []
    point_sizes = []

    if start_node is not None:
        points.append(start_node.position.tolist())
        point_colors.append(START_COLOR)
        point_sizes.append(PATH_NODE_SIZE)

    if end_node is not None:
        points.append(end_node.position.tolist())
        point_colors.append(END_COLOR)
        point_sizes.append(PATH_NODE_SIZE)

    if points:
        draw.draw_points(points, point_colors, point_sizes)


def clear_all():
    """
    Clear all debug draw lines and points.

    Call this before redrawing if you want to update the visualization
    (e.g., after recalculating a path to a new destination).
    """
    draw = _get_draw_interface()
    draw.clear_lines()
    draw.clear_points()
