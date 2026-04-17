"""
Hospital layout — node coordinates and edge connections.

This file defines the physical layout of the NVIDIA Isaac Sim Hospital
environment (Isaac/Environments/Hospital/hospital.usd) as a navigation graph.
Each room, hallway, and entrance is a node with a 3D coordinate. Edges
connect walkable paths between nodes.

The hospital is a single-floor environment loaded from NVIDIA's cloud assets.

IMPORTANT: The coordinates below are PLACEHOLDERS. To get the real coordinates:
1. Load the hospital in Isaac Sim streaming mode
2. Navigate to each location in the viewport
3. Read the X, Y, Z from the Property > Transform panel
4. Replace the placeholder values here

Node naming convention:
    Descriptive name prefixed with area: lobby_, hall_, room_, nurse_, etc.

To add a new room or connection:
    1. Add a node with graph.add_node(...)
    2. Add edges to connect it with graph.add_edge(...)
    3. Run the simulation to verify placement
"""

import numpy as np

from .graph import Graph, NodeType


def build_hospital_graph() -> Graph:
    """
    Build and return the hospital navigation graph.

    Returns a Graph populated with all nodes (rooms, hallways, nurse
    stations, etc.) and edges (walkable connections) for the hospital.

    Returns:
        Graph: Fully populated navigation graph.
    """
    graph = Graph()

    # =========================================================================
    # ENTRANCES
    # =========================================================================
    # TODO: Replace placeholder coordinates with real values from hospital.usd

    graph.add_node(
        "entrance_main", "Main Entrance",
        np.array([0.0, 0.0, 0.0]), floor=1, node_type=NodeType.ENTRANCE,
    )

    # =========================================================================
    # LOBBY / RECEPTION
    # =========================================================================

    graph.add_node(
        "lobby", "Lobby",
        np.array([3.0, 0.0, 0.0]), floor=1, node_type=NodeType.HALLWAY,
    )
    graph.add_node(
        "reception", "Reception Desk",
        np.array([5.0, 2.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )

    # =========================================================================
    # MAIN HALLWAYS
    # =========================================================================

    graph.add_node(
        "hall_main_1", "Main Hallway 1",
        np.array([6.0, 0.0, 0.0]), floor=1, node_type=NodeType.HALLWAY,
    )
    graph.add_node(
        "hall_main_2", "Main Hallway 2",
        np.array([12.0, 0.0, 0.0]), floor=1, node_type=NodeType.HALLWAY,
    )
    graph.add_node(
        "hall_main_3", "Main Hallway 3",
        np.array([18.0, 0.0, 0.0]), floor=1, node_type=NodeType.HALLWAY,
    )
    graph.add_node(
        "junction_north", "North Junction",
        np.array([12.0, 6.0, 0.0]), floor=1, node_type=NodeType.INTERSECTION,
    )
    graph.add_node(
        "junction_south", "South Junction",
        np.array([12.0, -6.0, 0.0]), floor=1, node_type=NodeType.INTERSECTION,
    )

    # =========================================================================
    # PATIENT ROOMS
    # =========================================================================

    graph.add_node(
        "room_101", "Patient Room 101",
        np.array([8.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "room_102", "Patient Room 102",
        np.array([10.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "room_103", "Patient Room 103",
        np.array([14.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "room_104", "Patient Room 104",
        np.array([16.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "room_105", "Patient Room 105",
        np.array([8.0, -3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "room_106", "Patient Room 106",
        np.array([10.0, -3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )

    # =========================================================================
    # NURSE STATION / UTILITY
    # =========================================================================

    graph.add_node(
        "nurse_station", "Nurse Station",
        np.array([12.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )
    graph.add_node(
        "supply_room", "Supply Room",
        np.array([18.0, 3.0, 0.0]), floor=1, node_type=NodeType.ROOM,
    )

    # =========================================================================
    # EDGES — Walkable connections
    # =========================================================================

    # Entrance to lobby
    graph.add_edge("entrance_main", "lobby")
    graph.add_edge("lobby", "reception")
    graph.add_edge("lobby", "hall_main_1")

    # Main hallway chain
    graph.add_edge("hall_main_1", "hall_main_2")
    graph.add_edge("hall_main_2", "hall_main_3")

    # Junctions off main hallway
    graph.add_edge("hall_main_2", "junction_north")
    graph.add_edge("hall_main_2", "junction_south")

    # North side rooms (off main hallway and junction)
    graph.add_edge("hall_main_1", "room_101")
    graph.add_edge("hall_main_1", "room_102")
    graph.add_edge("hall_main_2", "nurse_station")
    graph.add_edge("junction_north", "room_103")
    graph.add_edge("junction_north", "room_104")
    graph.add_edge("hall_main_3", "supply_room")

    # South side rooms
    graph.add_edge("junction_south", "room_105")
    graph.add_edge("junction_south", "room_106")

    return graph
