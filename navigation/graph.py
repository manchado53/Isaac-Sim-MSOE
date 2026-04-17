"""
Core data structures and pathfinding for indoor navigation.

This module defines the building blocks of the navigation system:

- NodeType: Enum classifying what kind of location a node represents
  (room, hallway, stairwell, etc.). Used for color-coding in visualization
  and filtering nodes by category.

- Node: A location in the building. Has a position (3D coordinate),
  a floor number, and a type. Think of it as a pin on a map.

- Edge: A walkable connection between two nodes. The weight is the
  travel cost (defaults to straight-line distance between the nodes,
  but can be overridden for stairs, elevators, etc.).

- Graph: Holds all nodes and edges. Provides Dijkstra's algorithm
  to find the shortest path between any two nodes.

Nodes are the locations. Edges are the paths between them.
Example:
    [Room 101] --- [Hallway Junction] --- [Room 102]
                          |
                      [Stairwell]

    4 nodes, 3 edges. Dijkstra's finds the shortest route.
"""

import heapq
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class NodeType(Enum):
    """
    Classifies what kind of location a node represents.

    Each type gets a different color in the visualization:
        ROOM         -> Blue    (classrooms, offices, labs)
        HALLWAY      -> White   (straight hallway segments)
        INTERSECTION -> Yellow  (where multiple corridors meet)
        STAIRWELL    -> Red     (connects floors via stairs)
        ELEVATOR     -> Purple  (connects floors via elevator)
        ENTRANCE     -> Green   (building entrance/exit)
    """

    ROOM = "room"
    HALLWAY = "hallway"
    INTERSECTION = "intersection"
    STAIRWELL = "stairwell"
    ELEVATOR = "elevator"
    ENTRANCE = "entrance"


class Node:
    """
    A location in the building.

    A node is a point in 3D space representing a key location:
    a room, a hallway junction, a stairwell, etc. It holds a position
    and metadata but no logic — it's just a data container.

    Attributes:
        id: Unique string identifier (e.g., "f1_room_101").
            Prefixed with floor number to avoid collisions across floors.
        name: Human-readable label (e.g., "Room 101"). For display only.
        position: 3D coordinate as np.array([x, y, z]) matching the
            building USD scene.
        floor: Integer floor number (1, 2, 3...).
        node_type: NodeType enum classifying this location.
    """

    def __init__(
        self,
        id: str,
        name: str,
        position: np.ndarray,
        floor: int,
        node_type: NodeType,
    ):
        self.id = id
        self.name = name
        self.position = np.array(position, dtype=float)
        self.floor = floor
        self.node_type = node_type

    def __repr__(self):
        return f"Node('{self.id}', '{self.name}', floor={self.floor}, type={self.node_type.value})"


class Edge:
    """
    A walkable connection between two nodes.

    An edge says "you can travel from node A to node B" and how costly
    it is. The weight is typically the Euclidean distance between the
    two node positions, auto-calculated when not provided.

    You can override the weight for special cases:
        - Stairs: higher weight (slower to travel than flat ground)
        - Elevator: even higher weight (wait time)
        - Blocked path: weight = infinity (Dijkstra's will avoid it)

    Attributes:
        start_node_id: Source node ID.
        end_node_id: Destination node ID.
        weight: Travel cost. Auto-calculated from distance if not provided.
        bidirectional: If True (default), the connection works both ways.
    """

    def __init__(
        self,
        start_node_id: str,
        end_node_id: str,
        weight: float,
        bidirectional: bool = True,
    ):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.weight = weight
        self.bidirectional = bidirectional

    def __repr__(self):
        direction = "<->" if self.bidirectional else "->"
        return f"Edge('{self.start_node_id}' {direction} '{self.end_node_id}', weight={self.weight:.2f})"


class Graph:
    """
    The full navigation graph — holds all nodes and edges.

    Internally stores nodes in a dict (id -> Node) and edges in an
    adjacency list (id -> list of (neighbor_id, weight)). Provides
    Dijkstra's algorithm to find the shortest path between any two nodes.

    Usage:
        graph = Graph()
        graph.add_node("f1_entrance", "Entrance", np.array([0,0,0]), 1, NodeType.ENTRANCE)
        graph.add_node("f1_lobby", "Lobby", np.array([5,0,0]), 1, NodeType.HALLWAY)
        graph.add_edge("f1_entrance", "f1_lobby")

        path, distance = graph.dijkstra("f1_entrance", "f1_lobby")
        # path = ["f1_entrance", "f1_lobby"], distance = 5.0

        positions = graph.get_path_positions(path)
        # [np.array([0,0,0]), np.array([5,0,0])]
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._edges: List[Edge] = []

    def add_node(
        self,
        id: str,
        name: str,
        position: np.ndarray,
        floor: int,
        node_type: NodeType,
    ) -> Node:
        """
        Register a new node in the graph.

        Args:
            id: Unique identifier (e.g., "f1_room_101")
            name: Human-readable label (e.g., "Room 101")
            position: 3D coordinate as np.array([x, y, z])
            floor: Floor number (1, 2, 3...)
            node_type: NodeType enum value

        Returns:
            The created Node.

        Raises:
            ValueError: If a node with this ID already exists.
        """
        if id in self._nodes:
            raise ValueError(f"Node '{id}' already exists in the graph")

        node = Node(id, name, position, floor, node_type)
        self._nodes[id] = node
        self._adjacency[id] = []
        return node

    def add_edge(
        self,
        start_id: str,
        end_id: str,
        weight: Optional[float] = None,
        bidirectional: bool = True,
    ) -> Edge:
        """
        Connect two nodes with a walkable path.

        If weight is None, it auto-calculates the Euclidean distance
        between the two node positions. Override for stairs, elevators,
        or blocked paths.

        Args:
            start_id: Source node ID.
            end_id: Destination node ID.
            weight: Travel cost. None = auto-calculate from distance.
            bidirectional: If True (default), connection works both ways.

        Returns:
            The created Edge.

        Raises:
            ValueError: If either node ID doesn't exist.
        """
        if start_id not in self._nodes:
            raise ValueError(f"Node '{start_id}' not found in graph")
        if end_id not in self._nodes:
            raise ValueError(f"Node '{end_id}' not found in graph")

        # Auto-calculate distance if no weight given
        if weight is None:
            start_pos = self._nodes[start_id].position
            end_pos = self._nodes[end_id].position
            weight = float(np.linalg.norm(end_pos - start_pos))

        edge = Edge(start_id, end_id, weight, bidirectional)
        self._edges.append(edge)

        self._adjacency[start_id].append((end_id, weight))
        if bidirectional:
            self._adjacency[end_id].append((start_id, weight))

        return edge

    def get_node(self, id: str) -> Optional[Node]:
        """Return a node by ID, or None if not found."""
        return self._nodes.get(id)

    def get_all_nodes(self) -> List[Node]:
        """Return all nodes in the graph."""
        return list(self._nodes.values())

    def get_all_edges(self) -> List[Edge]:
        """Return all edges in the graph."""
        return list(self._edges)

    def get_neighbors(self, id: str) -> List[Tuple[str, float]]:
        """
        Return connected nodes for a given node.

        Args:
            id: Node ID to get neighbors for.

        Returns:
            List of (neighbor_id, weight) tuples.
        """
        return self._adjacency.get(id, [])

    def get_nodes_on_floor(self, floor: int) -> List[Node]:
        """Return all nodes on a specific floor."""
        return [node for node in self._nodes.values() if node.floor == floor]

    def dijkstra(self, start_id: str, end_id: str) -> Tuple[Optional[List[str]], float]:
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.

        Uses a priority queue to always explore the lowest-cost unvisited
        node first. Tracks the previous node for each visited node so it
        can reconstruct the full path.

        Args:
            start_id: Starting node ID.
            end_id: Destination node ID.

        Returns:
            Tuple of (path, total_distance):
                path: List of node IDs from start to end (e.g.,
                    ["f1_entrance", "f1_lobby", "f1_room_101"])
                    Returns None if no path exists.
                total_distance: Total travel cost along the path.
                    Returns float('inf') if no path exists.

        Raises:
            ValueError: If start or end node doesn't exist.
        """
        if start_id not in self._nodes:
            raise ValueError(f"Start node '{start_id}' not found in graph")
        if end_id not in self._nodes:
            raise ValueError(f"End node '{end_id}' not found in graph")

        # Distance from start to each node (default: infinity)
        distances: Dict[str, float] = {id: float("inf") for id in self._nodes}
        distances[start_id] = 0.0

        # Track the previous node for path reconstruction
        previous: Dict[str, Optional[str]] = {id: None for id in self._nodes}

        # Priority queue: (distance, node_id)
        queue: List[Tuple[float, str]] = [(0.0, start_id)]

        # Set of visited nodes
        visited = set()

        while queue:
            current_distance, current_id = heapq.heappop(queue)

            # Skip if already visited (we found a shorter path earlier)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Reached the destination
            if current_id == end_id:
                break

            # Explore neighbors
            for neighbor_id, edge_weight in self._adjacency[current_id]:
                if neighbor_id in visited:
                    continue

                new_distance = current_distance + edge_weight

                # Found a shorter path to this neighbor
                if new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous[neighbor_id] = current_id
                    heapq.heappush(queue, (new_distance, neighbor_id))

        # No path found
        if distances[end_id] == float("inf"):
            return None, float("inf")

        # Reconstruct path by following previous pointers backwards
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return path, distances[end_id]

    def get_path_positions(self, path: List[str]) -> List[np.ndarray]:
        """
        Convert a list of node IDs (from dijkstra) to 3D coordinates.

        Use this to get the waypoints for a robot to follow.

        Args:
            path: List of node IDs (e.g., ["f1_entrance", "f1_lobby"])

        Returns:
            List of np.array positions (e.g., [np.array([0,0,0]), np.array([5,0,0])])
        """
        return [self._nodes[id].position for id in path]

    def __repr__(self):
        return f"Graph({len(self._nodes)} nodes, {len(self._edges)} edges)"
