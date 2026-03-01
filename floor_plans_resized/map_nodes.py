#!/usr/bin/env python3
"""
Map nodes onto a floor plan image and visualize the graph.

Reads nodes and edges from a JSON file, draws them on the floor plan,
runs Dijkstra's if start/end are specified, and saves the result.

Usage:
    # Just draw nodes and edges:
    ~/isaacsim/python.sh floor_plans_resized/map_nodes.py floor_plans_resized/floor1.jpg

    # Draw nodes/edges + shortest path:
    ~/isaacsim/python.sh floor_plans_resized/map_nodes.py floor_plans_resized/floor1.jpg --path "Main Entrance" "Room 103"

Input:  floor1_nodes.json (nodes and edges data)
Output: floor1_mapped.jpg (floor plan with graph overlay)

JSON format:
{
  "nodes": [
    {"id": "f1_entrance", "name": "Main Entrance", "x": 400, "y": 600, "type": "entrance"},
    {"id": "f1_hall_1", "name": "Main Hall", "x": 500, "y": 600, "type": "hallway"},
    ...
  ],
  "edges": [
    ["f1_entrance", "f1_hall_1"],
    ["f1_hall_1", "f1_room_101"],
    ...
  ]
}
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Add project root for navigation imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _project_dir)

from navigation.graph import Graph, NodeType

# Color scheme matching the visualizer
TYPE_COLORS = {
    "room": "blue",
    "hallway": "white",
    "intersection": "yellow",
    "stairwell": "red",
    "elevator": "purple",
    "entrance": "green",
}

TYPE_MAP = {
    "room": NodeType.ROOM,
    "hallway": NodeType.HALLWAY,
    "intersection": NodeType.INTERSECTION,
    "stairwell": NodeType.STAIRWELL,
    "elevator": NodeType.ELEVATOR,
    "entrance": NodeType.ENTRANCE,
}


def load_graph_from_json(json_path):
    """Load nodes and edges from JSON, build a Graph for pathfinding."""
    with open(json_path) as f:
        data = json.load(f)

    graph = Graph()
    node_positions = {}  # id -> (x, y) pixel coords

    for node in data.get("nodes", []):
        node_id = node["id"]
        # Use pixel coords as the position (z=0 since it's 2D)
        pos = np.array([node["x"], node["y"], 0.0])
        node_type = TYPE_MAP.get(node.get("type", "hallway"), NodeType.HALLWAY)
        graph.add_node(node_id, node["name"], pos, floor=node.get("floor", 1), node_type=node_type)
        node_positions[node_id] = (node["x"], node["y"])

    for edge in data.get("edges", []):
        graph.add_edge(edge[0], edge[1])

    return graph, node_positions, data


def draw_floor_plan(image_path, graph, node_positions, path=None, output_path=None):
    """Draw the graph on top of the floor plan image."""
    img = mpimg.imread(image_path)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img)

    # Draw edges as gray lines
    for edge in graph.get_all_edges():
        if edge.start_node_id in node_positions and edge.end_node_id in node_positions:
            sx, sy = node_positions[edge.start_node_id]
            ex, ey = node_positions[edge.end_node_id]
            ax.plot([sx, ex], [sy, ey], color="gray", linewidth=1.5, alpha=0.6)

    # Draw path if provided (thick green line)
    if path and len(path) >= 2:
        for i in range(len(path) - 1):
            if path[i] in node_positions and path[i + 1] in node_positions:
                sx, sy = node_positions[path[i]]
                ex, ey = node_positions[path[i + 1]]
                ax.plot([sx, ex], [sy, ey], color="lime", linewidth=3, alpha=0.9, zorder=5)

        # Start marker
        if path[0] in node_positions:
            sx, sy = node_positions[path[0]]
            ax.plot(sx, sy, "o", color="lime", markersize=14, zorder=6)
            ax.annotate("START", (sx, sy), textcoords="offset points", xytext=(8, 8),
                       fontsize=8, color="lime", fontweight="bold", zorder=6)

        # End marker
        if path[-1] in node_positions:
            ex, ey = node_positions[path[-1]]
            ax.plot(ex, ey, "o", color="red", markersize=14, zorder=6)
            ax.annotate("END", (ex, ey), textcoords="offset points", xytext=(8, 8),
                       fontsize=8, color="red", fontweight="bold", zorder=6)

    # Draw nodes as colored dots
    for node in graph.get_all_nodes():
        if node.id in node_positions:
            x, y = node_positions[node.id]
            color = TYPE_COLORS.get(node.node_type.value, "white")
            ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="black",
                   markeredgewidth=0.5, zorder=4)
            ax.annotate(node.name, (x, y),
                       textcoords="offset points", xytext=(6, 6),
                       fontsize=5, color="white", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                       zorder=4)

    title = f"Floor Plan — {len(node_positions)} nodes, {len(graph.get_all_edges())} edges"
    if path:
        start_name = graph.get_node(path[0]).name if graph.get_node(path[0]) else path[0]
        end_name = graph.get_node(path[-1]).name if graph.get_node(path[-1]) else path[-1]
        title += f"\nPath: {start_name} → {end_name}"
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Legend
    legend_items = [
        ("Room", "blue"),
        ("Hallway", "white"),
        ("Intersection", "yellow"),
        ("Stairwell", "red"),
        ("Elevator", "purple"),
        ("Entrance", "green"),
    ]
    if path:
        legend_items.append(("Path", "lime"))

    legend_x = 0.02
    legend_y = 0.95
    line_height = 0.035
    fig.text(legend_x + 0.005, legend_y, "Legend", fontsize=9, color="black",
             fontweight="bold", transform=fig.transFigure, va="top", zorder=11)
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - (i + 1) * line_height
        ax_dot = fig.add_axes([legend_x + 0.005, y_pos - 0.006, 0.012, 0.012])
        ax_dot.set_xlim(0, 1)
        ax_dot.set_ylim(0, 1)
        ax_dot.plot(0.5, 0.5, "o", color=color, markersize=7, markeredgecolor="black",
                   markeredgewidth=0.5)
        ax_dot.axis("off")
        fig.text(legend_x + 0.025, y_pos, label, fontsize=8, color="black",
                transform=fig.transFigure, va="center", zorder=11)

    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "_mapped.jpg"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: ~/isaacsim/python.sh map_nodes.py <floor_plan.jpg> [--path 'Start' 'End']")
        sys.exit(1)

    image_path = sys.argv[1]
    json_path = os.path.splitext(image_path)[0] + "_nodes.json"

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(json_path):
        print(f"No nodes file found at: {json_path}")
        print(f"Create it with this format:")
        print(json.dumps({
            "nodes": [
                {"id": "f1_entrance", "name": "Main Entrance", "x": 400, "y": 600, "type": "entrance"},
                {"id": "f1_hall_1", "name": "Hall 1", "x": 500, "y": 600, "type": "hallway"},
            ],
            "edges": [
                ["f1_entrance", "f1_hall_1"],
            ]
        }, indent=2))
        sys.exit(1)

    graph, node_positions, data = load_graph_from_json(json_path)
    print(f"Loaded: {len(node_positions)} nodes, {len(graph.get_all_edges())} edges")

    # Check for --path argument
    path_result = None
    if "--path" in sys.argv:
        idx = sys.argv.index("--path")
        if idx + 2 < len(sys.argv):
            start_name = sys.argv[idx + 1]
            end_name = sys.argv[idx + 2]

            # Find node IDs by name
            start_id = None
            end_id = None
            for node in graph.get_all_nodes():
                if node.name == start_name or node.id == start_name:
                    start_id = node.id
                if node.name == end_name or node.id == end_name:
                    end_id = node.id

            if start_id and end_id:
                path_result, distance = graph.dijkstra(start_id, end_id)
                if path_result:
                    print(f"Shortest path: {' -> '.join(path_result)}")
                    print(f"Distance: {distance:.1f} pixels")
                else:
                    print(f"No path found from '{start_name}' to '{end_name}'")
            else:
                if not start_id:
                    print(f"Node not found: '{start_name}'")
                if not end_id:
                    print(f"Node not found: '{end_name}'")

    output = draw_floor_plan(image_path, graph, node_positions, path=path_result)


if __name__ == "__main__":
    main()
