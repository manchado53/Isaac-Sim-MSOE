"""
generate_routes.py

Loads floor1_nodes.json into the navigation Graph, runs Dijkstra's on every
task defined in navigation/tasks.json, and writes the solved routes to
navigation/routes.json.

Usage (no Isaac Sim needed):
    python3 generate_routes.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from navigation.graph import Graph, NodeType

REPO_ROOT  = os.path.dirname(__file__)
GRAPH_FILE = os.path.join(REPO_ROOT, "floor_plans_resized", "floor1", "floor1_nodes.json")
TASKS_FILE = os.path.join(REPO_ROOT, "navigation", "tasks.json")
OUTPUT_FILE = os.path.join(REPO_ROOT, "navigation", "routes.json")

TYPE_MAP = {
    "room":         NodeType.ROOM,
    "hallway":      NodeType.HALLWAY,
    "intersection": NodeType.INTERSECTION,
    "stairwell":    NodeType.STAIRWELL,
    "elevator":     NodeType.ELEVATOR,
    "entrance":     NodeType.ENTRANCE,
}


def load_floor1_graph(path: str) -> Graph:
    with open(path) as f:
        data = json.load(f)
    graph = Graph()
    for node in data["nodes"]:
        graph.add_node(
            id=node["id"],
            name=node["name"],
            position=np.array([node["x"], node["y"], 0.0]),
            floor=1,
            node_type=TYPE_MAP.get(node["type"], NodeType.HALLWAY),
        )
    for edge in data["edges"]:
        try:
            graph.add_edge(edge[0], edge[1])
        except ValueError as e:
            print(f"  [WARN] Skipping edge {edge}: {e}")
    return graph


def solve_tasks(graph: Graph, tasks_path: str) -> dict:
    with open(tasks_path) as f:
        tasks_data = json.load(f)

    solved = 0
    failed = 0

    for task in tasks_data["tasks"]:
        tid, start, goal = task["id"], task["start"], task["goal"]

        if graph.get_node(start) is None:
            print(f"  [FAIL] {tid}: start node '{start}' not in graph")
            task["waypoints"] = None
            task["total_distance"] = None
            failed += 1
            continue

        if graph.get_node(goal) is None:
            print(f"  [FAIL] {tid}: goal node '{goal}' not in graph")
            task["waypoints"] = None
            task["total_distance"] = None
            failed += 1
            continue

        path, distance = graph.dijkstra(start, goal)

        if path is None:
            print(f"  [FAIL] {tid} ({task['difficulty']:6s}): no path  '{start}' -> '{goal}'")
            task["waypoints"] = None
            task["total_distance"] = None
            failed += 1
        else:
            task["waypoints"] = path
            task["total_distance"] = round(distance, 2)
            print(f"  [OK]   {tid} ({task['difficulty']:6s}): {len(path)} waypoints, distance={distance:.1f}px")
            solved += 1

    print(f"\n  Solved: {solved}/{solved + failed} tasks")
    return tasks_data


def main():
    print(f"Loading graph:  {GRAPH_FILE}")
    graph = load_floor1_graph(GRAPH_FILE)
    print(f"  {graph}\n")

    print(f"Running Dijkstra on tasks in:  {TASKS_FILE}\n")
    routes = solve_tasks(graph, TASKS_FILE)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(routes, f, indent=2)
    print(f"\nRoutes written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
