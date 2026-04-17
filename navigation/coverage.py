"""
navigation/coverage.py — Callable full-floor coverage route generator.

Exposes generate_route() which takes a parsed nav graph dict and a start
node ID, then returns an optimized full-path waypoint list using:
  - All-pairs Dijkstra (shortest inter-node distances)
  - Nearest-neighbor greedy TSP heuristic
  - 2-opt local search improvement
  - Multi-restart (try multiple start candidates, keep the best tour)
  - Path expansion (fill Dijkstra hops between each consecutive TSP node)

Usage:
    import json
    from navigation.coverage import generate_route

    with open("assets/Graph/floor_4_nav_graph_world_xflipped.json") as f:
        graph_data = json.load(f)

    full_path = generate_route(graph_data, start_node_id="node_12")
    # full_path is a list of node IDs covering every reachable node,
    # including intermediate corridor hops between TSP-ordered nodes.
"""

import heapq
import random
from typing import Optional


def generate_route(
    graph_data: dict,
    start_node_id: str,
    num_restarts: int = 10,
    seed: Optional[int] = None,
) -> list:
    """
    Generate an optimized full-floor coverage route from any start node.

    Args:
        graph_data:    Parsed nav graph JSON dict with 'nodes' and 'edges' keys.
                       Node dicts must have an 'id' field.
                       Edge dicts must have 'from', 'to', and 'distance' fields.
        start_node_id: Node ID to start the route from.  Should be the node
                       nearest to the robot's actual spawn position.
        num_restarts:  Number of random start candidates to try in the
                       multi-restart phase.  Higher = better tour quality
                       but slower.  Default 10 is a good balance on 79 nodes.
        seed:          Optional RNG seed for reproducible restart sampling.

    Returns:
        full_path: List of node ID strings — the complete ordered waypoint
                   sequence including intermediate Dijkstra hops between
                   TSP-ordered nodes.  Feed this directly to the robot
                   controller by looking up each node's XY position.

    Raises:
        ValueError: If start_node_id is not found in graph_data.
    """
    if seed is not None:
        random.seed(seed)

    # ------------------------------------------------------------------
    # Build adjacency list from graph_data
    # ------------------------------------------------------------------
    nodes = {n["id"]: n for n in graph_data["nodes"]}

    if start_node_id not in nodes:
        raise ValueError(
            f"start_node_id '{start_node_id}' not found in graph_data. "
            f"Available: {list(nodes.keys())[:5]} ..."
        )

    adj = {nid: [] for nid in nodes}
    for e in graph_data["edges"]:
        a, b = e["from"], e["to"]
        # Support both field names used across different graph JSON exports
        d = e.get("distance", e.get("distance_cm", 1.0))
        adj[a].append((b, d))
        adj[b].append((a, d))

    # ------------------------------------------------------------------
    # All-pairs Dijkstra — build distance + predecessor tables
    # ------------------------------------------------------------------

    def _dijkstra(start):
        dist = {n: float("inf") for n in nodes}
        prev = {n: None for n in nodes}
        dist[start] = 0.0
        pq = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # Compute reachability from start node first (cheap filter)
    start_dist, _ = _dijkstra(start_node_id)
    reachable = [n for n in nodes if start_dist[n] < float("inf")]

    # All-pairs over reachable nodes only
    all_dist = {start_node_id: start_dist}
    all_prev = {}
    _, all_prev[start_node_id] = _dijkstra(start_node_id)
    for nid in reachable:
        if nid != start_node_id:
            all_dist[nid], all_prev[nid] = _dijkstra(nid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct(prev, end):
        path, cur = [], end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def _tour_distance(order):
        return sum(all_dist[order[i]][order[i + 1]] for i in range(len(order) - 1))

    def _greedy(start):
        unvisited = set(reachable) - {start}
        order = [start]
        cur = start
        while unvisited:
            best = min(unvisited, key=lambda n: all_dist[cur][n])
            if all_dist[cur][best] == float("inf"):
                break
            order.append(best)
            unvisited.discard(best)
            cur = best
        return order

    def _two_opt(order):
        best = order[:]
        n = len(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    j_next = (j + 1) % n
                    before = (all_dist[best[i - 1]][best[i]] +
                              all_dist[best[j]][best[j_next]])
                    after  = (all_dist[best[i - 1]][best[j]] +
                              all_dist[best[i]][best[j_next]])
                    if after < before - 1e-9:
                        best[i:j + 1] = best[i:j + 1][::-1]
                        improved = True
        return best

    # ------------------------------------------------------------------
    # Build tour always starting from start_node_id, then improve with 2-opt.
    # Multi-restart tries additional random candidates but only keeps a result
    # if it can be rotated to begin at start_node_id (preserving the constraint
    # that the robot starts where it spawns).
    # ------------------------------------------------------------------
    best_order = _two_opt(_greedy(start_node_id))
    best_dist  = _tour_distance(best_order)

    candidates = random.sample(
        [n for n in reachable if n != start_node_id],
        min(num_restarts - 1, len(reachable) - 1),
    )
    for start in candidates:
        order = _two_opt(_greedy(start))
        # Rotate so the tour begins at start_node_id
        if start_node_id in order:
            idx   = order.index(start_node_id)
            order = order[idx:] + order[:idx]
        dist = _tour_distance(order)
        if dist < best_dist:
            best_order, best_dist = order, dist

    # ------------------------------------------------------------------
    # Expand TSP visit order → full waypoint path with Dijkstra hops
    # ------------------------------------------------------------------
    full_path = [best_order[0]]
    for i in range(1, len(best_order)):
        seg = _reconstruct(all_prev[best_order[i - 1]], best_order[i])
        full_path.extend(seg[1:])

    return full_path


def find_nearest_node(graph_data: dict, xy_m) -> str:
    """
    Return the node ID whose world position is closest to xy_m (in metres).

    Args:
        graph_data: Parsed nav graph JSON dict (nodes have 'world_xy_cm').
        xy_m:       [x, y] position in metres (e.g. robot spawn XY).

    Returns:
        Node ID string of the nearest node.
    """
    import numpy as np
    target = np.array(xy_m, dtype=float)
    best_id, best_dist = None, float("inf")
    for n in graph_data["nodes"]:
        pos = np.array([n["world_xy_cm"][0] / 100.0,
                        n["world_xy_cm"][1] / 100.0])
        d = float(np.linalg.norm(pos - target))
        if d < best_dist:
            best_dist, best_id = d, n["id"]
    return best_id
