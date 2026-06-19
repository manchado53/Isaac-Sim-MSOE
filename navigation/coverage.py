"""
navigation/coverage.py — TSP-based coverage route generators.

Two public entry points:

  generate_route(graph_data, start_node_id, ...)
      TSP over every reachable node in the graph.  Output is a fully
      Dijkstra-expanded path covering the whole floor.  Use for "patrol
      every node" / training-data runs.

  generate_door_route(graph_data, start_node_id, door_node_ids, ...)
      TSP over only the supplied door subset (plus the spawn node).
      Dijkstra runs on the *full* graph so door-to-door distances are
      true walking distances.  Output is the Dijkstra-expanded path
      between consecutive doors.  Use for door-coverage demos.

Both functions use the same algorithmic stages — they only differ in which
nodes form the TSP visit set:

  Stage 1: _compute_all_pairs   — all-pairs Dijkstra on the full graph
  Stage 2: _run_tsp             — nearest-neighbor + 2-opt + multi-restart
  Stage 3: _expand_tour         — fill Dijkstra hops between TSP-ordered nodes

Usage:
    import json
    from navigation.coverage import generate_route, generate_door_route, find_nearest_node

    with open("assets/.../floor_4_auto_navgraph.json") as f:
        graph = json.load(f)

    start = find_nearest_node(graph, robot_spawn_xy_m)

    # Coverage mode — visit every node
    full_path = generate_route(graph, start)

    # Door mode — visit only the supplied door nodes
    with open("assets/.../floor_4_door_node_map.json") as f:
        dm = json.load(f)
    door_ids = [m["closest_node_id"] for m in dm["mappings"]
                if m["closest_node_id"] is not None]
    door_path = generate_door_route(graph, start, door_ids)
"""

import heapq
import random
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Stage 1 — graph + Dijkstra
# ---------------------------------------------------------------------------

def _build_adj(graph_data: dict):
    """Return (nodes_dict, adjacency_dict) for the parsed nav-graph JSON.

    Edges are read from `distance` first, then `distance_m`, then
    `distance_cm` — supports both the legacy and auto_navgraph schemas
    without a per-caller shim.
    """
    nodes = {n["id"]: n for n in graph_data["nodes"]}
    adj = {nid: [] for nid in nodes}
    for e in graph_data["edges"]:
        a, b = e["from"], e["to"]
        d = e.get("distance", e.get("distance_m", e.get("distance_cm", 1.0)))
        adj[a].append((b, d))
        adj[b].append((a, d))
    return nodes, adj


def _dijkstra(start, nodes, adj):
    """Standard Dijkstra from `start`. Returns (dist, prev) dicts keyed by node id."""
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


def _compute_all_pairs(graph_data: dict, start_node_id):
    """Run Dijkstra from `start_node_id` plus every other reachable node.

    Returns:
        all_dist:  {src_id: {dst_id: distance}}
        all_prev:  {src_id: {dst_id: predecessor_on_shortest_path}}
        reachable: set of node ids reachable from start_node_id
    """
    nodes, adj = _build_adj(graph_data)
    if start_node_id not in nodes:
        raise ValueError(
            f"start_node_id {start_node_id!r} not found in graph "
            f"(have {len(nodes)} nodes; sample: {list(nodes)[:5]} ...)"
        )

    start_dist, start_prev = _dijkstra(start_node_id, nodes, adj)
    reachable = {n for n, d in start_dist.items() if d < float("inf")}

    all_dist = {start_node_id: start_dist}
    all_prev = {start_node_id: start_prev}
    for nid in reachable:
        if nid != start_node_id:
            all_dist[nid], all_prev[nid] = _dijkstra(nid, nodes, adj)

    return all_dist, all_prev, reachable


# ---------------------------------------------------------------------------
# Stage 2 — TSP (nearest-neighbor + 2-opt + multi-restart)
# ---------------------------------------------------------------------------

def _run_tsp(visit_set, start_node_id, all_dist, num_restarts: int = 10,
             seed: Optional[int] = None):
    """TSP over `visit_set` using precomputed pairwise distances.

    Always returns a tour rotated so it begins at `start_node_id` (which
    must be a member of `visit_set`).
    """
    if start_node_id not in visit_set:
        raise ValueError(
            f"start_node_id {start_node_id!r} must be in visit_set"
        )
    if seed is not None:
        random.seed(seed)

    visit = list(visit_set)

    def _tour_distance(order):
        return sum(all_dist[order[i]][order[i + 1]]
                   for i in range(len(order) - 1))

    def _greedy(start):
        unvisited = set(visit) - {start}
        order, cur = [start], start
        while unvisited:
            best = min(unvisited, key=lambda n: all_dist[cur][n])
            if all_dist[cur][best] == float("inf"):
                break
            order.append(best)
            unvisited.discard(best)
            cur = best
        return order

    def _two_opt(order):
        best, n = order[:], len(order)
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

    best_order = _two_opt(_greedy(start_node_id))
    best_dist = _tour_distance(best_order)

    other_starts = [n for n in visit if n != start_node_id]
    candidates = random.sample(
        other_starts, min(num_restarts - 1, len(other_starts))
    )
    for start in candidates:
        order = _two_opt(_greedy(start))
        if start_node_id in order:
            idx = order.index(start_node_id)
            order = order[idx:] + order[:idx]
        d = _tour_distance(order)
        if d < best_dist:
            best_order, best_dist = order, d

    return best_order


# ---------------------------------------------------------------------------
# Stage 3 — expand TSP order into a robot-driveable path along the graph
# ---------------------------------------------------------------------------

def _reconstruct(prev, end):
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def _expand_tour(best_order, all_prev):
    """Fill Dijkstra hops between consecutive TSP nodes.

    Result: full_path[0] == best_order[0]; for each i, the segment
    [best_order[i-1] ... best_order[i]] follows the shortest path through
    the graph.  Duplicate-junction nodes are dropped at concatenation.
    """
    full_path = [best_order[0]]
    for i in range(1, len(best_order)):
        seg = _reconstruct(all_prev[best_order[i - 1]], best_order[i])
        full_path.extend(seg[1:])
    return full_path


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_route(graph_data: dict, start_node_id, num_restarts: int = 10,
                   seed: Optional[int] = None) -> list:
    """Full-floor coverage tour: TSP over every reachable node, expanded.

    Backward-compatible with the prior single-function version.
    """
    all_dist, all_prev, reachable = _compute_all_pairs(graph_data, start_node_id)
    best_order = _run_tsp(reachable, start_node_id, all_dist,
                          num_restarts=num_restarts, seed=seed)
    return _expand_tour(best_order, all_prev)


def generate_door_route(graph_data: dict, start_node_id,
                        door_node_ids: Iterable, num_restarts: int = 10,
                        seed: Optional[int] = None,
                        expand: bool = True) -> list:
    """Door-coverage tour: TSP over `{start} ∪ door_node_ids`.

    Args:
        graph_data:    Parsed nav graph JSON dict.
        start_node_id: Robot spawn node id (must be in the graph).
        door_node_ids: Iterable of door node ids.  IDs not present in the
                       graph or not reachable from `start_node_id` are
                       dropped with a warning print.
        num_restarts:  TSP multi-restart count (default 10).
        seed:          Optional RNG seed.
        expand:        If True (default), insert Dijkstra hops between
                       consecutive TSP doors so the result is a
                       corridor-following waypoint list (good for driving
                       through a curated nav graph).  If False, return
                       just the TSP-ordered door visit list — handy when
                       a downstream planner (e.g. Nav2 SmacPlanner2D) is
                       responsible for corridor routing.

    Returns:
        List of node ids.  When `expand=True`, includes every nav-graph
        node on the shortest path between TSP-ordered doors.
        When `expand=False`, only the TSP-ordered doors (with
        `start_node_id` first).
    """
    all_dist, all_prev, reachable = _compute_all_pairs(graph_data, start_node_id)

    requested = set(door_node_ids)
    nodes_present = {n["id"] for n in graph_data["nodes"]}
    missing = requested - nodes_present
    unreachable = (requested & nodes_present) - reachable

    visit_set = {start_node_id} | (requested & reachable)

    if missing:
        sample = sorted(missing)[:10]
        suffix = " ..." if len(missing) > 10 else ""
        print(f"[coverage] WARN: {len(missing)} door ids not in graph: "
              f"{sample}{suffix}")
    if unreachable:
        sample = sorted(unreachable)[:10]
        suffix = " ..." if len(unreachable) > 10 else ""
        print(f"[coverage] WARN: {len(unreachable)} door ids unreachable "
              f"from start node {start_node_id}: {sample}{suffix}")

    door_in_visit = (visit_set - {start_node_id})
    print(f"[coverage] door TSP visit set: {len(visit_set)} nodes "
          f"(start + {len(door_in_visit)} doors)")

    best_order = _run_tsp(visit_set, start_node_id, all_dist,
                          num_restarts=num_restarts, seed=seed)
    if expand:
        return _expand_tour(best_order, all_prev)
    return best_order


def find_nearest_node(graph_data: dict, xy_m) -> str:
    """Return the node id whose world position is nearest to xy_m (metres).

    Reads `world_xy_m` first, then falls back to `world_xy_cm / 100` —
    works for both auto_navgraph and the older legacy schema.
    """
    import numpy as np
    target = np.array(xy_m, dtype=float)
    best_id, best_dist = None, float("inf")
    for n in graph_data["nodes"]:
        if "world_xy_m" in n:
            pos = np.array(n["world_xy_m"], dtype=float)
        else:
            pos = np.array(n["world_xy_cm"], dtype=float) / 100.0
        d = float(np.linalg.norm(pos - target))
        if d < best_dist:
            best_dist, best_id = d, n["id"]
    return best_id
