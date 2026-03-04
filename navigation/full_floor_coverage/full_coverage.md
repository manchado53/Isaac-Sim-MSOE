# Full Floor Coverage — Navigation Pipeline

Finds the shortest route that visits every reachable node on floor 1 and produces a PNG visualization.

---

## Overview

This is a **graph coverage problem**: starting from a single node, visit every other reachable node while minimizing total travel distance.

Because the floor plan is a weighted graph (not a complete graph with direct connections between all pairs), the travel cost between any two nodes is their **Dijkstra shortest-path distance**, not straight-line distance. This is solved in two stages:

1. **All-pairs shortest paths** — run Dijkstra from every node to get the full distance matrix
2. **Nearest-neighbor greedy heuristic** — at each step, move to the closest unvisited node

This is an approximation of the Traveling Salesman Problem (TSP). The nearest-neighbor heuristic gives a practical solution in O(n²) time without guaranteeing the global optimum.

---

## Files

| File | Description |
|------|-------------|
| `coverage.json` | Solved coverage route — visit order, full waypoint path, distances, unreachable nodes |
| `coverage_map.png` | Floor plan PNG with the route and all nodes visualized |
| `visualize_coverage.py` | Script that reads `coverage.json` and generates `coverage_map.png` |

---

## Step 1: Generate `coverage.json`

Run from the repo root:

```bash
python3 - << 'EOF'
import json, heapq, os

NODES_FILE = "floor_plans_resized/floor1/floor1_nodes.json"
OUT_FILE   = "navigation/full_floor_coverage/coverage.json"
START_NODE = "f1_junc_mid_center"   # change to any reachable node ID

with open(NODES_FILE) as f:
    data = json.load(f)

nodes = {n["id"]: n for n in data["nodes"]}
adj   = {nid: [] for nid in nodes}
for e in data["edges"]:
    a, b = e[0], e[1]
    d = ((nodes[a]["x"]-nodes[b]["x"])**2 + (nodes[a]["y"]-nodes[b]["y"])**2)**0.5
    adj[a].append((b, d))
    adj[b].append((a, d))

def dijkstra(start):
    dist = {n: float("inf") for n in nodes}
    prev = {n: None for n in nodes}
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def reconstruct(prev, end):
    path, cur = [], end
    while cur: path.append(cur); cur = prev[cur]
    return list(reversed(path))

# All-pairs Dijkstra
all_dist, all_prev = {}, {}
for nid in nodes:
    all_dist[nid], all_prev[nid] = dijkstra(nid)

reachable   = [n for n in nodes if all_dist[START_NODE][n] < float("inf")]
unreachable = [n for n in nodes if all_dist[START_NODE][n] == float("inf")]

# Nearest-neighbor greedy coverage
unvisited    = set(reachable) - {START_NODE}
visit_order  = [START_NODE]
full_path    = [START_NODE]
total_dist   = 0.0
current      = START_NODE

while unvisited:
    best = min(unvisited, key=lambda n: all_dist[current][n])
    if all_dist[current][best] == float("inf"):
        break
    seg = reconstruct(all_prev[current], best)
    full_path.extend(seg[1:])
    total_dist += all_dist[current][best]
    visit_order.append(best)
    unvisited.discard(best)
    current = best

result = {
    "description": "Greedy nearest-neighbor full-floor coverage route for floor 1.",
    "start_node": START_NODE,
    "start_name": nodes[START_NODE]["name"],
    "algorithm": "nearest-neighbor heuristic (greedy TSP approximation)",
    "floor": 1,
    "total_nodes": len(nodes),
    "reachable_nodes": len(reachable),
    "unreachable_nodes": len(unreachable),
    "nodes_visited": len(visit_order),
    "total_waypoints": len(full_path),
    "total_distance_px": round(total_dist, 2),
    "unreachable": [{"id": n, "name": nodes[n]["name"], "type": nodes[n]["type"]} for n in unreachable],
    "visit_order": [
        {"step": i+1, "node_id": n, "name": nodes[n]["name"], "type": nodes[n]["type"],
         "dist_from_prev": round(all_dist[visit_order[i-1]][n], 2) if i > 0 else 0.0}
        for i, n in enumerate(visit_order)
    ],
    "full_path": full_path
}
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {OUT_FILE}  ({len(visit_order)} nodes, {len(full_path)} waypoints, {total_dist:.0f} px)")
EOF
```

### Output: `coverage.json`

| Field | Description |
|-------|-------------|
| `start_node` | Node ID where the route begins |
| `nodes_visited` | Number of reachable nodes covered |
| `unreachable_nodes` | Count of isolated nodes (no graph path from start) |
| `total_waypoints` | Length of `full_path` (includes intermediate corridor hops) |
| `total_distance_px` | Sum of all Dijkstra hop costs in pixels |
| `visit_order` | Ordered list of target nodes with step number and cost |
| `full_path` | Complete waypoint sequence including intermediate nodes |
| `unreachable` | List of node IDs, names, and types with no graph connection |

---

## Step 2: Generate `coverage_map.png`

```bash
python3 navigation/full_floor_coverage/visualize_coverage.py
```

### Color scheme

| Element | Color |
|---------|-------|
| Room nodes | Blue |
| Hallway nodes | Light grey |
| Intersection nodes | Amber |
| Stairwell nodes | Red |
| Elevator nodes | Purple |
| Entrance nodes | Green |
| **Unreachable nodes** | **Magenta X** |
| Coverage route line | Dark orange |
| Start node | Large bright green circle |
| Graph edges (background) | Faint grey |

---

## Current Results (Floor 1)

| Metric | Value |
|--------|-------|
| Total nodes | 130 |
| Reachable from start | 118 |
| Unreachable (isolated) | 12 |
| Total coverage waypoints | 202 |
| Total travel distance | 15,455 px |
| Start node | `f1_junc_mid_center` (Mid-Center Junction) |

### Unreachable Nodes

These 12 nodes exist in `floor1_nodes.json` but have no edges connecting them to the main graph:

| Node ID | Name | Type |
|---------|------|------|
| `f1_stairs_NW` | Stairwell NW | stairwell |
| `f1_stairs_DN_1` | Stairs DN-1 | stairwell |
| `f1_stairs_DN_2` | Stairs DN-2 | stairwell |
| `f1_stairs_DN_3` | Stairs DN-3 | stairwell |
| `f1_stairs_SW` | Stairwell SW | stairwell |
| `f1_office_02` | Office 2 | room |
| `f1_office_03` | Office 3 | room |
| `f1_room_NW_1` | Room NW 1 | room |
| `f1_room_mid_W_3` | Room Mid W-3 | room |
| `f1_junc_atrium_S_center` | Atrium S Center | intersection |
| `f1_wp_15` | Waypoint 15 | hallway |
| `f1_wp_16` | Waypoint 16 | hallway |

To make these reachable, add edges for each in `floor_plans_resized/floor1/floor1_nodes.json` connecting them to a neighboring node.

---

## Changing the Start Node

Edit `START_NODE` in the Step 1 script to any node ID in the graph, e.g.:

```python
START_NODE = "f1_junc_left_wing_S"   # southwest corner
START_NODE = "f1_junc_top_mid"        # north mid corridor
START_NODE = "f1_junc_mid_center"     # central junction (default)
```

The nearest-neighbor result will differ depending on start position since the greedy choice at each step depends on current location.
