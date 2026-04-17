# Full Floor Coverage — Navigation Pipeline

Finds the shortest route that visits every reachable node on floor 4 and produces a PNG visualization.

---

## Overview

This is a **graph coverage problem**: starting from a single node, visit every other reachable node while minimizing total travel distance.

Because the floor plan is a weighted graph (not a complete graph with direct connections between all pairs), the travel cost between any two nodes is their **Dijkstra shortest-path distance**, not straight-line distance. This is solved in two stages:

1. **All-pairs shortest paths** — run Dijkstra from every reachable node to get the full distance matrix
2. **Nearest-neighbor greedy heuristic** — at each step, move to the closest unvisited node
3. **2-opt improvement** — repeatedly swap edge pairs to shorten the tour until no swap helps
4. **Multiple random restarts** — repeat steps 2–3 from several start nodes, keep the shortest result

This is an approximation of the Traveling Salesman Problem (TSP). The nearest-neighbor heuristic gives a practical solution in O(n²) time; 2-opt post-processing typically reduces the tour by a further 10–20%.

---

## Files

| File | Description |
|------|-------------|
| `coverage.json` | Solved coverage route — visit order, full waypoint path, distances, unreachable nodes |
| `coverage_map.png` | Floor plan PNG with the route and all nodes visualized |
| `coverage_metrics.txt` | Timing and quality report — per-restart log, 2-opt stats, wall time |
| `generate_coverage.py` | Generates `coverage.json` and `coverage_metrics.txt` — nearest-neighbor + 2-opt + multi-restart |
| `visualize_coverage.py` | Script that reads `coverage.json` and generates `coverage_map.png` |

---

## Step 1: Generate `coverage.json`

Run from the repo root:

```bash
python3 navigation/full_floor_coverage/generate_coverage.py
```

Or run inline without a separate file:

```bash
python3 - << 'EOF'
import json, heapq, os, random

NODES_FILE   = "test_building/test_nodes/Final_Data/floor_4_fullres_graph.json"
OUT_FILE     = "navigation/full_floor_coverage/coverage.json"
START_NODE   = "node_1"               # change to any reachable node ID
NUM_RESTARTS = 10                     # total start candidates tried (including START_NODE)

with open(NODES_FILE) as f:
    data = json.load(f)

nodes = {n["id"]: n for n in data["nodes"]}
adj   = {nid: [] for nid in nodes}
for e in data["edges"]:
    a, b = e["from"], e["to"]
    d = e["distance"]
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

# All-pairs Dijkstra (only over reachable nodes — skip unreachable sources)
all_dist, all_prev = {}, {}
all_dist[START_NODE], all_prev[START_NODE] = dijkstra(START_NODE)

reachable   = [n for n in nodes if all_dist[START_NODE][n] < float("inf")]
unreachable = [n for n in nodes if all_dist[START_NODE][n] == float("inf")]

for nid in reachable:
    if nid != START_NODE:
        all_dist[nid], all_prev[nid] = dijkstra(nid)

def run_greedy(start):
    unvisited = set(reachable) - {start}
    order = [start]
    dist  = 0.0
    cur   = start
    while unvisited:
        best = min(unvisited, key=lambda n: all_dist[cur][n])
        if all_dist[cur][best] == float("inf"):
            break
        dist += all_dist[cur][best]
        order.append(best)
        unvisited.discard(best)
        cur = best
    return order, dist

def two_opt(order):
    best = order[:]
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                j_next = (j + 1) % n
                before = all_dist[best[i-1]][best[i]] + all_dist[best[j]][best[j_next]]
                after  = all_dist[best[i-1]][best[j]] + all_dist[best[i]][best[j_next]]
                if after < before - 1e-9:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best

def tour_distance(order):
    return sum(all_dist[order[i]][order[i+1]] for i in range(len(order) - 1))

# Multiple random restarts: always include START_NODE plus k-1 random candidates
candidates = [START_NODE] + random.sample(
    [n for n in reachable if n != START_NODE],
    min(NUM_RESTARTS - 1, len(reachable) - 1)
)

best_order, best_dist = None, float("inf")
for start in candidates:
    order, _ = run_greedy(start)
    order = two_opt(order)
    dist  = tour_distance(order)
    print(f"  start={start:40s}  dist={dist:.0f} px  nodes={len(order)}")
    if dist < best_dist:
        best_order, best_dist = order, dist

# Rebuild full waypoint path with intermediate corridor hops
visit_order = best_order
full_path   = [visit_order[0]]
for i in range(1, len(visit_order)):
    seg = reconstruct(all_prev[visit_order[i-1]], visit_order[i])
    full_path.extend(seg[1:])

result = {
    "description": "2-opt + multi-restart nearest-neighbor full-floor coverage route for floor 4.",
    "start_node": visit_order[0],
    "algorithm": "nearest-neighbor heuristic + 2-opt + multi-restart",
    "num_restarts": NUM_RESTARTS,
    "floor": 4,
    "total_nodes": len(nodes),
    "reachable_nodes": len(reachable),
    "unreachable_nodes": len(unreachable),
    "nodes_visited": len(visit_order),
    "total_waypoints": len(full_path),
    "total_distance_px": round(best_dist, 2),
    "unreachable": [{"id": n} for n in unreachable],
    "visit_order": [
        {"step": i+1, "node_id": n,
         "dist_from_prev": round(all_dist[visit_order[i-1]][n], 2) if i > 0 else 0.0}
        for i, n in enumerate(visit_order)
    ],
    "full_path": full_path
}
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nBest tour: {len(visit_order)} nodes, {len(full_path)} waypoints, {best_dist:.0f} px")
print(f"Saved {OUT_FILE}")
EOF
```

The script uses two optimizations over the original single-pass greedy heuristic:

**2-opt post-processing** — after building the greedy tour, repeatedly scan all pairs of edges. If reversing the segment between them reduces total cost, apply the swap. Repeats until no improving swap exists. Typically reduces tour length by 10–20%.

**Multiple random restarts** — runs the greedy heuristic from `NUM_RESTARTS` different start candidates (always including `START_NODE`), applies 2-opt to each, and keeps the shortest result.

To configure, edit the constants at the top of `generate_coverage.py`:

```python
START_NODE   = "node_55"              # preferred start; always included as a candidate
NUM_RESTARTS = 10                     # total start candidates tried (including START_NODE)
```

### Output: `coverage.json`

| Field | Description |
|-------|-------------|
| `start_node` | Node ID where the best tour begins (may differ from `START_NODE` if another restart won) |
| `algorithm` | Records the strategy used (`nearest-neighbor heuristic + 2-opt + multi-restart`) |
| `num_restarts` | Number of start candidates that were tried |
| `nodes_visited` | Number of reachable nodes covered |
| `unreachable_nodes` | Count of isolated nodes (no graph path from start) |
| `total_waypoints` | Length of `full_path` (includes intermediate corridor hops) |
| `total_distance_px` | Sum of all Dijkstra hop costs in pixels for the winning tour |
| `visit_order` | Ordered list of target nodes with step number and cost |
| `full_path` | Complete waypoint sequence including intermediate nodes |
| `unreachable` | List of node IDs with no graph connection |

---

## Step 2: Generate `coverage_map.png`

```bash
python3 navigation/full_floor_coverage/visualize_coverage.py
```

### Color scheme

| Element | Color |
|---------|-------|
| All nodes | Blue |
| **Unreachable nodes** | **Magenta X** |
| Coverage route line | Dark orange |
| Start node | Large bright green circle |
| Graph edges (background) | Faint grey |

---

## Current Results (Floor 4)

Run `generate_coverage.py` to populate this section. Results will be printed to stdout and saved in `coverage.json`.

---

## Changing the Start Node

Edit `START_NODE` at the top of `generate_coverage.py` to any node ID in the graph, e.g.:

```python
START_NODE = "node_1"    # default
START_NODE = "node_40"   # another candidate
```

`START_NODE` is always included as a restart candidate so the best tour is guaranteed to be at least as good as starting from your chosen node. If a random restart produces a shorter tour, the output `start_node` field will reflect the winning start instead.

To increase solution quality at the cost of more compute, raise `NUM_RESTARTS`.
