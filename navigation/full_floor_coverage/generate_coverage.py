#!/usr/bin/env python3
"""
Generate coverage.json for full-floor coverage routing.

Optimizations over the original greedy heuristic:
  - 2-opt post-processing: repeatedly swaps edge pairs to shorten the tour
  - Multiple random restarts: runs greedy from k start candidates, keeps the best
"""
import json
import heapq
import os
import random
import time

NODES_FILE   = "test_building/test_nodes/Final_Data/floor_4_fullres_graph.json"
OUT_FILE     = "navigation/full_floor_coverage/coverage.json"
METRICS_FILE = "navigation/full_floor_coverage/coverage_metrics.txt"
START_NODE   = "node_55"            # change to any reachable node ID
NUM_RESTARTS = 10                   # number of random start candidates to try

wall_start = time.perf_counter()

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


def reconstruct(prev, end):
    path, cur = [], end
    while cur:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


# All-pairs Dijkstra (only over reachable nodes — skip unreachable sources)
t0 = time.perf_counter()
all_dist, all_prev = {}, {}
all_dist[START_NODE], all_prev[START_NODE] = dijkstra(START_NODE)

reachable   = [n for n in nodes if all_dist[START_NODE][n] < float("inf")]
unreachable = [n for n in nodes if all_dist[START_NODE][n] == float("inf")]

for nid in reachable:
    if nid != START_NODE:
        all_dist[nid], all_prev[nid] = dijkstra(nid)
t_dijkstra = time.perf_counter() - t0


def run_greedy(start):
    """Nearest-neighbor greedy tour from a given start node."""
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
    """
    2-opt improvement: scan all edge pairs and reverse the segment between them
    if doing so reduces total tour cost. Repeats until no improvement is found.
    """
    best = order[:]
    n = len(best)
    improved = True
    passes = 0
    total_swaps = 0
    while improved:
        improved = False
        passes += 1
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
                    total_swaps += 1
    return best, passes, total_swaps


def tour_distance(order):
    return sum(all_dist[order[i]][order[i + 1]] for i in range(len(order) - 1))


# Multiple random restarts: try START_NODE plus k-1 random reachable candidates
candidates = [START_NODE] + random.sample(
    [n for n in reachable if n != START_NODE],
    min(NUM_RESTARTS - 1, len(reachable) - 1)
)

restart_log = []
best_order, best_dist = None, float("inf")
t0 = time.perf_counter()
for start in candidates:
    t_r = time.perf_counter()
    order, greedy_dist = run_greedy(start)
    order, passes, swaps = two_opt(order)
    dist = tour_distance(order)
    elapsed = time.perf_counter() - t_r
    improvement_pct = (greedy_dist - dist) / greedy_dist * 100 if greedy_dist > 0 else 0.0
    restart_log.append({
        "start": start,
        "greedy_dist": round(greedy_dist, 2),
        "final_dist": round(dist, 2),
        "improvement_pct": round(improvement_pct, 2),
        "two_opt_passes": passes,
        "two_opt_swaps": swaps,
        "time_s": round(elapsed, 4),
        "nodes": len(order),
        "winner": False,
    })
    print(f"  start={start:12s}  greedy={greedy_dist:.0f}  final={dist:.0f} px"
          f"  improvement={improvement_pct:.1f}%  swaps={swaps}  t={elapsed:.3f}s")
    if dist < best_dist:
        best_order, best_dist = order, dist
t_tsp = time.perf_counter() - t0

# Mark the winning restart
for r in restart_log:
    if r["start"] == best_order[0] and r["final_dist"] == round(best_dist, 2):
        r["winner"] = True
        break

# Rebuild the full waypoint path (including intermediate corridor hops)
visit_order = best_order
full_path   = [visit_order[0]]
for i in range(1, len(visit_order)):
    seg = reconstruct(all_prev[visit_order[i - 1]], visit_order[i])
    full_path.extend(seg[1:])

wall_total = time.perf_counter() - wall_start

# Greedy-only baseline from START_NODE for improvement comparison
greedy_baseline_order, greedy_baseline_dist = run_greedy(START_NODE)
overall_improvement_pct = (greedy_baseline_dist - best_dist) / greedy_baseline_dist * 100

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
        {
            "step": i + 1,
            "node_id": n,
            "dist_from_prev": round(all_dist[visit_order[i - 1]][n], 2) if i > 0 else 0.0,
        }
        for i, n in enumerate(visit_order)
    ],
    "full_path": full_path,
}

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

# Write metrics report
winner = next(r for r in restart_log if r["winner"])
lines = [
    "=" * 60,
    "COVERAGE ALGORITHM METRICS REPORT",
    "=" * 60,
    f"Floor:                    4",
    f"Graph file:               {NODES_FILE}",
    f"Preferred start node:     {START_NODE}",
    f"Winning start node:       {visit_order[0]}",
    "",
    "── Graph ──────────────────────────────────────────────────",
    f"  Total nodes:            {len(nodes)}",
    f"  Reachable nodes:        {len(reachable)}",
    f"  Unreachable nodes:      {len(unreachable)}",
    f"  Total edges:            {len(data['edges'])}",
    "",
    "── Timing ─────────────────────────────────────────────────",
    f"  All-pairs Dijkstra:     {t_dijkstra:.4f} s  ({len(reachable)} sources × {len(reachable)} nodes)",
    f"  TSP restarts total:     {t_tsp:.4f} s",
    f"  Wall time (total):      {wall_total:.4f} s",
    "",
    "── Tour quality ────────────────────────────────────────────",
    f"  Greedy baseline (node_66): {greedy_baseline_dist:.2f} px",
    f"  Best tour distance:     {best_dist:.2f} px",
    f"  Improvement over greedy baseline: {overall_improvement_pct:.2f}%",
    f"  Nodes visited:          {len(visit_order)}",
    f"  Total waypoints (with corridor hops): {len(full_path)}",
    f"  Avg hop distance:       {best_dist / max(len(visit_order) - 1, 1):.2f} px",
    "",
    "── Restart log ─────────────────────────────────────────────",
    f"  {'Start':<12}  {'Greedy':>8}  {'Final':>8}  {'Improv':>7}  {'Passes':>6}  {'Swaps':>6}  {'Time':>7}",
]
for r in restart_log:
    tag = " <-- WINNER" if r["winner"] else ""
    lines.append(
        f"  {r['start']:<12}  {r['greedy_dist']:>8.0f}  {r['final_dist']:>8.0f}"
        f"  {r['improvement_pct']:>6.1f}%  {r['two_opt_passes']:>6}  {r['two_opt_swaps']:>6}"
        f"  {r['time_s']:>6.3f}s{tag}"
    )
lines += [
    "",
    "── 2-opt summary (winning restart) ────────────────────────",
    f"  Greedy dist before 2-opt: {winner['greedy_dist']:.2f} px",
    f"  Final dist after 2-opt:   {winner['final_dist']:.2f} px",
    f"  Improvement:              {winner['improvement_pct']:.2f}%",
    f"  2-opt passes:             {winner['two_opt_passes']}",
    f"  2-opt swaps applied:      {winner['two_opt_swaps']}",
    "=" * 60,
]

with open(METRICS_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nBest tour: {len(visit_order)} nodes, {len(full_path)} waypoints, {best_dist:.0f} px")
print(f"Saved {OUT_FILE}")
print(f"Saved {METRICS_FILE}")
