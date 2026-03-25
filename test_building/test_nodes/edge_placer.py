#!/usr/bin/env python3
"""
Interactive Edge Placement GUI for Navigation Graphs.

Displays a floor plan image with detected nodes overlaid. Click two nodes
to create an edge between them. Press S to save, L to load, Z to undo.

Usage:
    python3 edge_placer.py --floor floor_4
    python3 edge_placer.py --floor floor_1
"""

import argparse
import json
import os
import sys
import heapq
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # fallback; VSCode may override
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


SCRIPT_DIR = Path(__file__).parent


def load_nodes(floor):
    """Load detected nodes for the given floor."""
    path = SCRIPT_DIR / f"{floor}_detected_nodes.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def find_image(floor):
    """Find the floor plan image to use as background."""
    img_dir = SCRIPT_DIR / "images"
    # Try common patterns
    candidates = list(img_dir.glob(f"{floor}_occupancy*edge*node*"))
    if not candidates:
        candidates = list(img_dir.glob(f"{floor}_occupancy*"))
    if not candidates:
        candidates = list(img_dir.glob(f"{floor}*"))
    if candidates:
        return str(candidates[0])
    # Fallback to output directory
    fallback = SCRIPT_DIR.parent / "output" / f"{floor}_occupancy.png"
    if fallback.exists():
        return str(fallback)
    print(f"ERROR: No image found for {floor}")
    sys.exit(1)


def edges_path(floor):
    return SCRIPT_DIR / f"{floor}_manual_edges.json"


def dijkstra_path_img(floor):
    return SCRIPT_DIR / f"{floor}_dijkstra_path_manual.png"


def dijkstra(adj, start, end, n_nodes):
    """Dijkstra's shortest path algorithm."""
    dist = [float('inf')] * n_nodes
    prev = [-1] * n_nodes
    dist[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in adj[u]:
            if v not in visited and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    if dist[end] == float('inf'):
        return None, float('inf')
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[end]


class EdgePlacer:
    def __init__(self, floor):
        self.floor = floor
        self.data = load_nodes(floor)
        self.nodes = self.data['nodes']
        self.grid_w, self.grid_h = self.data['grid_size']

        # Load background image
        img_path = find_image(floor)
        self.bg_img = mpimg.imread(img_path)
        self.img_h, self.img_w = self.bg_img.shape[:2]

        # Convert node grid coords to image coords
        self.node_img_coords = []
        for node in self.nodes:
            x = node['grid_col'] * self.img_w / self.grid_w
            y = node['grid_row'] * self.img_h / self.grid_h
            self.node_img_coords.append((x, y))

        # Edge list: [(idx_a, idx_b), ...]
        self.edges = []
        self.selected_node = None  # index of first selected node

        # Try to load existing edges
        self._load_edges()

        # Track drawn matplotlib artists for undo
        self.edge_artists = []

        # Setup plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 9))
        self.fig.canvas.manager.set_window_title(f"Edge Placer — {floor}")
        self.ax.set_title(
            f"{floor} — Click two nodes to create an edge | "
            "S=Save | Z=Undo | D=Dijkstra | Q=Quit",
            fontsize=10
        )

        # Draw background
        self.ax.imshow(self.bg_img)

        # Draw existing edges
        for ia, ib in self.edges:
            self._draw_edge(ia, ib)

        # Draw nodes
        self.node_artists = []
        for idx, (x, y) in enumerate(self.node_img_coords):
            circle = plt.Circle((x, y), 6, color='blue', alpha=0.7, picker=True)
            self.ax.add_patch(circle)
            label = self.ax.text(x + 8, y - 3, str(idx + 1),
                                 fontsize=5, color='white',
                                 bbox=dict(boxstyle='round,pad=0.15',
                                           facecolor='black', alpha=0.6))
            self.node_artists.append((circle, label))

        self.ax.set_xlim(0, self.img_w)
        self.ax.set_ylim(self.img_h, 0)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Status text
        self.status_text = self.ax.text(
            0.5, -0.02, f"Edges: {len(self.edges)} | Click a node to start",
            transform=self.ax.transAxes, ha='center', fontsize=9,
            color='green'
        )

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.show()

    def _find_nearest_node(self, x, y, max_dist=15):
        """Find the nearest node to (x, y) within max_dist pixels."""
        best_idx = None
        best_dist = max_dist
        for idx, (nx, ny) in enumerate(self.node_img_coords):
            d = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _draw_edge(self, ia, ib):
        """Draw an edge line between two nodes."""
        x1, y1 = self.node_img_coords[ia]
        x2, y2 = self.node_img_coords[ib]
        line, = self.ax.plot([x1, x2], [y1, y2], 'g-', linewidth=1.5, alpha=0.8)
        self.edge_artists.append(line)
        return line

    def _update_status(self, msg=None):
        if msg is None:
            msg = f"Edges: {len(self.edges)} | Click a node to start"
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _highlight_node(self, idx, color='yellow'):
        circle, _ = self.node_artists[idx]
        circle.set_color(color)
        self.fig.canvas.draw_idle()

    def _unhighlight_node(self, idx):
        circle, _ = self.node_artists[idx]
        circle.set_color('blue')
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Right click = deselect
        if event.button == 3:
            if self.selected_node is not None:
                self._unhighlight_node(self.selected_node)
                self.selected_node = None
                self._update_status()
            return

        idx = self._find_nearest_node(event.xdata, event.ydata)
        if idx is None:
            return

        if self.selected_node is None:
            # First click — select node
            self.selected_node = idx
            self._highlight_node(idx)
            self._update_status(
                f"Selected node_{idx + 1} — click another node to create edge"
            )
        else:
            # Second click — create edge
            if idx == self.selected_node:
                # Clicked same node, deselect
                self._unhighlight_node(idx)
                self.selected_node = None
                self._update_status()
                return

            ia, ib = self.selected_node, idx

            # Check for duplicate
            edge_pair = (min(ia, ib), max(ia, ib))
            if edge_pair in [(min(a, b), max(a, b)) for a, b in self.edges]:
                self._unhighlight_node(ia)
                self.selected_node = None
                self._update_status(
                    f"Edge node_{ia+1} — node_{ib+1} already exists"
                )
                return

            # Add edge
            self.edges.append((ia, ib))
            self._draw_edge(ia, ib)
            self._unhighlight_node(ia)
            self.selected_node = None
            self._update_status(
                f"Added edge node_{ia+1} — node_{ib+1} | "
                f"Total: {len(self.edges)}"
            )

    def _on_key(self, event):
        if event.key == 's':
            self._save_edges()
        elif event.key == 'l':
            self._load_edges()
            self._redraw()
        elif event.key == 'z':
            self._undo()
        elif event.key == 'd':
            self._run_dijkstra()
        elif event.key == 'q':
            plt.close(self.fig)

    def _undo(self):
        if not self.edges:
            self._update_status("No edges to undo")
            return
        self.edges.pop()
        if self.edge_artists:
            artist = self.edge_artists.pop()
            artist.remove()
        if self.selected_node is not None:
            self._unhighlight_node(self.selected_node)
            self.selected_node = None
        self._update_status(f"Undone — Edges: {len(self.edges)}")
        self.fig.canvas.draw_idle()

    def _save_edges(self):
        """Save edges to JSON."""
        edge_list = []
        for ia, ib in self.edges:
            x1, y1 = self.node_img_coords[ia]
            x2, y2 = self.node_img_coords[ib]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            edge_list.append({
                "from": f"node_{ia + 1}",
                "to": f"node_{ib + 1}",
                "distance": round(float(dist), 2)
            })

        out = {
            "floor": self.floor,
            "edges": edge_list,
            "total_edges": len(edge_list)
        }

        path = edges_path(self.floor)
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

        self._update_status(f"Saved {len(edge_list)} edges to {path.name}")
        print(f"Saved {len(edge_list)} edges to {path}")

    def _load_edges(self):
        """Load edges from JSON if file exists."""
        path = edges_path(self.floor)
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        self.edges = []
        for e in data.get('edges', []):
            # Parse node indices from "node_X" labels
            ia = int(e['from'].replace('node_', '')) - 1
            ib = int(e['to'].replace('node_', '')) - 1
            if 0 <= ia < len(self.nodes) and 0 <= ib < len(self.nodes):
                self.edges.append((ia, ib))

        print(f"Loaded {len(self.edges)} edges from {path}")

    def _redraw(self):
        """Redraw all edges."""
        for artist in self.edge_artists:
            artist.remove()
        self.edge_artists = []
        for ia, ib in self.edges:
            self._draw_edge(ia, ib)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _run_dijkstra(self):
        """Find and display shortest path between farthest connected nodes."""
        if len(self.edges) < 2:
            self._update_status("Need at least 2 edges for Dijkstra")
            return

        # Build adjacency list
        adj = defaultdict(list)
        for ia, ib in self.edges:
            x1, y1 = self.node_img_coords[ia]
            x2, y2 = self.node_img_coords[ib]
            d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            adj[ia].append((ib, d))
            adj[ib].append((ia, d))

        # Find connected components via BFS
        all_visited = set()
        largest_comp = set()
        for n in range(len(self.nodes)):
            if n in all_visited:
                continue
            comp = set()
            queue = [n]
            comp.add(n)
            while queue:
                u = queue.pop(0)
                for v, _ in adj[u]:
                    if v not in comp:
                        comp.add(v)
                        queue.append(v)
            all_visited |= comp
            if len(comp) > len(largest_comp):
                largest_comp = comp

        # Find farthest pair in largest component
        best_dist = 0
        start, end = 0, 0
        nodes_list = list(largest_comp)
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                ni, nj = nodes_list[i], nodes_list[j]
                x1, y1 = self.node_img_coords[ni]
                x2, y2 = self.node_img_coords[nj]
                d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if d > best_dist:
                    best_dist = d
                    start, end = ni, nj

        path, total = dijkstra(adj, start, end, len(self.nodes))

        if path is None:
            self._update_status("No path found!")
            return

        # Draw path
        for k in range(len(path) - 1):
            x1, y1 = self.node_img_coords[path[k]]
            x2, y2 = self.node_img_coords[path[k + 1]]
            self.ax.plot([x1, x2], [y1, y2], 'lime', linewidth=3, alpha=0.9)

        # Mark start/end
        sx, sy = self.node_img_coords[path[0]]
        ex, ey = self.node_img_coords[path[-1]]
        self.ax.plot(sx, sy, 'go', markersize=12)
        self.ax.plot(ex, ey, 'ro', markersize=12)

        # Mark waypoints
        for node_idx in path[1:-1]:
            wx, wy = self.node_img_coords[node_idx]
            self.ax.plot(wx, wy, 'yo', markersize=8)

        self._update_status(
            f"Dijkstra: {len(path)} nodes, dist={total:.0f}px | "
            f"node_{start+1} → node_{end+1}"
        )
        self.fig.canvas.draw_idle()

        # Save path image
        out_path = dijkstra_path_img(self.floor)
        self.fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved Dijkstra path to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive edge placement GUI")
    parser.add_argument('--floor', default='floor_4',
                        help='Floor to edit (default: floor_4)')
    args = parser.parse_args()

    print(f"\n  Edge Placer — {args.floor}")
    print(f"  Controls:")
    print(f"    Left-click  : Select/connect nodes")
    print(f"    Right-click : Deselect current node")
    print(f"    S           : Save edges to JSON")
    print(f"    L           : Load edges from JSON")
    print(f"    Z           : Undo last edge")
    print(f"    D           : Run Dijkstra (farthest pair)")
    print(f"    Q           : Quit\n")

    EdgePlacer(args.floor)


if __name__ == "__main__":
    main()
