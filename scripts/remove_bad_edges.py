"""Remove validated-bad edges from the nav graph JSON."""
import json, os

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/Graph/floor_4_nav_graph_world_xflipped.json")

BAD_EDGES = {
    ("node_49", "node_41"),
    ("node_69", "node_43"),
    ("node_46", "node_42"),
}

with open(GRAPH_PATH) as f:
    data = json.load(f)

before = len(data["edges"])
data["edges"] = [
    e for e in data["edges"]
    if (e["from"], e["to"]) not in BAD_EDGES
]
after = len(data["edges"])
print(f"Removed {before - after} edges ({before} → {after})")

with open(GRAPH_PATH, "w") as f:
    json.dump(data, f, indent=2)
print("Saved.")
