"""Building eval scenario definitions for Nova Carter in the Innovation Center.

12 fixed scenarios across 4 floors, each testing a specific navigation
challenge. Coordinates are world-frame meters, derived from building_waypoints.json.

Pass/fail thresholds:
  Open run:       goal_reach_rate ≥ 80%,  max_collisions=0, max_time=20s
  Corner nav:     goal_reach_rate ≥ 60%,  max_collisions=1, max_time=40s
  Corridor:       goal_reach_rate ≥ 60%,  max_collisions=1, max_time=40s
  Doorway:        goal_reach_rate ≥ 40%,  max_collisions=2, max_time=60s

All doorway scenarios: doors are frozen kinematic (RigidBodyAPI.kinematicEnabled=True)
during scene setup. Door-opening is out of scope for this fine-tuning phase.

Each scenario runs 5 trials from the same fixed spawn + goal.
"""

from dataclasses import dataclass, field


@dataclass
class BuildingScenarioDefinition:
    name: str
    floor: str
    scenario_type: str          # "open_run", "corner_nav", "corridor", "doorway"
    spawn: tuple                # (x, y, z, yaw_deg) world frame meters
    goal: tuple                 # (x, y, z) world frame meters
    distance_m: float           # approximate path length
    num_trials: int = 5
    max_time_s: float = 40.0
    max_collisions: int = 1
    goal_reach_rate_pass: float = 0.60
    description: str = ""


BUILDING_SCENARIOS = [
    # ── Floor G — Open Runs ──
    BuildingScenarioDefinition(
        name="G-open-1",
        floor="G",
        scenario_type="open_run",
        spawn=(-12.9, 20.9, 4.47, 0.0),
        goal=(-10.5, 21.5, 4.47),
        distance_m=8.0,
        max_time_s=20.0,
        max_collisions=0,
        goal_reach_rate_pass=0.80,
        description="Straight lobby run, Event Floor center → north.",
    ),
    BuildingScenarioDefinition(
        name="G-open-2",
        floor="G",
        scenario_type="open_run",
        spawn=(-14.5, 22.0, 4.47, 45.0),
        goal=(-11.0, 18.5, 4.47),
        distance_m=5.0,
        max_time_s=20.0,
        max_collisions=0,
        goal_reach_rate_pass=0.80,
        description="Lobby diagonal run, northwest to southeast.",
    ),

    # ── Floor G — Corner Nav ──
    BuildingScenarioDefinition(
        name="G-corner-1",
        floor="G",
        scenario_type="corner_nav",
        spawn=(-9.5, 20.0, 4.47, 0.0),
        goal=(-15.0, 19.5, 4.47),
        distance_m=6.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="90° turn across lobby, east wall → west corridor.",
    ),

    # ── Floor G — Doorway ──
    BuildingScenarioDefinition(
        name="G-door-1",
        floor="G",
        scenario_type="doorway",
        spawn=(-12.0, 23.5, 4.47, -90.0),
        goal=(-12.9, 20.9, 4.47),
        distance_m=4.0,
        max_time_s=60.0,
        max_collisions=2,
        goal_reach_rate_pass=0.40,
        description="Through entrance doorway into lobby. Doors frozen kinematic.",
    ),

    # ── Floor 1 — Corridors ──
    BuildingScenarioDefinition(
        name="1-corridor-1",
        floor="1",
        scenario_type="corridor",
        spawn=(-17.1, 19.0, 7.62, 0.0),
        goal=(-13.5, 20.0, 7.62),
        distance_m=7.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="Robotics Lab hallway, entrance → open area east.",
    ),
    BuildingScenarioDefinition(
        name="1-corridor-2",
        floor="1",
        scenario_type="corridor",
        spawn=(-15.0, 19.5, 7.62, 180.0),
        goal=(-16.0, 17.0, 7.62),
        distance_m=5.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="Narrower corridor, Robotics Lab center → south.",
    ),

    # ── Floor 1 — Corner Nav ──
    BuildingScenarioDefinition(
        name="1-corner-1",
        floor="1",
        scenario_type="corner_nav",
        spawn=(-13.5, 20.0, 7.62, -90.0),
        goal=(-18.5, 21.0, 7.62),
        distance_m=6.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="90° hallway turn, east open area → north corridor.",
    ),

    # ── Floor 1 — Doorway ──
    BuildingScenarioDefinition(
        name="1-door-1",
        floor="1",
        scenario_type="doorway",
        spawn=(-15.0, 19.5, 7.62, 90.0),
        goal=(-17.1, 19.0, 7.62),
        distance_m=3.0,
        max_time_s=60.0,
        max_collisions=2,
        goal_reach_rate_pass=0.40,
        description="Through lab door into Robotics Lab entrance.",
    ),

    # ── Floor 2 — Open Run + Corridor ──
    BuildingScenarioDefinition(
        name="2-open-1",
        floor="2",
        scenario_type="open_run",
        spawn=(-17.2, 18.6, 10.67, 0.0),
        goal=(-14.0, 18.5, 10.67),
        distance_m=6.0,
        max_time_s=20.0,
        max_collisions=0,
        goal_reach_rate_pass=0.80,
        description="Open area run, Classroom 302 → east.",
    ),
    BuildingScenarioDefinition(
        name="2-corridor-1",
        floor="2",
        scenario_type="corridor",
        spawn=(-15.0, 20.0, 10.67, 0.0),
        goal=(-16.5, 22.0, 10.67),
        distance_m=5.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="Floor 2 north corridor.",
    ),

    # ── Floor 3 — Open Run + Corridor ──
    BuildingScenarioDefinition(
        name="3-open-1",
        floor="3",
        scenario_type="open_run",
        spawn=(-12.7, 20.6, 13.72, 0.0),
        goal=(-11.0, 19.5, 13.72),
        distance_m=5.0,
        max_time_s=20.0,
        max_collisions=0,
        goal_reach_rate_pass=0.80,
        description="Open area, Capstone Studio center → east.",
    ),
    BuildingScenarioDefinition(
        name="3-corridor-1",
        floor="3",
        scenario_type="corridor",
        spawn=(-14.5, 22.0, 13.72, -90.0),
        goal=(-14.0, 18.5, 13.72),
        distance_m=4.0,
        max_time_s=40.0,
        max_collisions=1,
        goal_reach_rate_pass=0.60,
        description="Floor 3 corridor, studio north → south.",
    ),
]

# Quick lookup
SCENARIOS_BY_NAME = {s.name: s for s in BUILDING_SCENARIOS}
SCENARIOS_BY_FLOOR = {
    floor: [s for s in BUILDING_SCENARIOS if s.floor == floor]
    for floor in ["G", "1", "2", "3"]
}
