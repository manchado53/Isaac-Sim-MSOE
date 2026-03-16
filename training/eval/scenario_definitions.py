"""Test scenario definitions for Nova Carter Patrol evaluation.

Each scenario specifies fixed furniture positions, robot start, goal(s),
and pass/fail criteria. Used by eval_scenarios.py.
"""

from dataclasses import dataclass, field


@dataclass
class ScenarioDefinition:
    name: str
    difficulty: str
    robot_start: tuple  # (x, y, yaw) in env-local coords
    furniture_positions: dict  # {scene_name: (x, y, z, yaw)}
    goals: list  # [(x, y), ...] in env-local coords
    max_steps_per_goal: int  # step budget per goal
    max_collisions: int  # collision budget for pass
    description: str = ""


# Default "out of the way" furniture positions (pushed to corners)
_CORNER_FURNITURE = {
    "furniture_table": (4.0, 4.0, 0.375, 0.0),
    "furniture_bookshelf": (-4.0, 4.0, 0.9, 0.0),
    "furniture_chair": (4.0, -4.0, 0.45, 0.0),
    "furniture_desk": (-4.0, -4.0, 0.375, 0.0),
    "furniture_cabinet": (4.0, 0.0, 0.7, 0.0),
    "furniture_couch": (-4.0, 0.0, 0.3, 0.0),
    "furniture_bin": (0.0, 4.0, 0.25, 0.0),
}


SCENARIOS = [
    # ── 1. Open Run (Easy) ──
    # Clear path across room. Tests basic goal-reaching and stopping.
    ScenarioDefinition(
        name="Open Run",
        difficulty="easy",
        robot_start=(-2.0, 0.0, 0.0),
        furniture_positions=_CORNER_FURNITURE,
        goals=[(3.0, 0.0)],
        max_steps_per_goal=180,  # 15s at 12Hz
        max_collisions=0,
        description="Clear 6m path, no obstacles between robot and goal.",
    ),

    # ── 2. Obstacle Between (Medium) ──
    # Desk blocks the direct path. Must navigate around.
    ScenarioDefinition(
        name="Obstacle Between",
        difficulty="medium",
        robot_start=(-2.0, 0.0, 0.0),
        furniture_positions={
            **_CORNER_FURNITURE,
            "furniture_desk": (0.0, 0.0, 0.375, 0.0),  # desk in the middle
        },
        goals=[(3.0, 0.0)],
        max_steps_per_goal=240,  # 20s
        max_collisions=0,
        description="Desk (1.5x0.7m) blocks direct path. Must detour around.",
    ),

    # ── 3. Narrow Gap (Medium-Hard) ──
    # Bookshelf and couch create a ~1.5m corridor.
    ScenarioDefinition(
        name="Narrow Gap",
        difficulty="med-hard",
        robot_start=(0.0, -3.0, 1.57),  # facing north
        furniture_positions={
            **_CORNER_FURNITURE,
            "furniture_bookshelf": (-0.85, 0.0, 0.9, 0.0),  # left side of gap
            "furniture_couch": (0.85, 0.0, 0.3, 1.57),  # right side, rotated
        },
        goals=[(0.0, 3.0)],
        max_steps_per_goal=300,  # 25s
        max_collisions=0,
        description="Bookshelf and couch create a ~1.5m gap. Must pass through.",
    ),

    # ── 4. Goal Behind Furniture (Hard) ──
    # Goal in NE corner behind bookshelf against wall.
    ScenarioDefinition(
        name="Goal Behind Furniture",
        difficulty="hard",
        robot_start=(0.0, 0.0, 0.785),  # facing NE
        furniture_positions={
            **_CORNER_FURNITURE,
            "furniture_bookshelf": (3.0, 3.5, 0.9, 0.0),  # blocking NE corner
        },
        goals=[(4.0, 4.0)],
        max_steps_per_goal=300,  # 25s
        max_collisions=0,
        description="Goal tucked in NE corner behind a bookshelf.",
    ),

    # ── 5. Cluttered Room (Hard) ──
    # Multiple obstacles in center force complex routing.
    ScenarioDefinition(
        name="Cluttered Room",
        difficulty="hard",
        robot_start=(-3.5, -3.5, 0.785),  # SW corner, facing NE
        furniture_positions={
            "furniture_table": (1.0, 2.0, 0.375, 0.3),
            "furniture_bookshelf": (-1.5, 1.0, 0.9, 0.0),
            "furniture_chair": (2.0, 0.0, 0.45, 0.5),
            "furniture_desk": (0.0, -1.0, 0.375, 0.8),
            "furniture_cabinet": (-1.0, -2.0, 0.7, 0.0),
            "furniture_couch": (-2.0, 0.0, 0.3, 1.57),
            "furniture_bin": (0.5, -2.5, 0.25, 0.0),
        },
        goals=[(3.5, 3.5)],
        max_steps_per_goal=360,  # 30s
        max_collisions=1,
        description="7 furniture pieces scattered in center. Must weave through.",
    ),

    # ── 6. Full Patrol (Hard) ──
    # 6 goals forming a patrol loop around the room.
    ScenarioDefinition(
        name="Full Patrol",
        difficulty="hard",
        robot_start=(0.0, 0.0, 0.0),
        furniture_positions={
            "furniture_table": (2.0, 2.0, 0.375, 0.3),
            "furniture_bookshelf": (-2.5, 2.5, 0.9, 0.0),
            "furniture_chair": (2.5, -1.5, 0.45, 0.0),
            "furniture_desk": (-2.0, -2.0, 0.375, 0.5),
            "furniture_cabinet": (0.0, 2.5, 0.7, 0.0),
            "furniture_couch": (-3.0, 0.0, 0.3, 0.0),
            "furniture_bin": (1.0, -3.0, 0.25, 0.0),
        },
        goals=[
            (-3.5, 3.5),   # G1: NW
            (3.5, 3.5),    # G2: NE
            (3.5, 0.0),    # G3: E
            (3.5, -3.5),   # G4: SE
            (-3.5, -3.5),  # G5: SW
            (-3.5, 0.0),   # G6: W
        ],
        max_steps_per_goal=180,  # 15s per goal
        max_collisions=3,
        description="6-goal patrol loop around room perimeter with furniture.",
    ),
]
