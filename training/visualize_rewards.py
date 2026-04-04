"""Reward calibration visualizer — no Isaac Sim required.

Shows the effect of Isaac Lab's dt multiplication on reward weights,
and simulates how rewards accumulate across a typical episode.

Run with:
    python3 training/visualize_rewards.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config (mirrors nova_carter_patrol_env_cfg.py) ──────────────────────────
PHYSICS_DT   = 1.0 / 60.0   # sim.dt
DECIMATION   = 5             # steps per RL action
RL_DT        = PHYSICS_DT * DECIMATION   # 0.0833s per RL step
RL_HZ        = 1.0 / RL_DT              # ~12 Hz

WEIGHTS = {
    "goal_reached":      100.0,
    "delta_progress":      3.0,   # per metre closed
    "distance_to_goal":    1.0,   # max when dist=0
    "collision":         -10.0,   # per contact step
    "time_penalty":       -0.05,
}

EFFECTIVE = {k: v * RL_DT for k, v in WEIGHTS.items()}

# ── Plot layout ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    f"Reward Calibration  |  physics {1/PHYSICS_DT:.0f} Hz  ×  decimation {DECIMATION}"
    f"  =  RL {RL_HZ:.1f} Hz  (dt = {RL_DT:.4f}s)",
    fontsize=13, fontweight="bold"
)

colors_pos = "#4CAF50"
colors_neg = "#F44336"
colors_light = "#90CAF9"

# ── Panel 1: Weight vs Effective value ──────────────────────────────────────
ax = axes[0]
ax.set_title("Config Weight  vs  Effective per RL Step", fontweight="bold")

terms     = list(WEIGHTS.keys())
weights   = [WEIGHTS[t] for t in terms]
effective = [EFFECTIVE[t] for t in terms]

x = np.arange(len(terms))
w = 0.35

bars1 = ax.bar(x - w/2, weights,   w, label="Config weight", color=colors_light, edgecolor="navy", linewidth=0.8)
bars2 = ax.bar(x + w/2, effective, w, label=f"Effective (× {RL_DT:.4f}s)", color=[colors_pos if v >= 0 else colors_neg for v in effective], edgecolor="black", linewidth=0.8)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(terms, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Reward value")
ax.legend(fontsize=9)

# Annotate effective values
for bar, val in zip(bars2, effective):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.3 if val >= 0 else -0.6),
            f"{val:+.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── Panel 2: Breakeven — collision steps per goal ───────────────────────────
ax = axes[1]
ax.set_title("How Many Collision Steps Wipe Out N Goals?", fontweight="bold")

goal_val      = EFFECTIVE["goal_reached"]           # +8.33 per goal
collision_val = abs(EFFECTIVE["collision"])          # 0.83 per step

goals_range = np.arange(1, 7)
breakeven_steps   = (goals_range * goal_val) / collision_val
breakeven_seconds = breakeven_steps / RL_HZ

ax.bar(goals_range, breakeven_steps, color=colors_neg, alpha=0.7, edgecolor="black", linewidth=0.8)

# Annotate with seconds
for g, steps, secs in zip(goals_range, breakeven_steps, breakeven_seconds):
    ax.text(g, steps + 0.3, f"{secs:.1f}s", ha="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Number of goals reached")
ax.set_ylabel("Collision steps to wipe out (RL steps)")
ax.set_xticks(goals_range)
ax.set_xticklabels([f"{g} goal{'s' if g > 1 else ''}" for g in goals_range], fontsize=9)

# Reference line: 1 second of contact
one_sec_steps = RL_HZ
ax.axhline(one_sec_steps, color="orange", linewidth=1.5, linestyle="--")
ax.text(6.4, one_sec_steps + 0.2, "1s contact", color="orange", fontsize=8, va="bottom")

ax.set_xlim(0.4, 6.8)

# ── Panel 3: Simulated episode reward ───────────────────────────────────────
ax = axes[2]
ax.set_title("Simulated Episode — 6 Goals, Some Collisions", fontweight="bold")

# Episode: 6 goals reached, with some contact steps along the way
# Roughly: navigate to each goal, maybe bump a wall once per goal
episode_length = 600   # RL steps (~50 seconds)
steps = np.arange(episode_length)
cumulative = np.zeros(episode_length)

rng = np.random.default_rng(42)

# Simulate per-step reward
reward_per_step = np.full(episode_length, EFFECTIVE["time_penalty"])  # baseline

# 6 goal arrivals at roughly equal spacing
goal_steps = [80, 160, 250, 340, 430, 520]
for gs in goal_steps:
    if gs < episode_length:
        reward_per_step[gs] += EFFECTIVE["goal_reached"]

# Delta progress: robot moves ~0.1m/step toward goal on average (some steps negative)
avg_delta = 0.05  # metres per step toward goal
reward_per_step += EFFECTIVE["delta_progress"] * (avg_delta + rng.normal(0, 0.03, episode_length))

# Collision bursts: ~5 steps each, once per segment
collision_starts = [40, 130, 200, 380, 490]
for cs in collision_starts:
    for i in range(5):
        if cs + i < episode_length:
            reward_per_step[cs + i] += EFFECTIVE["collision"]

# Distance to goal contribution (small, continuous)
reward_per_step += EFFECTIVE["distance_to_goal"] * rng.uniform(0, 0.5, episode_length)

cumulative = np.cumsum(reward_per_step)

ax.plot(steps, cumulative, color="#1565C0", linewidth=1.5, label="Cumulative reward")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

# Mark goal arrivals
for gs in goal_steps:
    if gs < episode_length:
        ax.axvline(gs, color=colors_pos, linewidth=1.0, alpha=0.6)
        ax.text(gs, cumulative[gs] + 1.5, f"+{EFFECTIVE['goal_reached']:.1f}",
                color=colors_pos, fontsize=7, ha="center", fontweight="bold")

# Mark collision bursts
for cs in collision_starts:
    if cs < episode_length:
        ax.axvspan(cs, min(cs + 5, episode_length), alpha=0.15, color=colors_neg)

# Annotations
goal_patch      = mpatches.Patch(color=colors_pos, alpha=0.6, label=f"Goal arrival (+{EFFECTIVE['goal_reached']:.2f})")
collision_patch = mpatches.Patch(color=colors_neg, alpha=0.3, label=f"Collision burst (-{abs(EFFECTIVE['collision']):.2f}/step)")
ax.legend(handles=[goal_patch, collision_patch], fontsize=8, loc="upper left")

ax.set_xlabel("RL step")
ax.set_ylabel("Cumulative episode reward")

final_reward = cumulative[-1]
ax.text(0.98, 0.05, f"Final: {final_reward:.1f}", transform=ax.transAxes,
        ha="right", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

# ── Save & show ──────────────────────────────────────────────────────────────
plt.tight_layout()
out = "training/reward_calibration.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
