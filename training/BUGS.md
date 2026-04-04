# Training Bug Tracker

Found via deep audit of the RL training pipeline against Isaac Lab source code.

---

## Isaac Lab Step Order (Root Cause Reference)

From `manager_based_rl_env.py` lines 204-238:

```
1. termination_manager.compute()    ← line 204
2. reward_manager.compute()         ← line 208  ← ALL rewards computed here
3. reset envs (handle dones)
4. command_manager.compute()        ← line 232  ← goal resampling happens here
5. observation_manager.compute()    ← line 238
```

At reward time, `get_command()` returns `pos_command_b` — the body-frame vector computed
at step N-1's `command_manager.compute()`. The current step's commands haven't been
updated yet. This makes rewards that use `get_command()` for proximity one step stale.

---

## ✅ Bug 1 — `goal_reached(+100)` structurally never fires

**File**: `training/nova_carter_patrol/mdp/rewards.py:154-166`
**Status**: FIXED

### Root Cause

`goal_reached()` uses `get_command()` → `pos_command_b` (stale body-frame vector from N-1).
`GoalReachPose2dCommand._update_command()` runs in step 4 (`command_manager.compute()`),
AFTER rewards (step 2). When the robot arrives within 0.75m:

- Step N-1 physics: robot moves to 0.6m from goal
- Step N-1 `command_manager.compute()` (step 4): `error_pos_2d = 0.6m < 0.75m` → **resample** → `pos_command_b` now points to NEW goal (3-5m away)
- Step N rewards (step 2): `goal_reached()` reads `pos_command_b` = distance to NEW goal >> 0.75m → **returns 0**

There is no step where `|pos_command_b| < 0.75m` without the goal having already been
resampled. **The +100 arrival bonus is dead code.** This explains why peak episode
rewards are ~150-200 instead of the expected ~600 (6 goals × +100 each).

### Fix

Use world-frame coordinates directly — `pos_command_w` at reward time = OLD goal (not
yet resampled this step), `root_pos_w` = current physics position:

```python
# Before (broken):
command = env.command_manager.get_command(command_name)
dist = torch.norm(command[:, :2], dim=-1)

# After (fixed):
cmd_term = env.command_manager.get_term(command_name)
dist = torch.norm(
    cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
    dim=1
)
```

---

## ✅ Bug 2 — `position_tracking_tanh` proximity gradient same issue

**File**: `training/nova_carter_patrol/mdp/rewards.py:92-105`
**Status**: FIXED

Same root cause as Bug 1. The tanh proximity gradient is most valuable in the final
0.3-0.5m of approach. But in the step the robot enters that zone, the command was just
resampled — `pos_command_b` points to the new far-away goal, so the reward returns ~0
instead of ~1.0 precisely when it should be pulling hardest.

### Fix

Same world-frame approach:

```python
cmd_term = env.command_manager.get_term(command_name)
dist = torch.norm(
    cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
    dim=1
)
return 1.0 - torch.tanh(dist / std)
```

---

## ✅ Bug 3 — `delta_progress` uses stale command (minor, mostly works)

**File**: `training/nova_carter_patrol/mdp/rewards.py:29-58`
**Status**: FIXED

`d_current = torch.norm(command[:, :2])` uses `pos_command_b` from step N-1, measuring
distance from P_{N-1} to goal instead of P_N. For normal navigation the error is small
(~0.025m/step at 0.3 m/s). The large-jump filter (>1m) already correctly handles goal
resampling spikes. This is a minor accuracy issue, not a fundamental failure.

### Fix

Use world-frame distance for true delta measurement:

```python
cmd_term = env.command_manager.get_term(command_name)
d_current = torch.norm(
    cmd_term.pos_command_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2],
    dim=1
)
```

---

## 🟡 Bug 4 — Wrong curriculum term name causes silent exceptions

**File**: `training/train_nova_carter.py:88`
**Status**: Open

`_get_curriculum_stage()` calls `curriculum_manager.get_term("terrain_levels")` — wrong
name, should be `"navigation_difficulty"`. Throws an exception on every render frame,
silently swallowed by try/except, falls back to a loop that finds the term anyway. Works
via fallback but generates exception overhead every step during video recording.

**Fix**: Change `"terrain_levels"` → `"navigation_difficulty"` on line 88.

---

## 🟡 Bug 5 — CheckpointCallback imported but never used

**File**: `training/train_nova_carter.py:47, ~line 466`
**Status**: Open

`CheckpointCallback` is imported but not in the callbacks list. Only `best_model.zip`
and final `model.zip` are saved. Early runs (234706, 234721, 234736) show "checkpoints
only" in EXPERIMENTS.md — the callback existed before and was accidentally dropped.
3+ jobs have timed out with no resumable intermediate checkpoint.

**Fix**: Add to callbacks:
```python
CheckpointCallback(
    save_freq=n_steps_per_rollout * args_cli.num_envs * 50,  # every 50 iters
    save_path=log_dir,
    name_prefix="checkpoint",
)
```

---

## 🔵 Issue 6 — Collision detection needs diagnostic

**File**: `training/nova_carter_patrol/nova_carter_patrol_env_cfg.py:406-410`
**Status**: Needs investigation

`undesired_contacts` structure looks correct (uses `net_forces_w_history`, `body_ids`
resolved from `body_names="chassis_link"`). Debug diagnostic already in code
(`train_nova_carter.py:106-111`) prints `num_bodies` and `max_force_history` on first
render step.

Check SLURM logs for jobs 238333 or 238291 for line:
```
[DEBUG] contact_sensor num_bodies=X, max_force_history=Y.YYYYN
```

- If `num_bodies=0`: sensor path didn't resolve — investigate `activate_contact_sensors`
  or try explicit `filter_prim_paths_expr`
- If `num_bodies=1` but `max_force_history≈0`: threshold 1.0N is too high → lower to 0.1N

---

## ✅ Fixed Items Summary

| Bug | Fix Location | What Changed |
|-----|-------------|--------------|
| Bug 1 | `rewards.py: goal_reached()` | Use `pos_command_w` + `root_pos_w` instead of `get_command()` |
| Bug 2 | `rewards.py: position_tracking_tanh()` | Same world-frame fix |
| Bug 3 | `rewards.py: delta_progress()` | Same world-frame fix |
