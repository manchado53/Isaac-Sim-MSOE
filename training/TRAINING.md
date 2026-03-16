# Nova Carter Patrol — RL Training Documentation

## Overview

This directory trains a PPO policy for the NVIDIA Nova Carter robot to navigate indoor environments — moving between goal positions while avoiding obstacles. The robot uses 2D LiDAR and relative goal information to make decisions, outputting linear and angular velocity commands.

Training runs on MSOE's Rosie supercomputer using NVIDIA Isaac Lab (GPU-parallelized RL) with Stable-Baselines3 PPO.

---

## Architecture

### Robot: Nova Carter

See [NOVA_CARTER_SPECS.md](NOVA_CARTER_SPECS.md) for full robot specifications, rigid body structure, collision geometry, and sensor details.

### Environment: 10m x 10m Walled Room

A flat-world prototype with randomized furniture obstacles (table, bookshelf, chair, desk, cabinet, couch, trash bin). Each piece randomizes position/orientation on episode reset. The robot spawns near the center (+-1.0m) to avoid spawning inside furniture.

### Observation Space (26-dim)

| Component | Dims | Description |
|-----------|------|-------------|
| LiDAR scan | 20 | 360-degree, 20 rays, normalized to [0,1] by 10m max range |
| Goal position | 2 | Relative (x, y) in robot body frame |
| Goal heading | 2 | (cos, sin) of heading error — avoids angle wrapping |
| Forward velocity | 1 | Robot's forward speed (m/s) |
| Yaw rate | 1 | Robot's angular velocity (rad/s) |

### Action Space (2-dim continuous)

| Action | Range | Unit |
|--------|-------|------|
| Linear velocity | [-1, 1] scaled to [-0.5, 0.5] | m/s |
| Angular velocity | [-1, 1] scaled to [-1.0, 1.0] | rad/s |

Actions are converted to wheel velocities via differential drive kinematics:
```
v_left  = (v - omega * wheel_base/2) / wheel_radius
v_right = (v + omega * wheel_base/2) / wheel_radius
```

Control frequency is 12 Hz (60 Hz physics with decimation=5).

---

## Reward Function

The reward design uses dual-scale tanh position tracking with heading alignment, combined with goal-reached resampling for patrol behavior.

### Current Reward Terms

| Term | Function | Weight | Parameters | Purpose |
|------|----------|--------|------------|---------|
| `position_tracking` | `position_tracking_tanh` | +0.5 | std=2.0 | Coarse gradient from far away |
| `position_tracking_fine` | `position_tracking_tanh` | +0.5 | std=0.2 | Sharp signal for precise arrival |
| `heading_alignment` | `heading_alignment` | -0.2 | — | Penalize not facing the goal |
| `velocity_near_goal` | `velocity_near_goal` | -0.3 | threshold=0.5m | Gentle deceleration in final approach |
| `goal_reached` | `goal_reached` | +10.0 | threshold=0.5m | Large bonus for arrival |
| `collision` | `mdp.undesired_contacts` | -4.0 | contact_sensor, threshold=1.0N | Physics-based collision penalty via contact sensor |
| `obstacle_proximity` | `obstacle_proximity` | -1.0 | safe_dist=0.8m | Smooth LiDAR-based avoidance before contact |
| `action_smoothness` | `action_rate_l2` | -0.1 | — | Penalize jerky actions |
| `survival` | `is_alive` | +0.1 | — | Positive baseline to stabilize training |

### Key Reward Functions

**`position_tracking_tanh(std)`**: Returns `1 - tanh(distance / std)`. Using two instances with different `std` values provides gradient at all distances — the coarse term (std=2.0) gives signal from far away, while the fine term (std=0.2) provides a sharp peak near the goal for precise stopping.

**`velocity_near_goal(threshold)`**: When within `threshold` meters of the goal, returns the robot's speed (penalized by negative weight). This is the key term that teaches the robot to decelerate on approach rather than overshooting or orbiting.

**`heading_alignment`**: Returns `|heading_error|`, penalized so the robot learns to face the goal direction.

**`goal_reached(threshold)`**: Returns 1.0 when distance < threshold, 0.0 otherwise. Gives an explicit success signal on top of the continuous tanh rewards. Weight is +10.0 — large enough to strongly incentivize closing the final distance rather than hovering nearby.

**`mdp.undesired_contacts`** (Isaac Lab built-in): Uses the contact sensor on `chassis_link` to detect actual physics collisions. Threshold is 1.0N (force-based). Replaces the old LiDAR-based collision penalty which had a blind spot due to LiDAR-to-bumper offset (see NOVA_CARTER_SPECS.md).

### Goal-Reached Command Resampling

The environment uses `GoalReachPose2dCommand` (subclass of `UniformPose2dCommand`) which resamples the goal immediately when the robot reaches it. This follows Isaac Lab's `InHandReOrientationCommand` pattern:

1. **`_update_command()`** checks if `error_pos_2d < goal_reached_threshold` (0.5m)
2. When reached, calls `self._resample(goal_reset_ids)` to generate a new goal immediately
3. The base class `_resample()` increments `command_counter` and resets the fallback timer
4. `command_resample` termination ends the episode after 6 successful reaches

**Why this works**: Instead of the robot sitting idle for 7-12s after reaching a goal (waiting for the fixed timer), it immediately gets a new goal. This teaches patrol behavior (goal -> goal -> goal) and maximizes training signal per episode.

**Fallback timer**: Set to 15s (not infinity like InHand) because some goals may be behind furniture and unreachable. The timer ensures the robot always gets a new goal eventually.

### Reward Design History

The original reward used `GoalProgressReward` (delta-distance per step) + `goal_reached` (binary bonus at 0.5m). Problems:
- **No deceleration signal** — the robot learned to approach but not stop
- **Overshooting/orbiting** — moving fast near the goal was rewarded (closing distance) with no penalty
- **Goal-reached termination** created sparse signal — episodes ended abruptly, PPO had fewer samples

The tanh approach (introduced second) provided continuous gradients but used a fixed 10-15s resampling timer. The current design (Pattern B) keeps the tanh shaping and adds goal-reached resampling so the robot doesn't waste simulation time sitting at reached goals.

---

## Termination Conditions

| Condition | Type | Description |
|-----------|------|-------------|
| `time_out` | Truncation | Episode ends at 83s (1000 steps at 12 Hz) |
| `bad_orientation` | Termination | Roll or pitch > 30 degrees (robot tipped) |
| `command_resample` | Termination | Episode ends after 6 goal reaches (patrol complete) |

The `command_resample` termination uses Isaac Lab's built-in `mdp.command_resample` function. It terminates when `command_counter == num_resamples` (6) and the command timer expires. With `GoalReachPose2dCommand`, each goal reach increments `command_counter`, so after 6 successful reaches the episode ends. At ~5-10s per goal, episodes last 30-60s naturally, or the full 83s if goals take longer.

No `collision` termination — handled as reward penalty only. Terminating on collision causes a death spiral when the robot spawns near furniture (instant death -> no useful signal -> NaN gradients).

### Collision Detection

**Important**: The `ContactSensorCfg` must target `chassis_link` only, NOT `Robot/.*` (all bodies). The wheels and casters are always in contact with the ground, producing ~200N+ of force from gravity every step. Using `Robot/.*` makes every step look like a collision.

```python
# WRONG — detects ground contact through wheels (always ~200N)
contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")

# CORRECT — chassis doesn't touch ground normally, so any force = real collision
contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/chassis_link")
```

The contact sensor uses `net_forces_w` (shape: `(num_envs, num_bodies, 3)`) which reports total contact force on each tracked body. For filtered contacts (e.g., only detecting contact with specific obstacles), use `filter_prim_paths_expr` and read `force_matrix_w` instead — see [Isaac Lab Contact Sensor docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html).

Both training and evaluation now use the contact sensor for collision detection. The old LiDAR-based collision penalty was removed because the LiDAR-to-bumper offset (0.36m) made the threshold unreliable. LiDAR is still used for `obstacle_proximity` (proactive avoidance before contact). See NOVA_CARTER_SPECS.md for robot dimensions.

---

## PPO Hyperparameters

Configured in `nova_carter_patrol/agents/sb3_ppo_cfg.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | PPO (SB3) | |
| Policy | MlpPolicy | 2x128 hidden layers, ReLU |
| Learning rate | 3e-4 | |
| Num envs | 128 | Reduced from 256 — Nova Carter USD is heavier |
| Steps per rollout | 512 | |
| Mini-batch size | 256 | |
| Epochs per update | 10 | |
| Gamma | 0.99 | |
| GAE lambda | 0.95 | |
| Clip range | 0.2 | |
| Entropy coef | 0.01 | |
| Target KL | 0.02 | Early-stops policy update if KL diverges (prevents NaN) |
| Max grad norm | 0.5 | |
| VecNormalize | obs only | Reward normalization disabled — causes NaN with short episodes |

---

## Running Training

### Submit a Training Job

```bash
cd ~/isaac-sim-msoe
sbatch training/submit_training.sh
```

Training parameters are set in `training/train_config.py`:
```python
NUM_ENVS = 128
SEED = 42
MAX_ITERATIONS = 50      # bump to 1000+ for full training
VIDEO = False             # enable for training videos (costs RAM)
```

### Monitor Job Status

```bash
squeue -u $USER                          # check if running
cat training/logs/train_<JOBID>.out      # view stdout
cat training/logs/train_<JOBID>.err      # view stderr
```

### Resume from Checkpoint

Set `CHECKPOINT` in `train_config.py`:
```python
CHECKPOINT = "training/logs/sb3/NovaCarter-Patrol/2026-03-14_19-42-01/model.zip"
```

---

## Evaluation

Training automatically runs evaluation after completion. To run standalone:

```bash
# Evaluate latest run
sbatch training/submit_eval.sh

# Evaluate specific checkpoint
sbatch training/submit_eval.sh training/logs/sb3/NovaCarter-Patrol/<run>/model.zip
```

Evaluation records 6 video clips of 500 steps each, saved to `<run>/videos/eval/`.

---

## Comparing Training Runs with TensorBoard

All training runs log TensorBoard events to `training/logs/sb3/NovaCarter-Patrol/<timestamp>/`.

### Launch TensorBoard

From a Rosie login node:
```bash
tensorboard --logdir ~/isaac-sim-msoe/training/logs/sb3/NovaCarter-Patrol --port 6006
```

This loads **all runs** as separate series. To view a single run, point at its subdirectory:
```bash
tensorboard --logdir ~/isaac-sim-msoe/training/logs/sb3/NovaCarter-Patrol/2026-03-14_19-42-01 --port 6006
```

### View in Browser

Set up SSH port forwarding from your local machine:
```bash
ssh -N -L 6006:localhost:6006 <user>@dh-ood.hpc.msoe.edu
```

Then open http://localhost:6006 in your browser.

### Key Metrics to Watch

| Metric | Where | What to Look For |
|--------|-------|------------------|
| `rollout/ep_rew_mean` | Scalars | Should trend upward — primary training signal |
| `rollout/ep_len_mean` | Scalars | Should increase as robot survives longer |
| `train/loss` | Scalars | Should decrease and stabilize |
| `train/approx_kl` | Scalars | Should stay below `target_kl` (0.02) |
| `train/entropy_loss` | Scalars | Gradual decrease = policy becoming more confident |
| `train/explained_variance` | Scalars | Closer to 1.0 = value function is accurate |

### Comparing Runs

TensorBoard's left sidebar lets you toggle individual runs on/off. Use the smoothing slider to compare noisy reward curves. The "Relative" X-axis option aligns runs that started at different times.

---

## File Structure

```
training/
├── train_config.py                     # Centralized training parameters
├── train_nova_carter.py                # Training entry point (PPO + SB3)
├── eval_nova_carter.py                 # Evaluation with video recording
├── submit_training.sh                  # SLURM batch script for training
├── submit_eval.sh                      # SLURM batch script for evaluation
├── TRAINING.md                         # This file
├── PLAN.md                             # Development plan and phase history
├── logs/                               # SLURM logs + TensorBoard data
│   ├── train_<JOBID>.out/err           # SLURM stdout/stderr
│   └── sb3/NovaCarter-Patrol/          # Per-run directories
│       └── <timestamp>/
│           ├── PPO_1/events.out.*      # TensorBoard events
│           ├── params/                 # Saved env + agent configs
│           ├── model.zip               # Final trained policy
│           ├── model_vecnormalize.pkl  # Observation normalization stats
│           └── videos/                 # Training/eval video clips
└── nova_carter_patrol/                 # Isaac Lab environment package
    ├── __init__.py                     # Gymnasium registration
    ├── nova_carter_patrol_env_cfg.py   # Full environment config
    ├── agents/
    │   └── sb3_ppo_cfg.yaml            # PPO hyperparameters
    └── mdp/                            # Custom MDP components
        ├── __init__.py
        ├── actions.py                  # DifferentialDriveAction
        ├── commands.py                 # GoalReachPose2dCommand (patrol resampling)
        ├── observations.py             # LiDAR, goal, velocity obs
        ├── rewards.py                  # Reward functions
        └── terminations.py             # Termination conditions
```

---

## References and Inspiration

### Reward Shaping

- **Isaac Lab ANYmal Navigation** — The dual-scale `tanh` position tracking and heading alignment reward pattern comes from Isaac Lab's built-in ANYmal rough-terrain navigation task (`IsaacLab/source/isaaclab_tasks/.../navigation/mdp/rewards.py`). This is NVIDIA's recommended approach for goal-reaching with continuous commands.

### Sim-to-Real Transfer

- **RL-Navigation** (Sahar et al., 2025) — Trained a JetBot in Isaac Sim and achieved zero-shot transfer to a real TurtleBot4 Lite for indoor navigation. Used differential drive with velocity commands, similar to our approach.
  - Paper: [arXiv:2501.02902](https://arxiv.org/abs/2501.02902)
  - Code: [github.com/sahars93/RL-Navigation](https://github.com/sahars93/RL-Navigation)

- **WheeledLab** (UW Robot Learning Lab, 2025) — Real wheel physics in Isaac Lab with custom action terms. Achieved zero-shot sim-to-real on RC cars, validating that real wheel joints (not dummy/kinematic) transfer better.
  - Paper: [arXiv:2502.07380](https://arxiv.org/abs/2502.07380)
  - Code: [github.com/UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab)

### Isaac Lab Framework

- **Isaac Lab JetBot Walkthrough** — NVIDIA's official tutorial using `set_joint_velocity_target()` on real wheel joints with PPO. Confirms the differential drive action approach.
  - [Isaac Lab Training JetBot Tutorial](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_gt.html)

- **Isaac Lab Discussion #1043** — PhysX added native cylinder collision support (early 2025), fixing the old n-gon wheel approximation that originally motivated dummy joints in older Isaac Sim versions.

- **Isaac Lab Discussion #2473** — Carter V1 differential drive attempt. Documented a backward-motion bug when angular velocity is high. Solution: clamp wheel velocities or constrain `omega <= 2V / wheel_base`.

### Goal-Reached Resampling Pattern

- **Isaac Lab InHandReOrientationCommand** — The goal-reached resampling pattern (checking error in `_update_command()` and calling `self._resample()`) comes from Isaac Lab's in-hand manipulation task. Our `GoalReachPose2dCommand` adapts this for 2D navigation with a position threshold instead of orientation threshold, and uses a finite fallback timer instead of infinite.
  - Source: `IsaacLab/source/isaaclab_tasks/.../inhand/mdp/commands/orientation_command.py`

### Algorithm

- **Proximal Policy Optimization (PPO)** (Schulman et al., 2017) — The RL algorithm used for training. Chosen for stability and compatibility with Isaac Lab's vectorized environments.
  - Paper: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| NaN in rewards/loss | Reward variance collapse with VecNormalize | `norm_reward=False` in VecNormalize (already set) |
| Robot tips over immediately | Bad USD articulation root | Use Nova Carter (not Ridgeback) — has proper wheel joints |
| Reward stuck negative | Collision penalty dominates | Reduce collision weight, increase positive rewards |
| `bad_orientation` terminations | Robot flipping from collisions | Check furniture spawn zones don't overlap robot spawn |
| Training OOM | Too many envs for GPU memory | Reduce `NUM_ENVS` in `train_config.py` |
| No TensorBoard data | Run hasn't started PPO updates yet | Wait for first iteration (can take 1-2 min for scene loading) |

---

## GPU Partitions

| Partition | GPU | Time Limit | Use Case |
|-----------|-----|------------|----------|
| `teaching` | T4 (16GB) | 7 days | Development, short runs (default) |
| `batch` | T4 (16GB) | 2 days | Production training |
| `dgx` | V100 (32GB) | 21 days | Large-scale training (more envs) |
| `dgxh100` | H100 (80GB) | 21 days | Maximum performance |

Change partition in `submit_training.sh`: `#SBATCH --partition=<name>`
