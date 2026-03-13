# RL Training Pipeline for Nova Carter Security Patrol

## Context

The team is responsible for the RL training piece of a BIM-to-Sim robot deployment framework. Another team handles the navigation graph/path planning. We need to build the RL policy that handles **local navigation** — moving the robot from waypoint to waypoint while avoiding obstacles in indoor building environments. No RL infrastructure exists in the repo yet.

## Approach: Isaac Lab (Manager-Based Workflow)

Isaac Lab 2.3.x is NVIDIA's official RL framework for Isaac Sim 5.1.0. It provides:
- **GPU-parallelized environments** — 256-512 Nova Carters training simultaneously on a T4
- **Built-in RL algorithms** — PPO via Stable-Baselines3, rl_games, RSL-RL, SKRL
- **Manager-Based architecture** — modular observation, action, reward, and termination managers
- **Domain randomization** — built-in for sim-to-real transfer
- **Multi-GPU support** — scales across Rosie's GPU partitions

---

## Phase 1: Install Isaac Lab on Rosie

Isaac Sim 5.1.0 is already installed at `~/isaacsim`. We install Isaac Lab alongside it.

**Create `scripts/install_isaac_lab.sh`**:
```bash
#!/bin/bash
# Run on a GPU node (srun --partition=teaching --gres=gpu:1 --time=1:00:00 --pty bash)

# Set Isaac Sim paths
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Clone Isaac Lab 2.3.x (compatible with Isaac Sim 5.1.0)
cd "${HOME}"
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2

# Create symlink to Isaac Sim installation
ln -s "${ISAACSIM_PATH}" _isaac_sim

# Install Isaac Lab + RL frameworks
./isaaclab.sh --install  # installs core + extensions
./isaaclab.sh --install sb3  # Stable-Baselines3
./isaaclab.sh --install rl_games  # rl_games (fast PPO)
```

**Verify installation**:
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

**Update `.gitignore`** — add `checkpoints/`, `tb_logs/`, `logs/`

---

## Phase 2A: Flat-World Prototype Environment (No Building Required)

Build and validate the full RL pipeline on a **flat ground plane with random box obstacles**. This decouples RL development from the building USD and navigation graph, which are not yet available. The environment, observations, actions, rewards, and training loop are all identical to the final version — only the scene geometry changes later.

### Why Flat-World First

- **No dependency on building USD** — the TestBuilding with physics/collisions is not ready
- **No dependency on nav team waypoints** — goals are random (x, y) positions on the ground plane
- **90% of the work is the same** — env config, reward shaping, PPO tuning, SLURM scripts all carry over
- **Faster iteration** — flat scenes are lighter, faster to load, easier to debug
- **Clear upgrade path** — swap `GroundPlaneCfg` for `BuildingSceneCfg` and random goals for waypoint goals in Phase 2C

### Flat-World Scene

The scene spawns N parallel copies of:
- **Ground plane** — Isaac Lab's built-in `GroundPlaneCfg`
- **Nova Carter robot** — from `Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`
- **Random box obstacles** — 5-10 rigid cuboids scattered at random positions per env (via `EventManager` randomization on reset)
- **Goal marker** — visual sphere at target position (for debugging)
- **LiDAR sensor** — `RotatingLidarPhysX` on robot chassis, 20 beams, 360°, 10m range

### Goal Generation (Flat-World)

Instead of sampling from building waypoints, `CommandManager` samples random (x, y) positions within a radius (e.g., 3-8m from robot spawn). This tests the same policy interface — the robot still receives a relative goal vector and must navigate to it while avoiding obstacles.

### Transition to Phase 2C

When the building USD and waypoints are ready:
1. Replace `GroundPlaneCfg` with building USD scene
2. Replace random goal sampling with waypoint-based `CommandManager`
3. Adjust obstacle proximity thresholds for indoor corridors
4. Retrain (fine-tune from flat-world checkpoint or train from scratch)

---

## Phase 2B: Migrate to Real Wheel Physics (After Phase 2A works)

Phase 2A uses the Ridgeback robot with dummy joints (`NonHolonomicAction`) because it's the fastest path to a working pipeline. Once the RL loop is validated, we swap to a real differential-drive robot with actual wheel physics for better sim-to-real transfer.

**Why not start here**: Dummy joints are a known-working approach in Isaac Lab. Real wheel physics is validated by the community (see below) but adds friction tuning and a custom action term. Get the pipeline working first, then improve fidelity.

**What changes**:

| Component | Phase 2A (current) | Phase 2B (upgrade) |
|-----------|--------------------|--------------------|
| Robot USD | Ridgeback (dummy joints + Franka arm) | JetBot, Turtlebot3, or Create3 (real wheels) |
| Action term | `NonHolonomicActionCfg` (built-in) | Custom `DifferentialDriveActionCfg` (~50 lines) |
| Wheel control | Invisible sliding rails | `set_joint_velocity_target()` on real wheel joints |
| Everything else | Same | Same |

**Custom `DifferentialDriveAction`**: A Manager-Based `ActionTerm` that takes `[linear_vel, angular_vel]` and converts to left/right wheel velocity targets:
```
v_left  = (v - ω * wheel_base/2) / wheel_radius
v_right = (v + ω * wheel_base/2) / wheel_radius
```
Then calls `asset.set_joint_velocity_target()`. The policy's action space and obs space stay identical — only the physics layer changes.

**Robot choice**: Any differential-drive robot from Isaac Sim's asset library. Candidates:
- **JetBot** — used in Isaac Lab's official RL walkthrough with real wheel joints
- **Turtlebot3** — most ROS 2 sim-to-real precedent (RL-Navigation used it)
- **iRobot Create3** — clean differential drive, no arm

**Prior art**:
- [Isaac Lab JetBot Walkthrough](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_gt.html) — NVIDIA's official tutorial uses `set_joint_velocity_target()` on real JetBot wheel joints with PPO. Confirms this approach works.
- [RL-Navigation (sahars93)](https://github.com/sahars93/RL-Navigation) — trained JetBot in Isaac Sim, zero-shot transferred to real TurtleBot4 Lite. Used `DifferentialController` with velocity commands. Paper: arXiv:2501.02902 (Jan 2025).
- [WheeledLab (UW)](https://github.com/UWRobotLearning/WheeledLab) — real wheel physics in Isaac Lab with custom action terms. Achieved zero-shot sim-to-real on RC cars (Ackermann, not diff-drive, but validates the approach). Paper: arXiv:2502.07380 (Feb 2025).
- [Isaac Lab Discussion #1043](https://github.com/isaac-sim/IsaacLab/discussions/1043) — PhysX added native cylinder collision support (early 2025), fixing the old n-gon wheel approximation that originally motivated dummy joints.
- [Isaac Lab Discussion #2473](https://github.com/isaac-sim/IsaacLab/discussions/2473) — Carter V1 differential drive attempt. Documented a backward-motion bug when angular velocity is high: clamp wheel velocities or constrain `ω ≤ 2V/b`.

---

## Phase 2C: Full Building Environment (When Ready)

Swap in the real building and navigation waypoints. Everything else (obs, actions, rewards, terminations, training scripts) carries over from Phase 2A/2B.

**Blocked on**:
- TestBuilding USD with collision physics (`assets/environments/msoe_building/TestBuilding_physics.usd`)
- Navigation team waypoint definitions (`training/waypoints.py`)

---

## Phase 2 (Shared): Create Navigation Environment (Manager-Based)

The core deliverable. A Manager-Based RL environment following Isaac Lab conventions. The config structure below is shared across Phase 2A (flat-world), Phase 2B (real wheels), and Phase 2C (building).

### File Structure

```
~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/
└── navigation/
    └── carter_patrol/
        ├── __init__.py              # Gymnasium registration
        ├── carter_patrol_env_cfg.py # Full environment config (scene, obs, actions, rewards, etc.)
        └── mdp/                     # Custom MDP components
            ├── __init__.py
            ├── observations.py      # Observation term functions
            ├── rewards.py           # Reward term functions
            ├── terminations.py      # Termination conditions
            └── commands.py          # Goal waypoint command generator
```

Additionally in our repo:
```
~/isaacsim/isaac-sim-msoe/
├── scripts/
│   ├── install_isaac_lab.sh         # Isaac Lab installer
│   ├── submit_training.sh           # SLURM training job
│   ├── submit_eval.sh               # SLURM evaluation job
│   └── train_carter_patrol.py       # Training entry point (thin wrapper)
├── configs/
│   └── carter_patrol/
│       ├── ppo_base.yaml            # PPO hyperparameters
│       └── env_testbuilding.yaml    # TestBuilding environment config
└── training/
    ├── PLAN.md                      # This file
    ├── evaluate.py                  # Evaluation + metrics collection
    └── waypoints.py                 # Waypoint definitions (shared with nav team)
```

### Scene Configuration

The scene spawns N parallel copies of:
- **Nova Carter robot** — from `Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`
- **Building environment** — from `assets/environments/msoe_building/TestBuilding_physics.usd`
- **Goal marker** — visual sphere at target waypoint (for debugging)
- **LiDAR sensor** — `RotatingLidarPhysX` on robot chassis, 20 beams, 360°, 10m range

### Observation Space (26-dim float vector)

Defined via `ObservationManager` with an `ObservationGroupCfg`:

| Component | Dims | Source | Notes |
|-----------|------|--------|-------|
| LiDAR range scan | 20 | `RotatingLidarPhysX` | 20 beams, 360°, normalized by max range |
| Relative goal position | 2 | Computed | `[dx, dy]` in robot's local frame (egocentric) |
| Relative goal heading | 2 | Computed | `[cos(theta), sin(theta)]` avoids angle wrapping |
| Robot linear velocity | 1 | Base velocity | Forward speed (m/s) |
| Robot angular velocity | 1 | Base velocity | Yaw rate (rad/s) |

### Action Space (2-dim continuous)

Defined via `ActionManager`:

| Action | Range | Unit |
|--------|-------|------|
| linear_velocity | [-0.5, 0.5] | m/s |
| angular_velocity | [-1.0, 1.0] | rad/s |

Mapped to Nova Carter's `DifferentialController` (wheel_radius=0.14, wheel_base=0.4). **Action repeat** = 5 substeps (60 Hz physics -> 12 Hz RL control).

### Reward Function

Defined via `RewardManager` with weighted `RewardTermCfg` entries:

| Term | Weight | Function | Purpose |
|------|--------|----------|---------|
| `goal_progress` | 5.0 | `prev_dist - curr_dist` | Dense signal toward waypoint |
| `goal_reached` | 10.0 | Binary, within 0.5m | Sparse terminal bonus |
| `collision` | -10.0 | Min LiDAR < 0.15m | Hard penalty |
| `obstacle_proximity` | -1.0 | `max(0, 1 - min_lidar/0.5)` | Soft avoidance within 0.5m |
| `action_smoothness` | -0.1 | `norm(action - prev_action)` | Penalize jerk |
| `time_penalty` | -0.01 | Constant per step | Encourage efficiency |

### Termination & Reset

Defined via `TerminationManager`:

**Terminated (task failure/success)**:
- Goal reached: distance to waypoint < 0.5m
- Collision: minimum LiDAR range < 0.15m

**Truncated**:
- Max episode length: 1000 steps (~83 seconds at 12 Hz control)
- Robot tipped over: roll or pitch > 30 degrees

**Reset**:
- Teleport robot to random spawn waypoint via `set_world_pose`
- Zero out velocities
- Sample new goal waypoint from building waypoint set
- 5 physics settle steps

### Goal Command Generator

Defined via `CommandManager` — samples random goal waypoints from a predefined set per building. The navigation team's waypoint definitions feed directly into this.

---

## Phase 3: Training

### PPO Hyperparameters (starting point)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Learning rate | 3e-4 |
| Num envs | 256 (T4), 512+ (V100/H100) |
| Steps per rollout | 512 |
| Mini-batch size | 256 |
| Epochs per update | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.01 |
| Total timesteps | 1M (initial), 5M+ (full) |

### Training Command

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/carter_patrol/train.py \
  --task Isaac-Carter-Patrol-v0 \
  --num_envs 256 \
  --headless \
  --max_iterations 5000
```

Or via our SLURM wrapper:
```bash
cd ~/isaacsim/isaac-sim-msoe
sbatch scripts/submit_training.sh
```

### SLURM Script (`scripts/submit_training.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=rl-carter-patrol
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/rl_train_%j.out
#SBATCH --error=logs/rl_train_%j.err

export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export OMNI_KIT_ACCEPT_EULA=YES

cd "${HOME}/IsaacLab"
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-Carter-Patrol-v0 \
  --num_envs 256 \
  --headless \
  --max_iterations 10000
```

### Expected Performance

- 256 parallel envs on T4 at 12 Hz control ~ 3K steps/sec
- 1M timesteps ~ 5-6 minutes
- 5M timesteps ~ 30 minutes
- Dramatically faster than single-environment training

---

## Phase 4: Evaluation (`training/evaluate.py`)

Load trained policy checkpoint, run N=100 evaluation episodes, collect:

| Metric | Description |
|--------|-------------|
| **Success rate** | % episodes reaching goal waypoint |
| **Collision rate** | % episodes ending in collision |
| **Path efficiency** | straight-line distance / actual path |
| **Avg time to goal** | Steps for successful episodes |
| **Min obstacle clearance** | Closest LiDAR reading per episode |
| **Avg episode reward** | Cumulative reward |

Optional: record video of best/worst episodes using existing `VideoRecorder` + `FollowCamera`.

---

## Phase 5: Integration with Navigation Team

### Interface Contract

The RL policy handles **waypoint-to-waypoint** local navigation. The nav team provides the sequence.

```
Nav team: [W0, W1, W2, ..., Wn]  (patrol route)
    |
RL policy: receives W_i as goal -> navigates -> reports "reached"
    |
Nav team: issues W_{i+1}
    |
... repeat until patrol complete
```

### Shared Artifact

`training/waypoints.py` — building waypoint definitions used by both teams:
- Nav team defines graph nodes here
- RL training samples start/goal pairs from these nodes
- Evaluation tests all waypoint pairs for success rate

---

## Implementation Order

### Phase 1: Installation (DONE)
1. ~~**Install Isaac Lab** on Rosie~~ — IsaacLab v0.54.2 verified
2. **Verify** with a built-in Isaac Lab example (e.g., Ant locomotion)

### Phase 2A: Flat-World Prototype (No external dependencies)
3. **Create the environment config** (`carter_patrol_env_cfg.py`) — flat ground + random box obstacles, obs, actions
4. **Create MDP components** (`mdp/`) — reward functions, termination conditions, custom observations
5. **Register the environment** (`Isaac-Carter-Patrol-Flat-v0`) and test with random actions
6. **Run PPO training** with 256 envs, 1M steps on flat world
7. **Evaluate** trained policy — can the robot reach random goals while avoiding boxes?
8. **Iterate** on reward shaping based on results

### Phase 2B: Real Wheel Physics (After Phase 2A works)
9. **Write `DifferentialDriveActionCfg`** — custom action term in `carter_patrol/mdp/actions.py`
10. **Choose robot** — pick a diff-drive robot USD, configure `ArticulationCfg` with `ImplicitActuatorCfg` on wheel joints
11. **Update env config** — swap robot + action term, adjust LiDAR mount point
12. **Retrain** — compare reward curves and success rates vs Phase 2A dummy-joint baseline
13. **Domain randomization** — add friction, mass, and actuator gain randomization for sim-to-real (WheeledLab's key finding)

### Phase 2C: Building Environment (When building USD + waypoints are ready)
14. **Swap scene** — replace ground plane with TestBuilding USD
15. **Define building waypoints** (`training/waypoints.py`) with nav team
16. **Retrain or fine-tune** in building environment
17. **Evaluate** on building-specific waypoint pairs

### Phase 3-5: Training at scale, evaluation, integration (unchanged)

## Verification

### Phase 2A Checkpoints
1. ~~Isaac Lab installs cleanly on Rosie GPU node~~ — DONE
2. Built-in Isaac Lab example (Ant) trains successfully
3. `Isaac-Carter-Patrol-Flat-v0` env runs with random actions, correct obs shape (26,)
4. Training on flat world shows reward curve trending upward in TensorBoard
5. Trained policy navigates to random goals while avoiding box obstacles (>50% success rate)

### Phase 2B Checkpoints
6. Custom `DifferentialDriveAction` passes smoke test (env steps with correct action shape)
7. Real-wheel robot trains to comparable reward curve as Phase 2A dummy-joint baseline
8. No backward-motion bug (wheel velocities clamped correctly at high turn rates)
9. Domain randomization (friction, mass) doesn't crash training, improves eval robustness

### Phase 2C Checkpoints
10. Building USD loads in Isaac Lab scene with working collisions
11. `Isaac-Carter-Patrol-v0` env runs with building + waypoints
12. Trained policy achieves >50% success rate on building waypoint pairs
13. Evaluation script produces metrics table + optional video

## Reference Files

- `origin/del_branch:rl/isaaclab_jetbot_env.py` — Del's Isaac Lab DirectRLEnv with JetBot, LiDAR, curriculum learning (uses kinematic teleport, not real wheel physics — good reference for obs/reward design)
- `origin/del_branch:navigation/graph.py` — Clean navigation graph with Dijkstra pathfinding (reusable for Phase 2B/5)
- `simulations/robot/jetbot_hospital_demo.py` — Nova Carter spawn + DifferentialController pattern
- `simulations/testbuilding_overhead.py` — TestBuilding USD loading with 0.01 scale + Z offset
- `sim_utils/video_capture.py` — VideoRecorder for evaluation videos
- `sim_utils/follow_camera.py` — FollowCamera for 3rd-person views
- `sim_utils/building_physics.py` — Physics/collision setup for buildings
- Isaac Lab built-in navigation examples at `isaaclab_tasks/manager_based/navigation/`
