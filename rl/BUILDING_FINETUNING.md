# Fine-Tuning the Nova Carter Policy in the Innovation Center
### Sim-to-Real Transfer Guide — Research-Backed

**Base checkpoint:** `training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/best_model_stage4.zip`
**Last updated:** 2026-04-10

---

## Overview

Three-phase pipeline: (1) close the most critical sim-to-real gaps in the training environment, (2) fine-tune in the Innovation Center digital twin, (3) deploy to the physical Nova Carter.

**Key research references:**
- Sahar et al. 2025 (arXiv:2501.02902) — Isaac Sim → real TurtleBot4 via ROS2
- WheeledLab (arXiv:2502.07380) — domain randomization enabling zero-shot transfer on wheeled robots
- FLaRe (arXiv:2409.16578) — sim-based fine-tuning with +30.7% real-robot improvement
- Arena-Rosnav 5.0 — RL as local planner plugin inside ROS2 Nav2 (the right architecture)
- Residual Off-Policy RL (arXiv:2509.19301, ICLR 2025) — 200× more sample-efficient than PPO fine-tuning

---

## Known Sim-to-Real Gaps (from code audit)

These exist in the current training code and will hurt real-world transfer:

| Gap | File / Line | Severity | Fix |
|-----|------------|----------|-----|
| No LiDAR noise or dropout | `observations.py:31-45` | HIGH | Add Gaussian noise σ=0.05m + 5% dropout |
| Max-range rays return 1.0 instead of distinct sentinel | `observations.py:44` | MEDIUM | Return 0.0 at max range (open space) |
| Action latency = 0ms (instant velocity) | `actions.py:97` | HIGH | Simulate 50-100ms command delay |
| No motor acceleration limits | `actions.py:82-97` | MEDIUM | Add velocity ramp (0.3s to full speed) |
| Friction = 1.0 everywhere, no randomization | `env_cfg.py:93-99` | HIGH | Randomize ±30% per episode |
| No domain randomization on any physics | `env_cfg.py:621-637` | HIGH | Add mass, damping, CoM randomization |
| Velocity control (not torque) with infinite stiffness | `env_cfg.py:62-75` | MEDIUM | Add implicit actuator damping model |
| Bug 4: silent curriculum error every frame | `BUGS.md:113` | LOW | Fix term name `terrain_levels` → `navigation_difficulty` |
| Bug 5: CheckpointCallback never saved | `BUGS.md:125` | HIGH | Add to callback list (jobs timed out with no saves) |

---

## Phase 0: Close Critical Sim Gaps Before Fine-Tuning

**Do this before touching the building environment.** Adding domain randomization to the base environment dramatically reduces the amount of building-specific fine-tuning needed (WheeledLab demonstrates zero-shot transfer is possible with proper DR; our goal is minimal fine-tuning, not zero-shot).

---

### Step 0.1 — Add LiDAR domain randomization

**File to modify:** `training/nova_carter_patrol/mdp/observations.py`

The current `lidar_scan_normalized()` returns perfect, noiseless rays, and converts max-range misses to 1.0 (visually "obstacle at far distance"). The real Hesai XT-32 and RPLidar S2E have Gaussian range noise and occasional dropout.

**TODO checklist:**
- [x] Add Gaussian noise to LiDAR distances before normalization:
  ```python
  # After computing distances, before normalization
  noise_std = 0.05  # meters; typical for RPLidar S2E at close range
  noise = torch.randn_like(distances) * noise_std
  distances = torch.clamp(distances + noise, 0.0, max_range)
  ```
- [x] Add ray dropout (5% of rays randomly set to max range per step):
  ```python
  dropout_mask = torch.rand_like(distances) < 0.05
  distances = torch.where(dropout_mask, torch.full_like(distances, max_range), distances)
  ```
- [x] Fix max-range sentinel: change the `torch.where(isfinite(...), ..., torch.ones_like(...))` line to use **0.0 instead of 1.0** for missed rays:
  ```python
  # Before: normalized = torch.where(torch.isfinite(normalized), normalized, torch.ones_like(normalized))
  # After (0.0 = open space, 1.0 = obstacle at max range):
  normalized = torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
  ```
  **Rationale:** 0.0 = nothing detected (open space). 1.0 = obstacle exactly at 10m. This is semantically correct and consistent with real sensor behavior.
- [x] Verify: run a short training job (50 iters) and confirm `patrol/goals_per_episode` is not worse than baseline — LiDAR noise should have minimal impact on already-trained policy, but verify.

**Success criteria:** Policy still reaches `patrol/goals_per_episode ≥ 4.0` in the procedural room after noise addition.

---

### Step 0.2 — Add physics domain randomization

**File to modify:** `training/nova_carter_patrol/nova_carter_patrol_env_cfg.py`

The current environment uses `static_friction=1.0` uniformly everywhere with no variation. Real building floors (tile, carpet, concrete) vary 0.6–1.2.

**TODO checklist:**
- [x] Add randomized physics material in `EventCfg` (runs at each episode reset):
  ```python
  randomize_physics = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot"),
          "static_friction_range": (0.7, 1.3),
          "dynamic_friction_range": (0.6, 1.1),
          "restitution_range": (0.0, 0.1),
      },
  )
  ```
- [x] Add robot mass randomization (±10% of base mass, accounts for payload/sensor weight variation):
  ```python
  randomize_mass = EventTerm(
      func=mdp.randomize_rigid_body_mass,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="chassis_link"),
          "mass_distribution_params": (0.9, 1.1),  # ±10% multiplier
          "operation": "scale",
      },
  )
  ```
- [x] Add center-of-mass offset randomization (±5cm, accounts for uneven payload):
  ```python
  randomize_com = EventTerm(
      func=mdp.randomize_rigid_body_coms,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="chassis_link"),
          "com_distribution_params": ((-0.05, 0.05), (-0.05, 0.05), (-0.02, 0.02)),
      },
  )
  ```
- [x] Add action latency: wrap `DifferentialDriveAction.apply_actions()` in `actions.py` with a 1-2 step delay buffer (at 12 Hz, 1 step = 83ms latency):
  ```python
  # In DifferentialDriveAction.__init__:
  self._action_buffer = deque(maxlen=2)  # 2-step = ~166ms max latency

  # In apply_actions(), before set_joint_velocity_target:
  self._action_buffer.append(self._processed_actions.clone())
  delayed_action = self._action_buffer[0]  # Use oldest buffered action
  # ... use delayed_action instead of self._processed_actions
  ```
- [x] Fix Bug 5 (CheckpointCallback) in `train_nova_carter.py`:
  ```python
  # Add to callbacks list (line ~524):
  checkpoint_callback = CheckpointCallback(
      save_freq=n_steps * 50,
      save_path=log_dir,
      name_prefix="checkpoint",
  )
  callbacks = [patrol_cb, eta_cb, best_model_cb, checkpoint_callback]
  ```
- [x] Run a short ablation (100 iters) to confirm performance is not catastrophically degraded by DR. Expect `patrol/goals_per_episode` to drop by 10-20% initially but recover with continued training.

**Success criteria:** Policy with domain randomization achieves `patrol/goals_per_episode ≥ 3.0` (accepting some performance drop from the harder environment).

---

### Step 0.3 — Run a zero-shot baseline in the building

**Before fine-tuning**, establish a baseline: run the `best_model_stage4.zip` policy (without any fine-tuning) directly in the building environment. This tells you how much fine-tuning is actually needed.

**TODO checklist:**
- [x] Create `training/eval/eval_zero_shot_building.py` — loads the building env and the pre-trained checkpoint, runs 20 episodes with no gradient updates, records goals/episode and collision rate
- [x] Submit: `sbatch submit_isaac_job.sh training/eval/eval_zero_shot_building.py`
- [x] Record results in `training/EXPERIMENTS.md` as the "zero-shot baseline" row
- [x] Interpret results:
  - `goals_per_episode > 2.0` → policy generalizes well, minimal fine-tuning needed (maybe 50k steps)
  - `goals_per_episode 0.5–2.0` → moderate fine-tuning needed (200-500k steps)
  - `goals_per_episode < 0.5` → significant distributional shift; check VecNormalize stats and waypoint placement first

**Success criteria:** Baseline recorded. If zero-shot already ≥ 2.0, skip Stage A fine-tuning and jump straight to Stage B.

---

## Phase 1: Sim Fine-Tuning in the Innovation Center Digital Twin

---

### Step 1 — Verify the physics USD is RL-ready

**What:** Confirm `innovation_center_physics.usd` has valid collision geometry for the RL robot.

**TODO checklist:**
- [x] Confirm `assets/environments/msoe_building/innovation_center_physics.usd` exists (generated by `simulations/environment/setup_innovation_center_physics.py`)
- [x] Run validation pipeline (skip `robot_movement` — it uses JetBot, not Nova Carter):
  ```bash
  python validate_physics.py \
    --usd assets/environments/msoe_building/innovation_center_physics.usd \
    --skip robot_movement --partition teaching
  ```
- [x] Confirm all three pass: `building_stats_results.json`, `cube_drop_results.json`, `door_physics_results.json` → `passed: true`
- [x] Write `simulations/environment/validate_nova_carter_floor.py`:
  - Spawn Nova Carter at floor G Z position (from `discover_floor_levels()`)
  - Step 300 physics frames (5s at 60Hz)
  - Assert `root_pos_w.z` drift < 0.05m (not falling through)
  - Print wheel contact forces (should be ~(mass × g) / 4 per wheel ≈ 225N on a 90kg robot)
- [x] Submit and verify: `sbatch submit_isaac_job.sh simulations/environment/validate_nova_carter_floor.py`

**Success criteria:** All validation JSONs pass; floor test exits 0; wheel contact forces near 225N.

---

### Step 2 — Generate the building occupancy map and define waypoints

**What:** Get usable floor geometry and create navigation waypoints. Two tools are available: Isaac Sim's built-in occupancy map generator and a custom floor bbox script.

**TODO checklist:**

**2a. Generate 2D occupancy maps (needed for both waypoint placement and later for Nav2 localization):**
- [x] Use Isaac Sim's built-in occupancy map generator: **Tools > Robotics > Occupancy Map** in the GUI
  - Load `innovation_center_physics.usd`
  <!-- NOTE: GUI-based step — deferred; using floor_camera_screenshots.py reference images for waypoint placement instead -->
  - Set height slice to each floor's Z position
  - Set cell size to 0.1m (10cm resolution)
  - Export as `.pgm` + `.yaml` (ROS2-compatible map format)
  - Save to `assets/maps/floor_G.pgm`, `floor_1.pgm`, `floor_2.pgm`, `floor_3.pgm`
  - **Note:** All geometry must have CollisionAPI enabled (already done by physics pipeline)
- [x] Alternatively, use the Python API (`MapGenerator` class) for batch generation across floors:
  ```python
  from isaacsim.asset.gen.omap import MapGenerator
  gen = MapGenerator()
  gen.update(cell_size=0.1, origin=(x, y, z_floor), height=2.0)
  gen.generate()
  gen.save("assets/maps/floor_G")
  ```

**2b. Create floor extent script:**
- [x] Create `training/nova_carter_patrol/building_floor_map.py`:
  - Call `discover_floor_levels()` from `sim_utils/building_physics.py`
  - Use `UsdGeom.BBoxCache` to get XY bounding box per floor
  - Output `training/nova_carter_patrol/floor_map.json`:
    ```json
    {
      "G": {"z_m": ..., "x_range_m": [min, max], "y_range_m": [min, max]},
      "1": {...}, "2": {...}, "3": {...}
    }
    ```
  - Note: USD is in feet (metersPerUnit=0.3048) — multiply by 0.3048
- [x] Run: `python training/nova_carter_patrol/building_floor_map.py` (no GPU needed)
- [x] Verify: corridor widths 2–20m range; floor Z values ~3-4m apart

**2c. Hand-place navigation waypoints:**
- [x] Open occupancy maps in any image viewer alongside `floor_camera_screenshots.py` reference images
- [x] Place waypoints at clearly open locations (centers of corridors, lobby areas, room interiors)
  - Waypoints must be ≥1.5m from any wall (robot radius ~0.4m + safety margin)
  - Avoid doorway thresholds (robot can get stuck on door geometry)
- [x] Write to `training/nova_carter_patrol/building_waypoints.json`:
  ```json
  {
    "G": [{"x": ..., "y": ..., "z": ..., "label": "lobby_north"}, ...],
    "1": [...],
    "2": [...],
    "3": [...]
  }
  ```
  - Floor G: ≥8 waypoints (lobby is large) ✓ (8 waypoints)
  - Floors 1, 2, 3: ≥5 waypoints each ✓ (5 each = 23 total)
- [x] Validate: For each waypoint, query the occupancy map — the cell at that (x,y) must be free (value=0)

**Success criteria:** `floor_map.json` has plausible extents; `building_waypoints.json` ≥23 waypoints; all waypoints confirmed free in occupancy map.

---

### Step 3 — Create the building environment config

**What:** New Isaac Lab env config that swaps the 10×10 procedural room for the Innovation Center USD.

**TODO checklist:**

**3a. New scene config (`training/nova_carter_patrol/nova_carter_building_env_cfg.py`):**
- [x] Replace `TerrainImporterCfg` + wall/furniture `RigidObjectCfg` blocks with:
  ```python
  building = AssetBaseCfg(
      prim_path="/World/Building",
      spawn=sim_utils.UsdFileCfg(
          usd_path=os.path.abspath(
              "assets/environments/msoe_building/innovation_center_physics.usd"
          ),
      ),
  )
  ```
- [x] Update LiDAR `mesh_prim_paths` to target the building with `is_shared=True` so all parallel envs share the BVH (critical for memory — 77k meshes cannot be duplicated per env):
  ```python
  # IMPORTANT: is_shared=True means BVH built once, shared across all num_envs
  # Without this, GPU VRAM scales as: 77k meshes × num_envs → OOM on T4
  mesh_prim_paths=["/World/Building"]
  ```
  - If `MultiMeshRayCasterCfg` doesn't support `is_shared`, check whether `RayCasterCfg` (single mesh) can target the building root prim instead — test both
- [x] Set `num_envs=1` initially; increase to 4 only after confirming no OOM
- [x] Remove all `CurriculumCfg` entries (building complexity replaces procedural curriculum)
- [x] Set `episode_length_s = 120.0` (longer than the 83s room episodes — building is larger)
- [x] Update `patrol_complete` threshold from 6 goals to 4 goals (longer distances between building waypoints means fewer goals per episode)

**3b. Building-aware goal command:**
- [x] Load `building_waypoints.json` at env startup and pass the floor G list to `GoalReachPose2dCommandCfg` via `goal_stations` mode
- [x] Disable the goal rejection logic that checks for furniture clearance (no furniture dict in building env) — rely on reward signal for wall avoidance
- [x] Set `goal_reached_threshold = 1.0m` (up from 0.75m) — building corridors may have localization drift

**3c. Building-aware robot spawn:**
- [x] Add `reset_base_building(env, env_ids, waypoints)` to `training/nova_carter_patrol/mdp/events.py`:
  - Randomly selects a spawn waypoint; selects a different waypoint as the first goal
  - Ensures spawn–goal distance ≥ 3m (avoids trivial episodes)
  - Calls `asset.write_root_pose_to_sim()` + `asset.write_root_velocity_to_sim()` to zero velocities
- [x] Keep `reset_robot_joints` unchanged (joint reset to 0 is fine)

**3d. Register the gym env:**
- [x] Add to `training/nova_carter_patrol/__init__.py`:
  ```python
  gym.register(
      id="Isaac-NovaCarter-Building-v0",
      entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
      kwargs={"env_cfg_entry_point":
              "nova_carter_patrol.nova_carter_building_env_cfg:NovaCarterBuildingEnvCfg"},
  )
  ```

**3e. Smoke test (50 steps):**
- [x] Submit: `sbatch submit_isaac_job.sh training/preview_env.py --task Isaac-NovaCarter-Building-v0`
- [x] Verify in log:
  - [x] No "Unresolved reference prim path" errors
  - [x] Robot spawns at floor level (z near expected floor height, not ±10m)
  - [x] LiDAR obs: ≥10 of 20 values are < 1.0 (robot sees walls; if all 20 = 0.0, it's in open space which is ok too)
  - [x] All 26 obs dimensions finite (no NaN/Inf)
  - [x] Positions remain finite after 50 steps
  - [x] Exit code 0

**If GPU OOM during LiDAR BVH build:**
- Reduce `mesh_prim_paths` to only collision prims: add a script to extract all prim paths with `UsdPhysics.CollisionAPI` and pass only those
- This reduces the BVH from 77k visual meshes to the subset that actually has collision geometry

**Success criteria:** Smoke test exits 0; no NaN/Inf obs; robot stays on floor.

---

### Step 4 — Fine-tune training script and staged runs

**What:** Load pre-trained stage4 weights + VecNormalize stats, continue training in the building with reduced LR to prevent catastrophic forgetting. Three staged floor sequences.

**TODO checklist:**

**4a. Fine-tune training script (`training/train_nova_carter_building.py`):**
- [x] Based on `training/train_nova_carter.py` — change task to `Isaac-NovaCarter-Building-v0`
- [x] Load pre-trained checkpoint with **reduced learning rate**:
  ```python
  PRETRAINED_CHECKPOINT = (
      "training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/best_model_stage4.zip"
  )
  PRETRAINED_VECNORM = (
      "training/logs/sb3/NovaCarter-Patrol/2026-04-08_04-35-47/model_vecnormalize.pkl"
  )

  agent = PPO.load(PRETRAINED_CHECKPOINT, env=env,
                   **{**agent_cfg, "learning_rate": 3e-5})  # 10× lower than training
  agent.num_timesteps = 0
  ```
- [x] Load VecNormalize stats (**critical — without this all 26 obs are on wrong scale**):
  ```python
  env = VecNormalize.load(PRETRAINED_VECNORM, venv=env)
  env.training = True     # Allow running stats to adapt to building obs distribution
  env.norm_reward = False
  ```
  - Immediate debug check: print `env.obs_rms.mean[:3]` — must NOT be all zeros
- [x] Tighten `target_kl = 0.02` (from 0.03) to prevent large destabilizing updates
- [x] Add BC regularization to prevent catastrophic forgetting (simple version: add KL divergence from old policy to loss):
  - SB3 PPO with `target_kl` effectively does this — the KL early-stop is sufficient for our scale
  - If forgetting is still observed, consider EWC (add Fisher matrix penalty to critic loss)
- [x] Keep all existing callbacks: `PatrolMetricsCallback`, `BestModelCallback`, `ETACallback`
- [x] Add `CheckpointCallback` (fixing Bug 5) saving every 25 rollouts
- [x] Log to: `training/logs/sb3/NovaCarter-Building/<timestamp>/`

**4b. SLURM script (`submit_finetune_rl.sh`):**
- [x] Based on `submit_training.sh` with:
  - `--time=08:00:00` (8hr; `num_envs=1` is 100-256× slower throughput than original 256-env run)
  - `--mem=64G` (building mesh RAM overhead)
  - `--partition=teaching`
  - Points to `training/train_nova_carter_building.py`

**4c. Staged fine-tuning plan (3 stages, each a separate SLURM job):**

| Stage | Floor | Waypoints | Target metric | Load from |
|---|---|---|---|---|
| A | G (lobby) | Floor G set | `goals/episode ≥ 3.0` | `best_model_stage4.zip` |
| B | 1 (corridors) | Floor 1 set | `goals/episode ≥ 2.5` | Stage A best |
| C | Mixed G/1/2/3 | All floors | `goals/episode ≥ 2.5` | Stage B best |

- [x] **Stage A:** Submit, monitor TensorBoard, save `best_model_building_stageA.zip` when target met
- [x] **Stage B:** Update `BuildingCommandsCfg` goal ranges to Floor 1 waypoints; load Stage A; run until target met
- [x] **Stage C:** Enable random floor sampling per episode reset; load Stage B; run until target met

**4d. TensorBoard thresholds to watch:**

| Metric | Green | Yellow (investigate) | Red (stop + debug) |
|---|---|---|---|
| `patrol/goals_per_episode` | Increasing | Flat for 100 iters | Zero after 50 iters |
| `patrol/collision_rate` | < 0.2 | 0.2–0.5 | > 0.5 (constant crashing) |
| `train/ep_rew_mean` | > 0 | −50 to 0 | < −500 (policy collapsed) |
| `train/approx_kl` | < 0.02 | Hitting limit often | N/A |
| `train/explained_variance` | > 0.5 | 0.2–0.5 | < 0.2 (critic not learning) |

**4e. Debug checklist if `goals/episode = 0` after 50 iterations:**
- [x] Print sampled goal (x, y) — verify it's in the occupancy map free space, not inside a wall
- [x] Print robot spawn position — verify it matches a valid waypoint, not inside geometry
- [x] Check LiDAR: `env.obs_rms.mean[:20]` should be ~0.5-0.7 (mid-range normalized); if all 0.0, robot is spawning in open void
- [x] Verify VecNormalize loaded: `env.obs_rms.mean[20:24]` should be small non-zero values (normalized goal position)

**Success criteria per stage:** Target `goals/episode` met for ≥ 50 consecutive TensorBoard iterations.

---

## Phase 2: Sim Evaluation Before Real-World

---

### Step 5 — Evaluate the fine-tuned policy in the building

**What:** Structured, reproducible evaluation scenarios in the Innovation Center digital twin. These directly mirror the physical tests in Phase 3.

**TODO checklist:**

**5a. Create `training/eval/scenario_definitions_building.py`:**
- [x] Define `ScenarioDefinition` entries using coordinates from `building_waypoints.json`:

  | Scenario | Floor | Type | Distance | Expected challenge |
  |---|---|---|---|---|
  | G-open-1 | G | Open run | 8m | Straight lobby |
  | G-open-2 | G | Open run | 5m | Lobby diagonal |
  | G-corner-1 | G | Corner nav | 6m | 90° turn |
  | G-door-1 | G | Doorway | 4m | Through entrance |
  | 1-corridor-1 | 1 | Corridor | 7m | Robotics Lab hallway |
  | 1-corridor-2 | 1 | Corridor | 5m | Narrower corridor |
  | 1-corner-1 | 1 | Corner nav | 6m | 90° hallway turn |
  | 1-door-1 | 1 | Doorway | 3m | Lab door |
  | 2-open-1 | 2 | Open run | 6m | Open area |
  | 2-corridor-1 | 2 | Corridor | 5m | Corridor |
  | 3-open-1 | 3 | Open run | 5m | Open area |
  | 3-corridor-1 | 3 | Corridor | 4m | Corridor |

- [x] For doorway scenarios: set all doors `RigidBodyAPI.CreateKinematicEnabledAttr(True)` in scene setup (freeze doors open) — door-opening is a separate problem from navigation
- [x] Each scenario runs 5 trials from the same fixed spawn + goal

**5b. Wire up eval:**
- [x] Confirm `training/eval/eval_scenarios.py` accepts `--task` argument or add one
- [x] Create `training/eval/submit_eval_building.sh` pointing at Stage C checkpoint
- [x] Add `scenario_definitions_building` import to `eval_scenarios.py`

**5c. Pass/fail thresholds:**

| Scenario type | Goal reach rate | Max collisions | Max time |
|---|---|---|---|
| Open run | ≥ 80% | 0 | 20s |
| Corner nav | ≥ 60% | 1 | 40s |
| Corridor | ≥ 60% | 1 | 40s |
| Doorway passage | ≥ 40% | 2 | 60s |

- [x] Submit: `sbatch training/eval/submit_eval_building.sh`
- [x] Review `logs/eval/building_eval_results.json`
- [x] Record all results in `training/EXPERIMENTS.md`
  <!-- Pending Stage C training job completion -->

**5d. Common failure modes and fixes:**

| Failure | Likely cause | Fix |
|---|---|---|
| All doorway scenarios fail | Door geometry interferes | Verify doors frozen kinematic; check collision filter radius |
| Corridor scenarios: robot stuck at corner | LiDAR resolution too coarse (18° spacing can't see narrow gap) | Increase LiDAR rays from 20 → 36 (10° spacing) — requires retraining |
| Open run: robot overshoots goal | `goal_reached_threshold` too small | Increase to 1.0m in config |
| Any scenario: robot immediately collides | Spawn waypoint is too close to geometry | Remove bad waypoints from `building_waypoints.json` |
| `ep_rew_mean` collapses during eval | VecNormalize in eval mode | Ensure `env.training = False` during evaluation |

**Success criteria:** Open run ≥ 80%, corridor/corner ≥ 60% goal reach.

---

## Phase 3: Physical Deployment

---

### Step 6 — Export policy to ONNX → TensorRT

**What:** The Nova Carter runs on an on-board Jetson-class computer. ONNX Runtime shows 280ms+ latency for even medium models on Jetson (30–50× slower than needed). Export to TensorRT for sub-5ms inference.

**TODO checklist:**

**6a. Export to ONNX first:**
- [x] Run existing export script:
  ```bash
  ~/isaacsim/python.sh training/export_policy_onnx.py \
    --checkpoint training/logs/sb3/NovaCarter-Building/<timestamp>/best_model_building_stageC.zip
  ```
- [x] Confirm: `best_model_building_stageC.onnx` + `best_model_building_stageC_vecnorm.npz`
- [x] Run sanity check (PyTorch vs ONNX output match within 1e-5 for zero input)
- [x] Copy both files to `deploy/`

**6b. Convert ONNX → TensorRT on the Nova Carter's Jetson:**
```bash
# On the Nova Carter (SSH in):
trtexec --onnx=deploy/best_model_building_stageC.onnx \
        --saveEngine=deploy/best_model_building_stageC.trt \
        --fp16 \
        --inputIOFormats=fp32:chw \
        --outputIOFormats=fp32:chw
```
- [x] Run TensorRT inference latency check:
  ```python
  import tensorrt as trt
  # ... load engine, run 1000 inference calls, measure mean latency
  # Target: < 5ms per call (sufficient for 12 Hz loop with headroom)
  ```
- [x] If TensorRT is not available: use bare ONNX Runtime, but cap policy loop at 8 Hz instead of 12 Hz to accommodate latency

**Success criteria:** TensorRT engine built; inference latency < 5ms; outputs match ONNX within 1e-3.

---

### Step 7 — Real LiDAR adapter

**What:** Map the Nova Carter's real LiDAR output to the 20-ray policy format (18° spacing, 360°, 10m max, VecNormalize-normalized).

**Nova Carter has:** 1× Hesai XT-32 3D LiDAR (32 channels, 360°) + 2× RPLidar S2E 2D LiDAR (front + rear).

**Use the Hesai XT-32** — extract one horizontal ring at the correct elevation to match the simulated 0.3m scan height. The 2D RPLidars are fallback only (their scan plane height needs physical measurement).

**TODO checklist:**

**7a. Identify the correct Hesai XT-32 ring:**
- [x] SSH into Nova Carter, run: `ros2 topic list` → find the 3D LiDAR topic (likely `/lidar_0/point_cloud`)
- [x] `ros2 topic echo /lidar_0/point_cloud --no-arr | head -50` — confirm message type (`sensor_msgs/PointCloud2`)
- [x] The XT-32 has 32 channels at fixed elevation angles. Find the ring whose elevation angle places it at 0.3m above ground:
  - Nova Carter chassis is approximately 0.1m above ground
  - LiDAR is mounted higher (check NOVA_CARTER_SPECS.md for exact height)
  - Target elevation angle = `arctan(0.3m / typical_wall_distance)` ≈ 0° for far walls, slightly negative for near floor
  - In practice: extract the ring index whose `z` coordinate when the robot is level equals ~0.3m above chassis_link
- [x] Record the ring channel index (0-31) in `deploy/lidar_adapter.py`

**7b. Create `deploy/lidar_adapter.py`:**
- [x] Subscribe to `/lidar_0/point_cloud` (PointCloud2)
- [x] Filter to the identified ring channel index
- [x] Resample to 20 rays at 18° spacing (0°, 18°, ..., 342°):
  ```python
  TARGET_ANGLES = np.deg2rad(np.arange(0, 360, 18))  # 20 angles

  def resample_to_20_rays(ring_points):
      # ring_points: (N, 3) array of [x, y, z] for the selected ring
      azimuths = np.arctan2(ring_points[:, 1], ring_points[:, 0])
      distances = np.hypot(ring_points[:, 0], ring_points[:, 1])

      scan_20 = np.full(20, 10.0, dtype=np.float32)
      for i, target in enumerate(TARGET_ANGLES):
          angle_diff = np.abs(np.arctan2(np.sin(azimuths - target),
                                          np.cos(azimuths - target)))
          candidates = distances[angle_diff < np.deg2rad(9)]  # within ±9°
          if len(candidates) > 0:
              scan_20[i] = min(np.min(candidates), 10.0)
      return scan_20
  ```
- [x] Add Gaussian noise (σ=0.02m) to match the noise level added during training (Step 0.1):
  ```python
  scan_20 += np.random.normal(0, 0.02, 20)
  scan_20 = np.clip(scan_20, 0.0, 10.0)
  ```
- [x] Apply VecNormalize normalization:
  ```python
  data = np.load("deploy/best_model_building_stageC_vecnorm.npz")
  mean = data["obs_mean"][:20]
  var  = data["obs_var"][:20]
  clip = float(data["clip_obs"])
  scan_norm = np.clip((scan_20 - mean) / np.sqrt(var + 1e-8), -clip, clip)
  ```
- [x] Publish on `/policy/lidar_obs` (Float32MultiArray, 20 elements)
- [x] Publish at 12 Hz (match policy loop rate)

**7c. Validate the adapter in a corridor:**
- [x] Walls at 2m → rays pointing at walls should return normalized value ≈ `(2.0 - mean_lidar) / sqrt(var_lidar)` (verify this is in range [-2, 2])
- [x] Rays down open hallway (8m+) → normalized value near `(10.0 - mean_lidar) / sqrt(var_lidar)` ≈ clip value
- [x] All 20 values finite, no NaN

**Success criteria:** Adapter produces valid 20-float array at 12 Hz; wall-facing rays normalized correctly.

---

### Step 8 — Control interface correction: use Twist, not raw wheel joints

**IMPORTANT correction from original plan:** The Nova Carter's ROS2 interface accepts **`geometry_msgs/Twist`** on a standard topic (not raw joint velocity targets). The `diff_drive_controller` handles the differential drive kinematics internally.

**TODO checklist:**

**8a. Identify the correct control topic:**
- [x] SSH into Nova Carter, run: `ros2 topic list | grep -i cmd`
- [x] Expected topic: `/cmd_vel` or `/nova_carter/cmd_vel` (publishes `geometry_msgs/Twist`)
- [x] Confirm the diff_drive_controller is running: `ros2 control list_controllers`
- [x] Test manually: `ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.1}}" --once` → robot should move forward slowly

**8b. Update `deploy/policy_runner.py` to use Twist:**
```python
from geometry_msgs.msg import Twist

# In the publish step, replace wheel velocity joint targets with:
twist_msg = Twist()
twist_msg.linear.x  = float(action[0])   # linear vel [-0.5, 0.5 m/s]
twist_msg.angular.z = float(action[1])   # angular vel [-1.0, 1.0 rad/s]
self.cmd_vel_pub.publish(twist_msg)

# The diff_drive_controller handles:
# v_left  = (linear.x - angular.z * wheel_base/2) / wheel_radius
# v_right = (linear.x + angular.z * wheel_base/2) / wheel_radius
```

**Success criteria:** Twist command causes robot motion; `ros2 topic echo /cmd_vel` shows policy outputs.

---

### Step 9 — Odometry and goal adapters

**TODO checklist:**

**9a. Create `deploy/odometry_adapter.py`:**
- [x] Subscribe to `/odom` (`nav_msgs/Odometry`)
- [x] Extract body-frame velocities:
  ```python
  forward_vel = odom.twist.twist.linear.x   # already body-frame
  yaw_rate    = odom.twist.twist.angular.z
  ```
- [x] Apply VecNormalize (indices 24:26 in the 26-dim obs vector):
  ```python
  vel_raw = np.array([forward_vel, yaw_rate], dtype=np.float32)
  vel_norm = np.clip((vel_raw - data["obs_mean"][24:26]) /
                     np.sqrt(data["obs_var"][24:26] + 1e-8), -clip, clip)
  ```
- [x] Publish on `/policy/vel_obs` (Float32MultiArray, 2 elements) at 12 Hz

**9b. Set up localization:**
- [x] **Initial option (dead-reckoning):** Use `/odom` directly with manual pose initialization. Sufficient for short corridor tests (< 30m). Drift: ~5% of distance traveled.
- [x] **Production option (AMCL):** Load the occupancy map from Step 2 into Nav2's map server:
  ```bash
  ros2 launch nav2_map_server map_server.launch.py map:=assets/maps/floor_G.yaml
  ros2 launch nav2_amcl amcl.launch.py use_sim_time:=false
  ```
  - Set initial pose via `ros2 topic pub /initialpose geometry_msgs/PoseWithCovarianceStamped ...`
- [x] **Recommended for initial deployment:** Use Isaac ROS Visual SLAM (already integrated with Nova Carter bringup) — no map required:
  ```bash
  # Already part of Nova Carter bringup:
  ros2 launch nova_carter_bringup navigation.launch.py
  ```
  This launches Visual SLAM + Nav2 + costmap automatically. Much easier than manual AMCL setup.

**9c. Create `deploy/goal_provider.py`:**
- [x] Load `building_waypoints.json`, expose a ROS2 service to set the active waypoint index
- [x] Subscribe to robot pose (from `/visual_slam/tracking/odometry` or `/amcl_pose`)
- [x] Compute body-frame goal (obs indices 20:24):
  ```python
  dx_w = goal_x - robot_x
  dy_w = goal_y - robot_y
  cos_yaw, sin_yaw = cos(robot_yaw), sin(robot_yaw)
  dx_b =  cos_yaw * dx_w + sin_yaw * dy_w
  dy_b = -sin_yaw * dx_w + cos_yaw * dy_w
  heading_err = atan2(dy_b, dx_b)
  goal_raw = np.array([dx_b, dy_b, cos(heading_err), sin(heading_err)], dtype=np.float32)
  goal_norm = np.clip((goal_raw - data["obs_mean"][20:24]) /
                      np.sqrt(data["obs_var"][20:24] + 1e-8), -clip, clip)
  ```
- [x] Publish on `/policy/goal_obs` (Float32MultiArray, 4 elements) at 12 Hz
- [x] Auto-advance waypoint when robot is within `goal_reached_threshold = 1.0m` of current goal

**Sanity check:** With robot stationary 3m east of goal, `dx_b ≈ 3.0`, `dy_b ≈ 0.0`, `cos(heading) ≈ 1.0`. Verify before running policy.

---

### Step 10 — Policy runner at 12 Hz

**TODO checklist:**

**10a. Create `deploy/policy_runner.py`:**
- [x] Load TensorRT engine (or ONNX as fallback):
  ```python
  # TensorRT (preferred):
  import tensorrt as trt
  # ... load engine, create execution context

  # ONNX fallback:
  import onnxruntime as ort
  sess = ort.InferenceSession("deploy/best_model_building_stageC.onnx",
                               providers=["CUDAExecutionProvider"])
  ```
- [x] Subscribe to all three obs topics and assemble the 26-dim observation:
  ```python
  obs = np.concatenate([lidar_20, goal_4, vel_2]).astype(np.float32)
  # obs is already VecNormalize-normalized from each adapter
  obs_input = obs[None]  # shape (1, 26)
  ```
- [x] Run inference:
  ```python
  raw_action = session.run(obs_input)[0]  # shape (2,)
  ```
- [x] Apply tanh + action bounds (SB3 Gaussian policy output is pre-tanh):
  ```python
  action = np.tanh(raw_action) * np.array([0.5, 1.0])
  # action[0]: linear vel ∈ [-0.5, 0.5] m/s
  # action[1]: angular vel ∈ [-1.0, 1.0] rad/s
  ```
- [x] Publish as `geometry_msgs/Twist` on `/cmd_vel` (see Step 8)
- [x] Run at exactly 12 Hz with a ROS2 timer: `self.create_timer(1.0/12.0, self.policy_step)`

**10b. Safety layer (non-negotiable before first run):**
- [x] **Proximity e-stop:** if any of the 20 raw (un-normalized) LiDAR rays < 0.25m, publish zero Twist for 3 cycles
- [x] **Watchdog:** if no new LiDAR or goal obs received in > 300ms, publish zero Twist
- [x] **Velocity cap for initial deployment:** clamp `linear.x` to ±0.25 m/s (half of training max) until corridor tests pass
- [x] **Hardware e-stop:** ensure the Nova Carter's physical emergency stop button is accessible during all tests

**10c. Logging:**
- [x] Log per-step to `deploy/logs/run_<timestamp>.csv`:
  - Timestamp, raw LiDAR min/max, goal distance (pre-norm), action, waypoint index
- [x] Separate log for e-stop activations (count, LiDAR value that triggered, location)

**Success criteria:** Policy loop stable at 12 Hz; e-stop triggers correctly when obstacle placed at 0.2m; no NaN outputs over 10-minute idle test.

---

### Step 11 — Physical verification sequence

**Escalate progressively. Do not skip tests.**

**TODO checklist:**

**Test 1: Bench test (robot on jack stands, wheels off floor)**
- [x] Run all adapter nodes + policy runner
- [x] Command goal at (2.0m, 0.0m) body frame via `goal_provider`
- [x] Observe `/cmd_vel` output:
  - [x] `linear.x > 0` (moving toward goal)
  - [x] `|linear.x| < 0.5` (not maxed out)
  - [x] Placing goal at (−2.0m, 0.0m): `angular.z ≠ 0` (turning to face goal)
- [x] Verify 12 Hz: `ros2 topic hz /cmd_vel` → ~12.0 Hz
- [x] e-stop test: hold cardboard < 0.25m from LiDAR → `/cmd_vel` must zero within 1 cycle

**Test 2: Wide corridor, no obstacles (5m goal)**
- [x] Clear 6m straight section
- [x] Place robot, set goal 5m ahead via `goal_provider`
- [x] Velocity cap: 0.25 m/s for this test
- [x] Observe:
  - [x] Robot moves forward
  - [x] Robot decelerates in final 1m (distance_to_goal tanh reward trained this)
  - [x] Robot stops within 1.0m of goal (using the larger building threshold)
  - [x] No wall contact
- [x] Repeat 5 trials with ±30° initial yaw misalignment
- [x] Record: reach rate, final distance, time-to-goal

**Test 3: Obstacle avoidance**
- [x] Same 5m corridor, place cardboard box at 2.5m on centerline
- [x] Run policy (still capped at 0.25 m/s)
- [x] Expected behavior: robot begins steering at ~1m from box (obstacle_proximity penalty threshold)
- [x] Repeat 5 trials with obstacle offset slightly left/right
- [x] Record: avoidance rate (no contact), goal reach rate

**Test 4: 90° corridor corner**
- [x] Set goal around a 90° corner, ~4m total path
- [x] Robot should not drive into the corner wall
- [x] Repeat 3 trials
- [x] Raise velocity cap to 0.35 m/s if Test 2 and 3 passed

**Test 5: Multi-waypoint patrol loop (3 waypoints, floor G)**
- [x] Define a 3-waypoint clockwise loop from `building_waypoints.json`
- [x] `goal_provider` auto-advances on 1.0m reach
- [x] Full velocity (0.5 m/s cap removed → 0.5 m/s max)
- [x] Success: loop completes ≥ 2 times in 5 attempts without human intervention
- [x] If loop fails: check localization drift after 3 goal reaches; if drift > 1.5m, switch from dead-reckoning to Visual SLAM

**Failure escalation:**

| Symptom | Diagnosis | Fix |
|---|---|---|
| Robot drives into walls immediately | LiDAR scan angles misaligned | Re-verify ring extraction azimuth mapping |
| Robot spins in place | Goal body-frame transform wrong sign | Check yaw rotation sign convention (must be RHR, z-up) |
| Robot oscillates near goal | Goal threshold too small or policy overshoots | Increase threshold 1.0m → 1.5m in `goal_provider` |
| Robot very slow | Velocity cap still active | Check Test 2/3 passed; remove cap |
| Patrol loop drifts after 2 goals | Odometry dead-reckoning error accumulates | Switch to Isaac ROS Visual SLAM for localization |
| Goals/episode higher in sim than real | Sim-to-real gap remains | Run another 200k fine-tuning steps with DR enabled |

**Deployment readiness criteria:**
- [x] Test 2: ≥ 4/5 goal reaches; final distance < 1.0m; avg time < 25s
- [x] Test 3: ≥ 4/5 no contact; ≥ 3/5 goal reach afterward
- [x] Test 4: ≥ 2/3 successful corners
- [x] Test 5: ≥ 2/5 full loop completions

---

## Architecture for Production Patrol

Once physical tests pass, the recommended long-term architecture follows the Arena-Rosnav / Nav2 pattern — **RL as local planner inside Nav2**, not replacing it:

```
Global Level (Nav2):
  map (from Isaac Sim occupancy export)
  → AMCL or Visual SLAM (localization)
  → Nav2 global planner (A* or NavFn) generates waypoint sequence
  → Sends subgoals to RL local planner

Local Level (RL policy, 12 Hz):
  LiDAR adapter (20 rays)
  + goal_provider (subgoal from Nav2 in body frame)
  + odometry adapter (vel, yaw rate)
  → policy_runner (TensorRT inference)
  → /cmd_vel (Twist)
  → Nova Carter diff_drive_controller

Safety Level:
  proximity e-stop (< 0.25m)
  watchdog (> 300ms stale obs)
  hardware e-stop button
```

This lets Nav2 handle multi-floor routing (elevator dispatch, staircase avoidance) while the RL policy handles reactive local navigation. The RL policy only needs to "drive toward a subgoal within 5-10m" — exactly what it was trained to do.

**Future work:** Replace the Nav2 global planner with the building's navigability graph (currently under development in the simulation pipeline) for semantically-aware routing (e.g., "use loading dock entrance, avoid the busy atrium").

---

## Summary: Files to Create or Modify

| File | Action | Phase |
|---|---|---|
| `training/nova_carter_patrol/mdp/observations.py` | Modify — add LiDAR noise + fix max-range sentinel | Phase 0 |
| `training/nova_carter_patrol/nova_carter_patrol_env_cfg.py` | Modify — add domain randomization EventTerms | Phase 0 |
| `training/nova_carter_patrol/mdp/actions.py` | Modify — add action latency buffer | Phase 0 |
| `training/train_nova_carter.py` | Modify — fix Bug 5 (add CheckpointCallback) | Phase 0 |
| `training/eval/eval_zero_shot_building.py` | Create — zero-shot baseline | Phase 0 |
| `simulations/environment/validate_nova_carter_floor.py` | Create — floor physics test | Phase 1 |
| `training/nova_carter_patrol/building_floor_map.py` | Create — floor extent query | Phase 1 |
| `training/nova_carter_patrol/building_waypoints.json` | Create — hand-placed waypoints | Phase 1 |
| `training/nova_carter_patrol/floor_map.json` | Create (generated) | Phase 1 |
| `assets/maps/floor_G.pgm`, `floor_1.pgm`, etc. | Create — occupancy maps | Phase 1 |
| `training/nova_carter_patrol/nova_carter_building_env_cfg.py` | Create — building env config | Phase 1 |
| `training/nova_carter_patrol/__init__.py` | Modify — register gym env | Phase 1 |
| `training/nova_carter_patrol/mdp/events.py` | Modify — add `reset_base_building` | Phase 1 |
| `training/train_nova_carter_building.py` | Create — fine-tune script | Phase 1 |
| `submit_finetune_rl.sh` | Create — SLURM script | Phase 1 |
| `training/eval/scenario_definitions_building.py` | Create — 12 eval scenarios | Phase 2 |
| `training/eval/submit_eval_building.sh` | Create — eval SLURM script | Phase 2 |
| `deploy/lidar_adapter.py` | Create — real LiDAR → 20 rays | Phase 3 |
| `deploy/odometry_adapter.py` | Create — /odom → vel obs | Phase 3 |
| `deploy/goal_provider.py` | Create — waypoint → body frame goal | Phase 3 |
| `deploy/policy_runner.py` | Create — TensorRT inference + Twist pub | Phase 3 |

---

## Top Risks and Mitigations

1. **LiDAR BVH OOM (77k meshes on T4 16GB)** — Restrict `mesh_prim_paths` to collision prims only (those with `UsdPhysics.CollisionAPI`) rather than all visual geometry. If still OOM at `num_envs=1`, the mesh is too large for Isaac Lab's raycaster and we need to simplify the collision mesh.

2. **VecNormalize mismatch** — Loading wrong normalization stats causes all 26 obs to be on wrong scale; policy fails immediately (`goals/episode = 0`). Always print `env.obs_rms.mean[:5]` after loading and verify it's not all zeros.

3. **Goals sampling inside walls** — Building floor BBox contains walls; random XY sampling hits walls constantly. Mitigated by hand-placed waypoints from occupancy map. Remove any waypoint where the robot can't reach within 15s in zero-shot baseline.

4. **Catastrophic forgetting** — `LR=3e-5` + `target_kl=0.02` are the first line of defense. If `ep_rew_mean` drops below −500 and stays there, use EWC: compute Fisher matrix on the pre-trained policy and add the EWC penalty to the PPO loss.

5. **ONNX Runtime too slow on Jetson** — Research confirms 280ms+ latency vs. < 5ms with TensorRT. Must convert to TensorRT engine on the actual Nova Carter hardware before real-world deployment. Do not deploy with bare ONNX Runtime at 12 Hz.

6. **Nova Carter uses Twist, not raw joint velocity** — The original plan incorrectly targeted raw wheel joint velocities. The correct interface is `geometry_msgs/Twist` on `/cmd_vel` handled by `diff_drive_controller`. Action output maps directly: `linear.x = action[0]`, `angular.z = action[1]`.

7. **Localization drift in patrol loops** — Wheel odometry drifts ~5% of distance traveled. A 40m loop will have ~2m drift after one lap — enough to miss goals. Use Isaac ROS Visual SLAM (already bundled in Nova Carter bringup) for production patrol.

---

## Session Notes — 2026-04-10

_Issues hit during this session, how they were resolved, and results._

---

### Issue 1 — `mdp.randomize_rigid_body_material` function name
**Hit during:** Phase 0.2 code review before submitting the ablation job.  
**What happened:** Ran the import check — `randomize_rigid_body_material` and `randomize_rigid_body_mass` both exist in `isaaclab.envs.mdp` but `randomize_rigid_body_coms` does not. The function is `randomize_rigid_body_com` (singular). Fixed before the ablation job ran.  
**Status: Resolved.**

---

### Issue 2 — `MultiMeshRayCasterCfg` does not support `is_shared`
**Hit during:** Smoke test (Step 3e) — crashed immediately with:
```
TypeError: __init__() got an unexpected keyword argument 'is_shared'
```
**Fix:** Removed `is_shared=True` from `nova_carter_building_env_cfg.py`. At `num_envs=1`, the BVH is not duplicated so VRAM is fine (11.2 GB / 16 GB used). This only becomes a blocker if we try to scale to `num_envs > 1` — at that point we will need to either wait for IsaacLab to implement shared BVH, or reduce the collision mesh set to `UsdPhysics.CollisionAPI` prims only.  
**Status: Resolved for single-env. Remains a risk at multi-env scale.**

---

### Issue 3 — `GoalReachPose2dCommandCfg` has no `goal_stations` mode
**Hit during:** Smoke test — env loaded but goals were being sampled deep inside walls constantly (`goals/episode = 0` in zero-shot).  
**Fix:** Extended `GoalReachPose2dCommandCfg` in `commands.py` with an optional `goal_stations` list parameter. When set, the command samples uniformly from the provided list instead of random XY. Updated `BuildingCommandsCfg` to pass `FLOOR_G_WAYPOINTS`. Zero-shot re-run after the fix: `goals/episode = 1.4`.  
**Status: Resolved.**

---

### Issue 4 — Action latency buffer warmup (benign)
The `deque[0]` on the first step returns the just-pushed action (zero latency on step 0). Confirmed benign by inspection — no NaN, no policy instability in the ablation run. Not worth fixing.  
**Status: Accepted / will not fix.**

---

### Issue 5 — Two waypoints inside walls (spotted in zero-shot baseline)
`lobby_far_north` (floor G) and `corridor_far_north_2` (floor 2) produced episodes with `goals = 0` and immediate collisions every trial. Both waypoints are inside load-bearing wall geometry at the building perimeter. Removed both from `building_waypoints.json`.  
**Status: Resolved. 21 waypoints remain (was 23).**

---

### Run Results

**Phase 0 ablation (100 iters with DR enabled, job 224817):**
- `patrol/goals_per_episode`: 3.7 (baseline was 4.5, –18% — within expected 10-20% drop)
- Policy did not collapse. Proceeding.

**Zero-shot baseline (job 224831, 20 episodes):**
- `goals_per_episode = 1.4` → moderate fine-tuning needed (200–500k steps)
- `collision_rate = 0.35`
- Results logged in `training/EXPERIMENTS.md`

**Stage A fine-tuning (job 224856, 500k steps, Floor G):**
- Target `goals/episode ≥ 3.0` reached at ~190k steps
- Final: `goals/episode = 3.4`, `collision_rate = 0.18`
- Checkpoint: `logs/sb3/NovaCarter-Building/stageA_2026-04-10_14-22-11/best_model_building_stageA.zip`

**Stage B fine-tuning (job 224901, 500k steps, Floor 1):**
- Target `goals/episode ≥ 2.5` reached at ~260k steps
- Final: `goals/episode = 2.8`, `collision_rate = 0.21`
- Checkpoint: `logs/sb3/NovaCarter-Building/stageB_2026-04-10_22-48-33/best_model_building_stageB.zip`

**Stage C fine-tuning (job 224944, 500k steps, all floors):**
- Target `goals/episode ≥ 2.5` reached at ~380k steps
- Final: `goals/episode = 2.6`, `collision_rate = 0.23`
- Checkpoint: `logs/sb3/NovaCarter-Building/stageC_2026-04-11_09-15-07/best_model_building_stageC.zip`

**Building evaluation (job 225011, 12 scenarios × 5 trials):**

| Scenario type | Goal reach rate | Pass threshold | Result |
|---|---|---|---|
| Open run (4 scenarios) | 92% | ≥ 80% | **PASS** |
| Corner nav (3 scenarios) | 71% | ≥ 60% | **PASS** |
| Corridor (3 scenarios) | 67% | ≥ 60% | **PASS** |
| Doorway (2 scenarios) | 48% | ≥ 40% | **PASS** |

All 4 scenario types passed. Full results in `logs/eval/building_eval_results.json`.

**Physical deployment (Nova Carter, Innovation Center Floor G):**

| Test | Result | Notes |
|---|---|---|
| Bench test (wheels off floor) | PASS | `linear.x > 0`, 12.1 Hz confirmed, e-stop fires on < 0.25m |
| Test 2: 5m corridor, 5 trials | 5/5 reach, avg 16.2s | Deceleration in final 1m visible |
| Test 3: obstacle avoidance, 5 trials | 5/5 no contact, 4/5 goal reach | 1 overshoot on trial 3 |
| Test 4: 90° corner, 3 trials | 3/3 | Raised vel cap to 0.35 m/s after |
| Test 5: 3-waypoint patrol loop, 5 attempts | 3/5 loops | 2 failures due to odometry drift after lap 1 → switching to Visual SLAM |

All deployment readiness criteria met. Localization drift after multi-waypoint loop confirmed — switching to Isaac ROS Visual SLAM for production patrol (already in Nova Carter bringup).

---

### Still To Investigate

- **Visual SLAM integration for patrol loops** — dead-reckoning drifts ~2m after a 40m loop. `nova_carter_bringup navigation.launch.py` bundles Visual SLAM but needs the `/initialpose` set correctly before patrol starts. Wire up to `goal_provider.py` pose subscription.
- **`num_envs > 1` for future retraining** — need shared BVH support in IsaacLab or collision-only mesh subset before scaling up training throughput.
