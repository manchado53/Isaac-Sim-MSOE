# Training Experiments Log

Tracks all Nova Carter Patrol training runs with their configurations, results, and key observations.

---

## Run Index

| Run | Job ID | Iters | Peak Reward | Model | Status |
|-----|--------|-------|-------------|-------|--------|
| [2026-03-14_21-46-50](#2026-03-14_21-46-50) | 234430 | 50 | +10.8 | model.zip | Completed |
| [2026-03-14_23-10-04](#2026-03-14_23-10-04) | 234437 | 50 | +10.8 | model.zip | Completed |
| [2026-03-15_12-11-18](#2026-03-15_12-11-18) | 234706 | 768 | +18.9 | checkpoints only | Timed out |
| [2026-03-15_12-31-59](#2026-03-15_12-31-59) | 234721 | 760 | +18.8 | checkpoints only | Timed out |
| [2026-03-15_13-09-00](#2026-03-15_13-09-00) | 234736 | 763 | +18.6 | checkpoints only | Timed out |
| [2026-03-16_03-25-15](#2026-03-16_03-25-15) | 235054 | 1000 | +14.6 | model.zip | Completed |
| [2026-03-17_04-24-01](#2026-03-17_04-24-01) | 235402 | 1000 | +5.9 | model.zip | Completed |
| [2026-03-17_04-35-51](#2026-03-17_04-35-51) | 235405 | 1000 | +174 | model.zip | Completed |
| [2026-03-22_18-25-55](#2026-03-22_18-25-55) | 237451 | 1000 | +58.2 | model.zip | Completed |
| [2026-03-22_22-48-45](#2026-03-22_22-48-45) | 237476 | 581 | +102 | checkpoints only | Timed out |
| [2026-03-23_01-14-32](#2026-03-23_01-14-32) | 237487 | 466 | +99.9 | checkpoints only | Timed out |
| [2026-03-23_09-42-21](#2026-03-23_09-42-21) | 237492 | — | — | — | Running |
| [2026-03-23_12-07-03](#2026-03-23_12-07-03-job-237512) | 237512 | — | — | — | Running |
| [2026-03-23_21-20-54](#2026-03-23_21-20-54-job-237611) | 237611 | — | — | — | Running |
| [2026-03-23_22-58-20](#2026-03-23_22-58-20-job-237629) | 237629 | — | — | — | Running |
| [2026-03-24_00-41-07](#2026-03-24_00-41-07-job-237659) | 237659 | — | — | — | Running |
| TBD | 238023 | — | — | — | Cancelled (never started - disk full at submission) |
| [2026-03-24_23-22-03](#2026-03-24_23-22-03-job-238036) | 238036 | — | — | — | Failed (tensor bool bug) |
| [2026-03-24_23-45-19](#2026-03-24_23-45-19-job-238049) | 238049 | — | — | — | Failed (CUDA tensor → numpy bug) |
| [2026-03-25_00-06-50](#2026-03-25_00-06-50-job-238060) | 238060 | — | — | — | Running (still active) |
| [2026-03-25_01-18-16](#2026-03-25_01-18-16-job-238083) | 238083 | — | — | — | Cancelled |
| [2026-03-25_03-52-41](#2026-03-25_03-52-41-job-238124) | 238124 | — | — | — | Running |
| TBD | 238125 | — | — | — | Cancelled (debug print version, superseded) |
| [2026-03-25_03-26-13](#2026-03-25_03-26-13-job-238126) | 238126 | — | — | — | Cancelled |
| [2026-03-25_04-07-17](#2026-03-25_04-07-17-job-238133) | 238133 | 756 | -5.01 | — | Cancelled (episode_trigger bug — only 1 video recorded) |
| [2026-03-25_17-30-31](#2026-03-25_17-30-31-job-238291) | 238291 | — | — | — | Running |
| [2026-03-25_19-09-33](#2026-03-25_19-09-33-job-238333) | 238333 | — | — | — | Running |
| [TBD (Job 238442)](#tbd-job-238442) | 238442 | — | — | — | Running |
| [EVAL: 2026-03-25_21-04-04 best_model (scenarios)](#eval-238476) | 238476 | — | — | best_model.zip | Running (scenario eval) |
| [TBD (Job 238496)](#tbd-job-238496) | 238496 | — | — | — | Running |
| [2026-03-26_00-58-11](#2026-03-26_00-58-11-job-238479) | 238479 | — | — | — | Running |
| [EVAL: 2026-03-26_00-58-11 best_model (scenarios)](#eval-238720) | 238720 | — | — | best_model.zip | Running (scenario eval) |
| [EVAL: 2026-03-25_21-04-04 best_model (scenarios)](#eval-238721) | 238721 | — | — | best_model.zip | Running (scenario eval) |
| [2026-03-27_02-43-38](#2026-03-27_02-43-38-job-238807) | 238807 | — | — | — | Running |
| [EVAL: 2026-03-23_21-20-54 best_model (scenarios)](#eval-238901) | 238901 | — | — | best_model.zip | Running (scenario eval) |
| [EVAL: 2026-03-26_00-58-11 best_model (scenarios)](#eval-238830) | 238830 | — | — | best_model.zip | Running (scenario eval) |
| [EVAL: 2026-03-26_01-51-02 best_model (scenarios)](#eval-238831) | 238831 | — | — | best_model.zip | Running (scenario eval) |
| [EVAL: 2026-03-27_01-05-35 best_model (scenarios)](#eval-238832) | 238832 | — | — | best_model.zip | Running (scenario eval) |
| [EVAL: 2026-03-27_02-43-38 best_model (scenarios)](#eval-238833) | 238833 | — | — | best_model.zip | Running (scenario eval) |
| [2026-03-29_22-59-11](#tbd-job-239123) | 239123 | 500 | — | model.zip | Completed |
| [2026-04-05_03-21-31](#2026-04-05_03-21-31-job-240712) | 240712 | — | — | — | Running |
| [EVAL: 2026-03-29_22-59-11 model.zip (scenarios)](#eval-240742) | 240742 | — | — | model.zip | Running (scenario eval) |
| [EVAL: 2026-04-05_03-21-31 latest_model_stage0.zip (scenarios)](#eval-240743) | 240743 | — | — | latest_model_stage0.zip | Running (scenario eval) |
| [TBD (Job 240766)](#tbd-job-240766) | 240766 | — | — | — | Running (broken termination) |
| [TBD (Job 240773)](#tbd-job-240773) | 240773 | — | — | — | Running |
| [TBD (Job 240780)](#tbd-job-240780) | 240780 | — | — | — | Running |
| [TBD (Job 240869)](#tbd-job-240869) | 240869 | — | — | — | Running |
| [EVAL: 2026-04-06_02-35-55 latest_model_stage3.zip (scenarios)](#eval-240875) | 240875 | — | — | latest_model_stage3.zip | Running (scenario eval) |
| [TBD (Job 241387)](#tbd-job-241387) | 241387 | — | — | — | Running |
| [TBD (Job 241962)](#tbd-job-241962) | 241962 | — | — | — | Running |

---

## Detailed Run Notes

### TBD-Job-241962
- **Job**: 241962 | **Logs**: `training/logs/train_241962.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/TBD (Job 241962)/`
- **Partition**: teaching (T4, 7-day time limit)
- **What changed**: All curriculum stage durations doubled (stage1: 51200→102400 steps, stage2: 128000→256000 steps, stage3: 230400→460800 steps). Time limit increased from 12h → 7 days.
- **Why**: Previous 12h runs timed out before reaching Stage 3/4 with original stage lengths. Doubling stage lengths requires ~900 iters to complete all transitions; needed 7-day window to allow the robot to experience all 4 curriculum stages.
- **Hypothesis**: Longer exposure per stage gives the robot more time to master each difficulty level before advancing, potentially leading to more robust navigation in later stages.
- **Result**: TBD

### EVAL-240875
- **Job**: 240875 | **Logs**: `training/logs/eval_scenarios_240875.out/.err`
- **Checkpoint**: `training/logs/sb3/NovaCarter-Patrol/2026-04-06_02-35-55/latest_model_stage3.zip`
- **Script**: `training/eval/eval_scenarios.py` (6-scenario suite)
- **Goal**: Evaluate latest_model_stage3 from run 2026-04-06_02-35-55
- **Result**: TBD

### TBD-Job-240869
- **Job**: 240869 | **Logs**: `training/logs/train_240869.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/TBD (Job 240869)/`
- **What changed**: Curriculum Stage 2 redesigned — table fixed at center (±0.3m jitter), robot + goals spawn anywhere in full room (±4.5m). Stage 3 goal range expanded ±3m → ±4.5m. Added `goal_stations` support to `GoalReachPose2dCommand` (unused this run).
- **Why**: Stage 2 was nearly identical to Stage 1 (one table in north strip, full room). New design forces robot to always navigate around a central obstacle regardless of start/goal position.
- **Hypothesis**: Central table obstacle provides consistent navigation challenge throughout Stage 2; wider Stage 3 goals push robot to explore full room with 3 obstacles
- **Result**: TBD

### 2026-04-05_03-21-31-Job-240712
- **Job**: 240712 | **Logs**: `training/logs/train_240712.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/2026-04-05_03-21-31/`
- **What changed**: Collision weight reduced from -100.0 → -30.0
- **Why**: -100.0 equaled goal_reached reward per RL step (-8.33 vs +8.33 after dt scaling), risking "robot stands still" degenerate policy
- **Hypothesis**: -30.0 (-2.50/step) is still a strong deterrent but leaves the robot a clear incentive to keep moving toward goals
- **Result**: TBD

### EVAL-238721
- **Job**: 238721 | **Logs**: `logs/eval_scenarios_238721.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-25_21-04-04/best_model.zip`
- **Script**: `training/eval/eval_scenarios.py` (6-scenario suite, single video)
- **Goal**: Evaluate best_model from run 2026-03-25_21-04-04
- **Result**: TBD

### EVAL-238720
- **Job**: 238720 | **Logs**: `logs/eval_scenarios_238720.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-26_00-58-11/best_model.zip`
- **Script**: `training/eval/eval_scenarios.py` (6-scenario suite, single video)
- **Goal**: Evaluate best_model from run 2026-03-26_00-58-11 (shorter Stage 1, no backward motion penalty)
- **Result**: TBD

### EVAL-238476
- **Job**: 238476 | **Logs**: `logs/eval_scenarios_238476.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-25_21-04-04/best_model.zip`
- **Script**: `training/eval/eval_scenarios.py` (6-scenario suite, single video)
- **Goal**: Evaluate best_model from run 2026-03-25_21-04-04 on fixed scenarios
- **Result**: TBD

### 2026-03-14_21-46-50
- **Job**: 234430 | **Logs**: `train_234430.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-14_21-46-50/model.zip`
- **Config**: Original tanh rewards (position_tracking +0.5, heading -0.2, velocity_near_goal -2.0, collision -2.0 via LiDAR, obstacle_proximity -1.0, goal_reached +2.0)
- **Goal**: First test of tanh reward shaping with goal-reached resampling
- **Result**: Reward -38 → +10.8 over 50 iters. Robot navigated toward goals but stopped at 0.5-0.6m.
- **Key finding**: velocity_near_goal at -2.0 with 1.0m threshold caused premature stopping.

### 2026-03-14_23-10-04
- **Job**: 234437 | **Logs**: `train_234437.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-14_23-10-04/model.zip`
- **Config**: Same as above (first run with GoalReachPose2dCommand + command_resample termination)
- **Goal**: Validate goal-reached command resampling pattern
- **Result**: Reward -37 → +10.8 over 50 iters. Same stopping behavior. ep_len_mean stayed at ~990 (command_resample not triggering).
- **Key finding**: Robot reaches goals but doesn't consistently hit 6 per episode.

### 2026-03-15_12-11-18
- **Job**: 234706 | **Logs**: `train_234706.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-15_12-11-18/model_64256000_steps.zip` (latest checkpoint)
- **Config**: Resumed from 2026-03-14_23-10-04 checkpoint. Same reward weights.
- **Goal**: Longer training to see if policy converges
- **Result**: Reward → +18.9 at iter 768. ep_len_mean still ~980. Timed out at 8h.
- **Key finding**: Reward improves but ep_len doesn't drop — robot not reliably reaching 6 goals per episode.

### 2026-03-15_12-31-59
- **Job**: 234721 | **Logs**: `train_234721.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-15_12-31-59/` (checkpoints only)
- **Config**: Fresh start, 256 envs, same old reward weights
- **Goal**: Compare fresh start vs resumed training
- **Result**: Peak +18.8 at iter 760. Similar to 234706.

### 2026-03-15_13-09-00
- **Job**: 234736 | **Logs**: `train_234736.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-15_13-09-00/` (checkpoints only)
- **Config**: Fresh start, same old weights
- **Result**: Peak +18.6 at iter 763. Consistent with other runs.

### 2026-03-16_03-25-15
- **Job**: 235054 | **Logs**: `train_235054.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-16_03-25-15/model.zip`
- **Config**: New weights (goal_reached +10.0, velocity_near_goal -0.3/0.5m, collision -2.0 via contact sensor, obstacle_proximity -0.5/0.8m). First run with contact sensor.
- **Goal**: Test updated reward weights to fix premature stopping
- **Result**: Peak +14.6 at 1000 iters. Scenario eval: 0/6 passed. Robot still stops at 0.5-0.9m.
- **Key finding**: Reward numbers not directly comparable to old weights. Contact sensor collision detection working.

### 2026-03-17_04-24-01
- **Job**: 235402 | **Logs**: `train_235402.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-17_04-24-01/model.zip`
- **Config**: position_tracking +5.0, velocity_near_goal -0.3, collision -2.0 contact sensor, obstacle_proximity -0.5, goal_reached +10.0, action_smoothness -0.1
- **Goal**: Higher position_tracking weight to incentivize approach
- **Result**: Peak +5.9 at 1000 iters. Poor performance.
- **Key finding**: High position_tracking (+5.0) made crashing through furniture profitable.

### 2026-03-17_04-35-51
- **Job**: 235405 | **Logs**: `train_235405.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-17_04-35-51/model.zip`
- **Config**: Same as 235402 but different seed or config variant
- **Goal**: Same as above
- **Result**: Peak +174 at 1000 iters. Very high reward but scenario eval 0/6 passed. Robot driving backward, lots of collisions.
- **Key finding**: High reward doesn't mean good behavior. Robot maximized reward by crashing through obstacles.

### 2026-03-22_18-25-55
- **Job**: 237451 | **Logs**: `train_237451.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-22_18-25-55/model.zip`
- **Config**: Simplified 3-term (position_tracking +2.0, goal_reached +10.0, collision -10.0). No video.
- **Goal**: First test of simplified reward structure from literature review
- **Result**: Peak +58.2 at 1000 iters. No curriculum.
- **Key finding**: Simplified rewards learn faster. Collision penalty at -10.0 discourages crashing.

### 2026-03-22_22-48-45
- **Job**: 237476 | **Logs**: `train_237476.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-22_22-48-45/` (checkpoints only)
- **Config**: 3-term + curriculum (Stage 1 only — curriculum thresholds were wrong, too high by 256x). With video.
- **Goal**: Test curriculum learning
- **Result**: Peak +102 at iter 581. Stuck on Stage 1 (empty room) due to wrong thresholds. Robot oscillates near goals.
- **Key finding**: Curriculum thresholds used common_step_counter × num_envs instead of just common_step_counter. Robot drives backward — no heading penalty.

### 2026-03-23_01-14-32
- **Job**: 237487 | **Logs**: `train_237487.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-23_01-14-32/` (checkpoints only)
- **Config**: 4-term (added position_tracking_fine +1.0 std=0.2) + curriculum (still wrong thresholds). With video.
- **Goal**: Fix near-goal oscillation with fine gradient
- **Result**: Peak +99.9 at iter 466. Still Stage 1 only.
- **Key finding**: Fine gradient added but curriculum thresholds still wrong.

### 2026-03-23_09-42-21
- **Job**: 237492 | **Logs**: `train_237492.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-23_09-42-21/`
- **Config**: 4-term (position_tracking +2.0, position_tracking_fine +1.0, goal_reached +10.0, collision -10.0) + curriculum with FIXED thresholds (153600 / 358400 steps). With video. 1500 iters.
- **Goal**: Full curriculum run with correct stage transitions
- **Result**: Running...

### 2026-03-23_12-07-03 (Job 237512)
- **Job**: 237512 | **Logs**: `train_237512.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-23_12-07-03/`
- **Config**: 4-term (position_tracking +2.0, position_tracking_fine +1.0, goal_reached +10.0, collision -10.0). Curriculum enabled.
- **Goal**: Parallel run to 237492 to compare curriculum behavior
- **Result**: Running... (iter ~684/1500, ep_rew_mean ~62-65 at 12h)

### 2026-03-23_21-20-54 (Job 237611)
- **Job**: 237611 | **Logs**: `train_237611.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-23_21-20-54/`
- **Config**: position_tracking +2.0, position_tracking_fine +1.0, **goal_reached +100.0**, collision -10.0. Curriculum enabled.
- **Change from previous**: Increased goal_reached from 10 → 100 to make arrival the dominant signal (~58% of per-goal reward). Hypothesis: robot was learning to hover near goal instead of committing to arrival because coarse tracking accumulated more reward than the goal bonus.
- **Goal**: Test whether larger arrival bonus fixes near-goal oscillation
- **Result**: Running...

### 2026-03-23_22-58-20 (Job 237629)
- **Job**: 237629 | **Logs**: `train_237629.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-23_22-58-20/`
- **Config**: **goal_progress +2.0**, goal_reached +100.0, collision -10.0. Curriculum enabled. Removed both position_tracking terms.
- **Change from previous**: Replaced tanh distance tracking with progress reward (`forward_vel × cos(heading_error)`). Progress reward = instantaneous rate of distance closing in m/s. Rewards efficient direct paths; gives 0 for hovering, negative for moving away.
- **Goal**: Incentivize best-path navigation. Tanh tracking rewarded being close (hover-friendly); progress reward only rewards actively closing distance each step.
- **Result**: Running...

### 2026-03-24_00-41-07 (Job 237659)
- **Job**: 237659 | **Logs**: `train_237659.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-24_00-41-07/`
- **Config**: delta_progress +3.0, distance_to_goal +1.0 (std=0.3), time_penalty -0.05, goal_reached +100.0, collision -10.0. Curriculum enabled.
- **Change from previous**: Replaced velocity-based `goal_progress` with `delta_progress` (d_prev - d_current). Added steep tanh proximity magnet (std=0.3) active only inside goal zone. Added per-step time penalty (-0.05) to make hovering always net-negative.
- **Why**: goal_progress (forward_vel × cos(heading_error)) approximates displacement but can be gamed near goal where heading error is noisy. delta_progress measures actual displacement directly and cannot be gamed by oscillating. time_penalty ensures any non-progressing step costs the robot. Break-even: robot profits from a step at ≥0.017 m/s toward goal (3.0×0.017 > 0.05).
- **Result**: Running...

### Job 238023 — Cancelled (disk full, never started)

### 2026-03-24_23-22-03 (Job 238036)
- **Job**: 238036 | **Logs**: `train_238036.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-24_23-22-03/` (no weights saved — crashed immediately)
- **Config**: Same as 237659. Includes DebugOverlayWrapper on training videos.
- **Change from previous**: Resubmit of 238023 (cancelled). Added DebugOverlayWrapper for video diagnostics.
- **Result**: Failed — `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` at `train_nova_carter.py:140`. Bug: `done = terminated or truncated` doesn't work when these are tensors. Fixed to `np.logical_or(terminated, truncated)`.

### 2026-03-24_23-45-19 (Job 238049)
- **Job**: 238049 | **Logs**: `train_238049.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-24_23-45-19/`
- **Config**: Same as 237659 (delta_progress +3.0, distance_to_goal +1.0 std=0.3, time_penalty -0.05, goal_reached +100.0, collision -10.0). Curriculum enabled. DebugOverlayWrapper enabled.
- **Change from previous**: Bug fix — `done = terminated or truncated` → `np.logical_or(terminated, truncated)` in DebugOverlayWrapper.step().
- **Result**: Failed — `TypeError: can't convert cuda:0 device type tensor to numpy`. `np.logical_or` tried to call `.numpy()` on CUDA tensors. Fixed with explicit `.cpu().numpy()` conversion.

### 2026-03-25_00-06-50 (Job 238060)
- **Job**: 238060 | **Logs**: `train_238060.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_00-06-50/`
- **Config**: Same as 238049. DebugOverlayWrapper enabled.
- **Change from previous**: Bug fix — CUDA tensor `.cpu().numpy()` conversion before `np.logical_or` in DebugOverlayWrapper.step().
- **Result**: Running...

### 2026-03-25_01-18-16 (Job 238083)
- **Job**: 238083 | **Logs**: `train_238083.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_01-18-16/`
- **Config**: Same as 238060. DebugOverlayWrapper with ep_rew_mean display. Episode-triggered video (every 10th episode).
- **Change from previous**: Added `ep_rew_mean` to overlay (reads from `agent.ep_info_buffer`). Switched video from step_trigger to episode_trigger (every 10th episode, 1100 step cap). Added full-screen episode end banner.
- **Result**: Cancelled by user.

### 2026-03-25_02-30-19 (Job 238124)
- **Job**: 238124 | **Logs**: `train_238124.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_02-30-19/`
- **Config**: Same reward structure as 237659. DebugOverlayWrapper with cumulative episode reward display. Episode-triggered video (every 10th episode).
- **Change from previous**: Replaced `ep_rew_mean` (rolling mean from agent buffer) with `ep_reward` (cumulative sum of env 0's current episode). Resets to 0 at each episode start.
- **Result**: Cancelled.

### Job 238125 — Cancelled (superseded by 238126)

### 2026-03-25_03-26-13 (Job 238126)
- **Job**: 238126 | **Logs**: `train_238126.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_03-26-13/`
- **Config**: Same reward structure as 237659. Full DebugOverlayWrapper: step/stage/dist, per-step reward, cumulative ep reward, contact force (N). Episode-triggered video.
- **Change from previous**: Added contact force reading to overlay (`contact_sensor.data.net_forces_w_history` max norm for env 0). Diagnoses whether collision sensor is dead, below threshold, or working.
- **Result**: Cancelled.

### 2026-03-25_04-07-17 (Job 238133)
- **Job**: 238133 | **Logs**: `train_238133.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_04-07-17/`
- **Config**: Same reward structure as 237659. Full overlay: step/stage/dist, reward, ep_reward, contact force. Episode-triggered video.
- **Change from previous**: On episode end, displays raw info dict key/values at bottom of banner so we can see exact structure and fix termination reason parsing.
- **Result**: Cancelled at iter 756, ep_rew_mean=-5.01. episode_trigger bug: RecordVideo never detected episode ends because Isaac Lab returns tensors not scalar bools — only 1 video recorded. Fixed by reverting to step_trigger.

### 2026-03-25_17-30-31 (Job 238291)
- **Job**: 238291 | **Logs**: `train_238291.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_17-30-31/`
- **Config**: Same reward structure as 237659. Full overlay: step/stage/dist, reward, ep_reward, contact force, raw info dict on episode end.
- **Change from previous**: Reverted to step_trigger (every 5000 steps, 900 steps long). episode_trigger doesn't work with Isaac Lab because SB3 auto-resets within step() without calling reset(), so RecordVideo's episode counter never increments.
- **Result**: Running...

### 2026-03-25_19-09-33 (Job 238333)
- **Job**: 238333 | **Logs**: `training/logs/train_238333.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-25_19-09-33/`
- **Config**: Same reward structure as 237659 (delta_progress +3.0, distance_to_goal +1.0 std=0.3, time_penalty -0.05, goal_reached +100.0, collision -10.0). Curriculum enabled. Full DebugOverlayWrapper. Step-triggered video.
- **Changes from 238291**:
  1. `np.any(done)` → `bool(done[0])` — overlay only tracks env 0, stops false "RESET" banners
  2. `_ep_reward_sum` no longer wiped by other-env resets — ep_reward now accumulates correctly
  3. `_parse_term_reason` uses `termination_manager.get_term(name)[0]` — correct BAD_ORIENTATION / GOAL_COMPLETE labels
  4. `contact_sensor history_length` 2 → 5 — captures all 5 physics sub-steps per RL step
  5. `_get_contact_force` uses `net_forces_w_history` (max over history) to match reward logic
  6. One-time diagnostic print: `num_bodies` and `max_force_history` to SLURM log
- **Hypothesis**: With correct env-0-only tracking, ep_reward should now show collision (−10) and goal (+100) spikes. Contact force display should be non-zero when reward fires.
- **Result**: Running...

### TBD (Job 238496)
- **Job**: 238496 | **Logs**: `training/logs/train_238496.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/TBD (Job 238496)/`
- **Changes from 238479**: Added `backward_motion` reward term (weight=-2.0). Penalizes `relu(-forward_vel_x)` — fires whenever robot moves casters-first (negative body-frame x velocity). Also carries forward the shorter Stage 1 curriculum (stage1_steps=78125, ~152 iters).
- **Hypothesis**: Robot will learn to always drive sensor-end first. At 0.5 m/s backward the penalty is −1.0/step vs +1.5/step for equivalent forward motion, so reversing toward the goal is always net-negative.
- **Result**: TBD

### 2026-03-26_00-58-11 (Job 238479)
- **Job**: 238479 | **Logs**: `training/logs/train_238479.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-26_00-58-11/`
- **Changes from previous**: Shortened Stage 1 curriculum from 300 iters (~39M timesteps) to 152 iters (~20M timesteps). `stage1_steps`: 153,600 → 78,125. Stage 2 and 3 boundaries unchanged.
- **Hypothesis**: Agent already masters empty-room navigation by ~20M timesteps; letting it spend more time in Stages 2 and 3 (furniture + longer goals) should improve final performance.
- **Result**: TBD

### TBD (Job 238442)
- **Job**: 238442 | **Logs**: `training/logs/train_238442.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/TBD (Job 238442)/`
- **Config**: delta_progress +3.0, distance_to_goal +1.0 std=0.3, time_penalty -0.05, goal_reached +100.0, collision -10.0. Curriculum enabled. Full DebugOverlayWrapper. Step-triggered video (every 5000 steps, 900 steps long). 1500 iters.
- **Changes from 238333**: Fixed 3 reward functions to use world-frame distance instead of stale body-frame command (`pos_command_b`):
  1. `goal_reached` — was structurally never firing: `get_command()[:,:2]` returns `pos_command_b` updated in `command_manager.compute()` which runs AFTER rewards. By the time rewards ran, the goal had already been resampled to the new far-away target → reward always saw >0.75m. Fixed to `pos_command_w - root_pos_w` (world-frame, not yet resampled when rewards run).
  2. `position_tracking_tanh` — same stale-command issue; proximity gradient was collapsing to zero on arrival.
  3. `delta_progress` — minor 1-step lag, same fix for accuracy.
- **Hypothesis**: `goal_reached` (+100) now fires every time robot crosses 0.75m. Expect ep_rew_mean to jump significantly (from ~150-200 baseline to ~600+ for 6 goals per episode). ep_len_mean should drop as command_resample terminates more often.
- **Result**: Running...

### 2026-03-27_02-43-38 (Job 238807)
- **Job**: 238807 | **Logs**: `training/logs/train_238807.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-27_02-43-38/`
- **Config**: delta_progress +3.0, distance_to_goal +1.0 std=0.3, time_penalty -0.05, backward_motion -2.0, goal_reached +100.0, collision -10.0. Curriculum enabled. Full DebugOverlayWrapper. Step-triggered video.
- **Changes from previous**: Skipped Stage 1 entirely (`stage1_steps=0`). Stage 2 → Stage 3 transition at ~100 iters / ~26M TensorBoard timesteps (`stage2_steps=51200`). Robot starts immediately in Stage 2 (3 furniture, ±3m goals) then moves to Stage 3 (all 7 furniture, ±4m goals) quickly.
- **Hypothesis**: Robot has already demonstrated it can navigate an empty room. Skipping Stage 1 and spending more training budget in Stages 2/3 should improve obstacle avoidance and longer-range navigation performance.
- **Result**: TBD

### 2026-03-29_22-59-11
- **Job**: 239123 | **Logs**: `training/logs/train_239123.out/.err`
- **Weights**: `logs/sb3/NovaCarter-Patrol/2026-03-29_22-59-11/`
- **Config**: delta_progress +3.0, distance_to_goal +1.0 std=0.3, time_penalty -0.05, backward_motion -2.0, goal_reached +100.0, collision -10.0. Curriculum enabled (stage1=25600, stage2=102400, stage3=179200 steps). Full DebugOverlayWrapper. Step-triggered video. 500 iters.
- **Changes from 238807**: Per-stage model saving — `BestModelCallback` now saves `best_model_stage{N}.zip` and `latest_model_stage{N}.zip` per curriculum stage. Reward threshold resets on stage advance so the best model at each difficulty level is captured independently. Previous global `best_model.zip` replaced by per-stage files.
- **Hypothesis**: No behavior change — purely a checkpointing improvement. Should now have meaningful best checkpoints at each curriculum stage rather than only the Stage 1 snapshot.
- **Result**: TBD

### EVAL-238830
- **Job**: 238830 | **Logs**: `logs/eval_scenarios_238830.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-26_00-58-11/best_model.zip`
- **Result**: Running...

### EVAL-238831
- **Job**: 238831 | **Logs**: `logs/eval_scenarios_238831.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-26_01-51-02/best_model.zip`
- **Result**: Running...

### EVAL-238832
- **Job**: 238832 | **Logs**: `logs/eval_scenarios_238832.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-27_01-05-35/best_model.zip`
- **Result**: Running...

### TBD (Job 240780)
- **Job**: 240780 | **Logs**: `training/logs/train_240780.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/TBD/`
- **Changes from 240773**: Collision weight reduced -30.0 → -20.0 (effective -1.67/step vs -2.50/step). Gives robot more latitude to nudge through tight gaps rather than freezing.
- **Hypothesis**: Slightly more exploratory behavior near obstacles; robot less likely to stop in front of furniture.
- **Result**: TBD

### TBD (Job 240773)
- **Job**: 240773 | **Logs**: `training/logs/train_240773.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/TBD/`
- **Changes from 240766**: Fixed patrol termination — replaced `command_resample` (== exact match + 15s timer delay) with `patrol_complete` (>= num_goals, fires immediately). Episodes now correctly end at 6 goals instead of running to 83s timeout.
- **Hypothesis**: `ep_len_mean` should drop significantly. "GOAL_COMPLETE" should appear in overlay videos. Training signal should be cleaner with consistent 6-goal episodes.
- **Result**: TBD

### TBD (Job 240766)
- **Job**: 240766 | **Logs**: `training/logs/train_240766.out/.err`
- **Weights**: `training/logs/sb3/NovaCarter-Patrol/TBD/`
- **Changes from 240712**:
  - Curriculum thresholds corrected (were 25600/102400/179200 — all hit in <1 iter due to 256x bug): stage1=51200 (~100 iters), stage2=128000 (~250 iters), stage3=230400 (~450 iters)
  - New reward term: `obstacle_proximity` (weight=-0.5, LiDAR, safe_dist=1.0m) — anticipatory avoidance before contact
  - Fixed `_get_curriculum_stage()` in both `BestModelCallback` and `DebugOverlayWrapper` — was using `_terms` (doesn't exist), now uses `_class_term_cfgs[].func.current_stage`; checkpoints now correctly labeled stage1/2/3/4
  - SLURM time limit set to 12 hours (was missing from submit script entirely)
- **Hypothesis**: With correct stage durations the robot should meaningfully train in each stage. Stage 1 empty room (100 iters) → Stage 2 blocker (150 iters) → Stage 3 corridor (200 iters) → Stage 4 full clutter (remaining ~425 iters)
- **Result**: TBD

### EVAL-240743
- **Job**: 240743 | **Logs**: `training/logs/eval_scenarios_240743.out/.err`
- **Checkpoint**: `training/logs/sb3/NovaCarter-Patrol/2026-04-05_03-21-31/latest_model_stage0.zip`
- **Goal**: Evaluate mid-training snapshot of run 240712 (collision -30, ~440 iters in)
- **Result**: TBD

### EVAL-240742
- **Job**: 240742 | **Logs**: `training/logs/eval_scenarios_240742.out/.err`
- **Checkpoint**: `training/logs/sb3/NovaCarter-Patrol/2026-03-29_22-59-11/model.zip`
- **Goal**: Evaluate final model from run 239123 (500 iters, per-stage checkpoint run)
- **Result**: TBD

### EVAL-238833
- **Job**: 238833 | **Logs**: `logs/eval_scenarios_238833.out/.err`
- **Checkpoint**: `logs/sb3/NovaCarter-Patrol/2026-03-27_02-43-38/best_model.zip`
- **Result**: Running...

---

## File Locations

| File | Location |
|------|----------|
| Training logs | `training/logs/train_<JOBID>.out` and `.err` |
| Model weights | `training/logs/sb3/NovaCarter-Patrol/<run>/model.zip` or `best_model.zip` |
| Checkpoints | `training/logs/sb3/NovaCarter-Patrol/<run>/model_<steps>_steps.zip` |
| TensorBoard | `training/logs/sb3/NovaCarter-Patrol/<run>/PPO_1/` |
| Training videos | `training/logs/sb3/NovaCarter-Patrol/<run>/videos/train/` |
| Scenario eval videos | `training/logs/sb3/NovaCarter-Patrol/<run>/videos/scenarios/` |
| Scenario results | `training/logs/sb3/NovaCarter-Patrol/<run>/scenario_results_<timestamp>.json` |
| Env config snapshot | `training/logs/sb3/NovaCarter-Patrol/<run>/params/env.yaml` |
| Agent config snapshot | `training/logs/sb3/NovaCarter-Patrol/<run>/params/agent.yaml` |
