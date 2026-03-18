### 3.4 RL Navigation Training Infrastructure

The reinforcement learning subsystem trains a local navigation policy that receives goal waypoints from the navigation graph (Section 3.3) and learns to reach them while avoiding obstacles. Training is conducted entirely in simulation using NVIDIA Isaac Lab on MSOE's Rosie supercomputer, with the trained policy targeting deployment on the physical NVIDIA Nova Carter platform.

#### 3.4.1 Robot Platform

The NVIDIA Nova Carter is a differential-drive autonomous mobile robot built on the Segway RMP Lite 220 chassis. The platform measures 0.722 m x 0.500 m x 0.556 m (L x W x H) with a net weight of 51.2 kg. It is driven by two powered wheels (0.14 m radius, 0.4 m wheel base) with passive caster wheels providing rear stability. The robot's sensor suite includes a Hesai XT-32 3D LiDAR mounted atop the sensor tower, two RPLidar S2E 2D LiDARs (front and rear), four Owl stereo cameras providing 360-degree visual coverage, and an internal IMU. Onboard compute is provided by an NVIDIA Jetson AGX Orin.

Nova Carter was selected for three reasons. First, it has native support in both NVIDIA Isaac Sim and Isaac Lab, with a validated USD robot model available in the NVIDIA asset library. This eliminates the need to author or debug a custom robot description and ensures that the simulated robot faithfully represents the physical platform's articulation structure, mass distribution, and sensor mounting geometry. Second, the platform is fully embedded in the NVIDIA robotics ecosystem — Isaac Sim for simulation, Isaac Lab for RL training, and the Jetson AGX Orin for edge deployment — providing a unified software stack from training to physical operation. Third, using the same robot platform in simulation and deployment minimizes the sim-to-real gap for locomotion dynamics, a critical factor for transfer of learned navigation policies.

In Isaac Lab, the Nova Carter is configured as an articulation with eight rigid body links and seven joints. The two drive wheel joints (`joint_wheel_left`, `joint_wheel_right`) are actuated as velocity-controlled continuous joints using implicit actuators with zero stiffness and high damping (1e5), implementing pure velocity control. The five caster joints are configured as passive free-spinning joints with zero effort limit, stiffness, and damping. The action space operates at the level of linear and angular velocity commands rather than direct wheel velocities; a custom `DifferentialDriveAction` term converts these commands to individual wheel angular velocity targets using standard differential drive kinematics:

v_left  = (v - omega * d / 2) / r
v_right = (v + omega * d / 2) / r

where v is the commanded linear velocity, omega is the commanded angular velocity, d = 0.4 m is the wheel base, and r = 0.14 m is the wheel radius. This abstraction decouples the policy from platform-specific mechanical parameters and matches the velocity command interface used by the physical Nova Carter's ROS 2 driver.

#### 3.4.2 Training Environment Setup

The Isaac Lab training environment uses a manager-based architecture in which the scene, actions, observations, rewards, terminations, and reset events are each defined as modular configuration objects composed into a single `ManagerBasedRLEnvCfg`. This structure allows individual components to be modified or replaced without affecting the rest of the pipeline.

The current training environment is a 10 m x 10 m walled room populated with seven randomized furniture obstacles (table, bookshelf, chair, desk, filing cabinet, couch, and trash bin). Each furniture piece is assigned to a distinct spatial zone within the room and randomizes its position and yaw orientation on every episode reset. The robot spawns near the room center (within +/- 1.0 m) to avoid overlap with furniture. This test environment serves as a development and validation environment for the full training pipeline; it is not intended to represent the target building. The full BIM-derived building environment (Section 3.2) is the intended training scene for the final policy.

Physics simulation runs at 60 Hz with a control decimation factor of 5, yielding an effective policy control frequency of 12 Hz. Each episode lasts a maximum of 83 seconds (1000 control steps). The environment supports GPU-parallelized execution through Isaac Lab's vectorized environment interface, with up to 256 environment instances running simultaneously on a single GPU. Environment instances are spaced 12.0 m apart to prevent inter-environment interference.

Training runs on MSOE's Rosie supercomputer using NVIDIA T4 GPUs (16 GB VRAM) on the teaching partition, with jobs submitted through SLURM. Each training job requests a single T4 GPU and 64 GB of system memory. The RL algorithm is Proximal Policy Optimization (PPO) [8] implemented through Stable-Baselines3, using an MLP policy with two hidden layers of 256 units each and ReLU activations. Observations are normalized using VecNormalize (observations only; reward normalization is disabled to prevent numerical instability with short episodes). Key hyperparameters include a learning rate of 3e-4, batch size of 256, 512 steps per rollout, GAE lambda of 0.95, and a target KL divergence of 0.03 for early stopping of policy updates.

At the time of writing, the training pipeline has been validated end-to-end in the test environment: the simulation runs, the agent collects experience, PPO updates converge, and checkpoints are saved and evaluated automatically. All preliminary results reported in Section 4.3 are from this test environment. Training in the full BIM-derived building environment is ongoing.

#### 3.4.3 Observation and Action Spaces

The observation space is a 26-dimensional continuous vector composed of five components. Table 1 summarizes the observation structure.

**Table 1: Observation space (26 dimensions)**

| Component | Dimensions | Description |
|---|---|---|
| LiDAR scan | 20 | 360-degree planar scan, 20 rays at 18-degree resolution, each normalized to [0, 1] by dividing by the 10 m maximum range |
| Goal position | 2 | Relative (x, y) displacement to the goal waypoint, expressed in the robot's body frame |
| Goal heading | 2 | Heading error to the goal encoded as (cos theta, sin theta) to avoid discontinuities at +/- pi |
| Forward velocity | 1 | Robot's linear velocity along its forward axis (m/s) |
| Yaw rate | 1 | Robot's angular velocity about the vertical axis (rad/s) |

The LiDAR is implemented as a `MultiMeshRayCaster` sensor attached to the robot's `chassis_link` at 0.3 m height, casting 20 rays in a full 360-degree horizontal sweep. The raycaster detects intersections with walls and furniture meshes. All observation values are sanitized to replace NaN and infinity values with safe defaults before being passed to the policy network.

The action space is a 2-dimensional continuous vector representing linear velocity and angular velocity commands. The policy network outputs values in [-1, 1] for each dimension, which are scaled to physical velocity ranges as shown in Table 2.

**Table 2: Action space (2 dimensions)**

| Action | Policy output range | Scaled range | Unit |
|---|---|---|---|
| Linear velocity | [-1, 1] | [-0.5, 0.5] | m/s |
| Angular velocity | [-1, 1] | [-1.0, 1.0] | rad/s |

These velocity commands are converted to individual wheel angular velocity targets via differential drive kinematics (Section 3.4.1) and applied to the simulated wheel joints at each control step.

During training, goal waypoints are sampled uniformly within the room interior (x in [-4.0, 4.0] m, y in [-4.0, 4.0] m) with random heading. In the full system, these goal positions will be replaced by waypoints from the navigation graph (Section 3.3), so the agent navigates along planned patrol routes rather than to random positions. The observation interface remains identical — the agent always receives the next waypoint as a relative position in its body frame, regardless of whether the waypoint was sampled randomly or provided by the route planner.

#### 3.4.4 Reward Function

The reward function uses six terms designed to produce a policy that reliably reaches goal waypoints while maintaining safe clearance from obstacles. The reward structure follows established patterns in the navigation RL literature, where successful systems consistently employ dense distance progress, a sparse goal bonus, and a sparse collision penalty as core signals [1, 3]. Table 3 summarizes the reward terms and their weights.

**Table 3: Reward function terms**

| Term | Weight | Function | Purpose |
|---|---|---|---|
| Position tracking | +5.0 | 1 - tanh(d / sigma), sigma = 2.0 | Dense gradient toward the goal at all distances |
| Goal reached | +10.0 | 1.0 if d < 0.5 m, else 0.0 | Large sparse bonus for successful arrival |
| Collision | -2.0 | Contact force on chassis > 1.0 N | Penalizes physical collisions via contact sensor |
| Obstacle proximity | -0.5 | max(0, 1 - d_min / 0.8) | Smooth penalty as LiDAR minimum range decreases below 0.8 m |
| Velocity near goal | -0.3 | Speed when within 0.5 m of goal | Teaches deceleration on final approach |
| Action smoothness | -0.1 | L2 norm of action rate | Penalizes jerky velocity commands |

The **position tracking** term provides continuous reward signal using a tanh function with standard deviation sigma = 2.0 m. This formulation produces strong gradient when the robot is far from the goal and saturates smoothly as it approaches, avoiding the reward sparsity that prevents learning in large environments.

The **goal reached** bonus provides an explicit sparse success signal of +10.0 when the robot comes within 0.5 m of the goal. This threshold accounts for the robot's physical dimensions (0.36 m from chassis center to bumper) and provides a clear incentive to close the final distance rather than hovering nearby. Upon reaching a goal, the environment immediately resamples a new goal position using a custom `GoalReachPose2dCommand` that follows the goal-reached resampling pattern from Isaac Lab's manipulation tasks. This teaches patrol behavior — the agent learns to navigate sequentially through a series of goals within a single episode. An episode terminates after six successful goal reaches or at the 83-second time limit.

The **collision** penalty uses a physics-based contact sensor on the robot's `chassis_link` rather than a LiDAR-based proximity threshold. This approach detects actual physical contact regardless of direction and avoids false negatives caused by the 0.36 m offset between the LiDAR origin and the robot's bumper perimeter. The contact sensor reports net force on the chassis body; any force exceeding 1.0 N triggers the penalty. Collisions are penalized but do not terminate the episode, as termination on collision was found to cause a training death spiral when the robot spawns near furniture.

The **obstacle proximity** term provides a proactive avoidance signal before physical contact occurs. It reads the minimum LiDAR range across all 20 rays and applies a linearly increasing penalty as the minimum range drops below the 0.8 m safe distance threshold. This complements the collision penalty: obstacle proximity encourages the agent to maintain clearance, while the collision penalty provides a stronger corrective signal when clearance is violated.

The **velocity near goal** term penalizes the robot's speed when it is within 0.5 m of the goal. Without this term, the agent learns to approach goals at full speed and either overshoots or orbits the target. The penalty is proportional to the robot's speed, teaching smooth deceleration on the final approach.

The **action smoothness** term penalizes the L2 norm of the change in actions between consecutive time steps. This discourages rapid oscillations in velocity commands that would be difficult to execute on physical hardware and would produce uncomfortable jerky motion.

The reward structure deliberately omits a heading alignment term. While heading penalties appear in approximately half of published navigation RL systems, they conflict with obstacle avoidance behavior: when the robot must turn away from the goal to navigate around an obstacle, an additive heading penalty punishes the correct action. The agent learns heading implicitly from the position tracking gradient, consistent with several successful sim-to-real transfer results in the literature that explicitly exclude heading from the reward [1].
