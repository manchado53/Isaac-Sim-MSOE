# NVIDIA Isaac Sim & Isaac Lab -- Comprehensive Reference

> **Document Version:** April 2026
> **Isaac Sim Latest Stable:** 5.1.0 (pip/binary) | 6.0.0 (Early Developer Release, source-only)
> **Isaac Lab Latest:** v2.3.2
> **Official Docs:** https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
> **Isaac Lab Docs:** https://isaac-sim.github.io/IsaacLab/main/index.html
> **GitHub (Isaac Sim):** https://github.com/isaac-sim/IsaacSim
> **GitHub (Isaac Lab):** https://github.com/isaac-sim/IsaacLab

---

## 1. What Is Isaac Sim

NVIDIA Isaac Sim is a **robotics simulation platform built on NVIDIA Omniverse** for developing, testing, training, and deploying AI-powered robots in physically-based virtual environments.

### Key Capabilities

- **Robot Import:** URDF, MJCF (MuJoCo XML), Onshape, CAD formats
- **Physics:** GPU-accelerated PhysX engine (rigid bodies, articulations, deformable bodies, fluids)
- **Rendering:** RTX ray-traced rendering for photorealistic sensor simulation
- **Sensors:** Cameras (RGB, depth, segmentation), RTX Lidar, RTX Radar, IMU, contact sensors, effort sensors
- **Synthetic Data:** Omniverse Replicator integration for labeled dataset generation
- **ROS 2 Integration:** Native bridge for topics, services, TF, navigation
- **Robot Learning:** Isaac Lab framework for RL, imitation learning, motion planning
- **USD-Based:** Universal Scene Description as the scene interchange format
- **Multi-Workflow:** GUI, standalone Python scripts, Jupyter Notebooks, VS Code, headless mode

### Architecture Stack

```
+--------------------------------------------------+
|  Isaac Lab (robot learning framework)            |
+--------------------------------------------------+
|  Isaac Sim (robotics simulation application)     |
+--------------------------------------------------+
|  NVIDIA Omniverse Kit (application framework)    |
+--------------------------------------------------+
|  PhysX (GPU physics)  |  RTX Renderer           |
+--------------------------------------------------+
|  USD (Universal Scene Description)               |
+--------------------------------------------------+
```

### Use Cases

- Digital twins of robots and environments
- Sensor simulation and synthetic data generation
- Reinforcement learning and policy training
- Sim-to-real transfer
- Motion planning and control validation
- Multi-robot fleet simulation
- ROS 2 application development and testing

---

## 2. Installation & Setup

### System Requirements (Isaac Sim 6.0)

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| **OS** | Ubuntu 22.04/24.04, Windows 10/11 | Ubuntu 22.04 | Ubuntu 22.04 |
| **CPU** | Intel i7 (7th Gen) / AMD Ryzen 5 | Intel i7 (9th Gen) / AMD Ryzen 7 | Intel i9 X / AMD Threadripper |
| **CPU Cores** | 4 | 8 | 16 |
| **RAM** | 32 GB | 64 GB | 64 GB+ |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16 GB | 16 GB | 48 GB |
| **Storage** | 50 GB SSD | 500 GB SSD | 1 TB NVMe SSD |
| **Driver (Linux)** | 580.65.06+ | | |
| **Driver (Windows)** | 580.88+ | | |

**Important:** GPUs without RT Cores (A100, H100) are NOT supported. GPUs with less than 16 GB VRAM may be insufficient for complex scenes.

### Installation Method 1: Pip Install (Recommended for Python development)

**Requires Python 3.12** (Isaac Sim 6.0) or **Python 3.11** (Isaac Sim 5.x).

```bash
# Create virtual environment
python3.12 -m venv env_isaacsim
source env_isaacsim/bin/activate  # Linux
# env_isaacsim\Scripts\activate   # Windows

pip install --upgrade pip

# Install PyTorch (CUDA 12)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Sim (full)
pip install isaacsim[all,extscache]==6.0.0 --extra-index-url https://pypi.nvidia.com

# Accept EULA
export OMNI_KIT_ACCEPT_EULA=YES  # Linux
# set OMNI_KIT_ACCEPT_EULA=YES   # Windows

# Launch
isaacsim                    # GUI mode
python path/to/script.py    # Run scripts
```

**Available pip bundles:** `all`, `extscache`, `compatibility-check`, `ros2`

### Installation Method 2: Workstation (Binary Download)

```bash
# Linux
mkdir ~/isaacsim
cd ~/Downloads
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./isaac-sim.sh

# Windows
mkdir C:\isaacsim
cd %USERPROFILE%/Downloads
tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C C:\isaacsim
cd C:\isaacsim
post_install.bat
isaac-sim.bat
```

First launch takes 5-10 minutes for shader cache warmup.

### Installation Method 3: Docker Container

```bash
# Pull image
docker pull nvcr.io/nvidia/isaac-sim:6.0.0-dev2

# Create cache directories
mkdir -p ~/docker/isaac-sim/{cache/main/ov,cache/main/warp,cache/computecache,config,data/documents,data/Kit,logs,pkg}
sudo chown -R 1234:1234 ~/docker/isaac-sim

# Run container (headless with livestream)
docker run --name isaac-sim --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" --rm --network=host \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -u 1234:1234 nvcr.io/nvidia/isaac-sim:6.0.0-dev2

# Inside container, launch headless
./runheadless.sh -v
```

**Streaming ports:** 49100 (TCP, WebRTC signaling), 47998 (UDP, media), 8210 (TCP, web viewer)

### Installation Method 4: Build from Source

```bash
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim
git lfs install && git lfs pull
./build.sh    # Linux
# build.bat   # Windows
# Output: _build/[platform]/release/
```

---

## 3. Core Concepts

### USD (Universal Scene Description)

USD is the open-source 3D scene format used as the foundation of Omniverse and Isaac Sim.

- **Stage:** The root container holding all scene data (like a "document")
- **Prims (Primitives):** Basic building blocks / nodes in the scene graph (meshes, lights, cameras, xforms, etc.)
- **Attributes:** Key-value properties attached to prims (position, color, mass, etc.)
- **Relationships:** Connections between prims (e.g., material bindings)
- **Layers:** Composable files that can be stacked to build scenes non-destructively
- **Xform:** Transform prim representing position, rotation, and scale in the scene hierarchy

**Critical constraint:** All scene design must happen BEFORE simulation starts. Modifying the scene during simulation can corrupt GPU physics buffers.

### Key USD Python Modules (from `pxr`)

```python
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, PhysxSchema
```

| Module | Purpose |
|--------|---------|
| `Usd` | Core USD API (stages, prims, attributes) |
| `UsdGeom` | Geometry schemas (Mesh, Cube, Sphere, Xform, Camera, BBoxCache) |
| `UsdPhysics` | Physics schemas (RigidBodyAPI, CollisionAPI, JointAPI) |
| `UsdShade` | Material and shader definitions |
| `Sdf` | Scene description foundation (paths, layers, value types) |
| `Gf` | Math (Vec3d, Vec3f, Quatf, Matrix4d, Range3d) |
| `PhysxSchema` | PhysX-specific physics extensions |

### Accessing the USD Stage

```python
import omni.usd
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/Robot")
```

### PhysX Simulation

- GPU-accelerated rigid body, articulation, and soft body physics
- Supports position, velocity, and effort (force/torque) joint drive modes
- Collision types: Convex Hull, Convex Decomposition, Bounding Sphere, Bounding Cube, Triangle Mesh
- Physics materials control friction (static/dynamic) and restitution

### RTX Rendering

- Real-time ray tracing for photorealistic images
- Physically-based lighting, reflections, shadows, global illumination
- Used for camera and lidar sensor simulation
- Supports non-visual materials for sensor-specific responses

---

## 4. Python API

### API Generations

Isaac Sim has two API generations:
- **Core API** (`isaacsim.core.api`): Original wrapper around USD and physics APIs
- **Core Experimental API** (`isaacsim.core.experimental`): Newer, recommended API (introduced in 5.0.0) -- more robust and flexible

NVIDIA strongly encourages adopting the **Core Experimental API**.

### Standalone Application Pattern

```python
# For Isaac Sim standalone scripts (pip-installed)
import isaacsim
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})  # or True for headless

# --- Your simulation code here ---

simulation_app.close()
```

### Core Experimental API -- Key Classes

```python
# Scene and stage utilities
import isaacsim.core.experimental.utils.stage as stage_utils
stage_utils.create_new_stage()

# Objects
from isaacsim.core.experimental.objects import GroundPlane, Cube, DistantLight
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial

# Prims
from isaacsim.core.experimental.prims import XformPrim, RigidPrim, GeomPrim

# Example: Create ground plane
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

# Example: Create a dynamic cube
cube = Cube(paths="/dynamic_cube", positions=[0, 0.5, 1.0], sizes=0.3)
RigidPrim(paths="/dynamic_cube")
GeomPrim(paths="/dynamic_cube", apply_collision_apis=True)

# Example: Apply material
yellow = PreviewSurfaceMaterial("/Materials/yellow")
yellow.set_input_values("diffuseColor", [1.0, 1.0, 0.0])
cube.apply_visual_materials(yellow)

# Transform operations
xform = XformPrim(paths="/my_prim")
xform.set_world_poses([1.5, 1.2, 1.0], [0.7, 0.7, 0, 1])
xform.set_local_scales([1, 1.5, 0.2])
```

### Legacy Core API -- Key Classes

```python
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

DynamicCuboid(
    prim_path="/new_cube_2",
    name="cube_1",
    position=np.array([0, 0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2])
)
```

### Articulation Controller

```python
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import SingleArticulation, Articulation
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

# Create and apply joint commands
action = ArticulationAction(
    joint_positions=np.array([0.0, 0.0]),    # Position targets
    # joint_velocities=np.array([...]),       # OR velocity targets
    # joint_efforts=np.array([...]),          # OR effort/torque targets
    joint_indices=np.array([7, 8])           # Optional: specific joints
)
robot.apply_action(action)

# Read joint states
positions = robot.get_joint_positions()
```

**Constraint:** A joint can only be controlled by ONE method at a time (position, velocity, or effort).

### Raw USD API (Alternative)

```python
import omni
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
cubeGeom = UsdGeom.Cube.Define(stage, "/visual_cube_usd")
cubeGeom.CreateSizeAttr(0.5)
UsdGeom.Xformable(stage.GetPrimAtPath("/visual_cube_usd")).AddTranslateOp().Set(Gf.Vec3f(1.5, -0.2, 1.0))
```

---

## 5. Robot Simulation

### Importing Robots

#### URDF Import

```python
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

importer = URDFImporter(
    URDFImporterConfig(
        urdf_path="path/to/robot.urdf",
        usd_path="path/to/output",
        merge_mesh=True,
        allow_self_collision=True,
    )
)
output_path = importer.import_urdf()
```

**URDF Import Configuration Options:**
- `merge_mesh` -- consolidate meshes per rigid body
- `allow_self_collision` -- enable self-collision
- Collision type: Convex Hull, Convex Decomposition, Bounding Sphere, Bounding Cube
- Joint drive type: Position, Velocity, or None
- Joint drive strength (damping parameters)

**GUI:** File > Import, then select URDF

**CLI:**
```bash
./python.sh standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py --urdf_path /path/to/robot.urdf
```

#### MJCF Import

Extension: `isaacsim.asset.importer.mjcf` -- similar pattern to URDF importer for MuJoCo XML files.

### Robot Controller Types

1. **Articulation Controller** -- direct joint-level control (position/velocity/effort)
2. **Mobile Robot Controllers** -- differential drive, Ackermann steering, omnidirectional
3. **Manipulator Controllers** -- kinematics solvers, trajectory generation, path planning (Lula RRT, RMPflow)
4. **cuRobo / cuMotion** -- GPU-accelerated motion generation
5. **RL Policy Controllers** -- deploy trained neural network policies

### Sensors

| Sensor | Type | Key Outputs |
|--------|------|-------------|
| **Camera** | Vision | RGB, depth |
| **Depth Camera** | Vision | Stereoscopic depth, ground truth depth |
| **RTX Lidar** | RTX (ray-traced) | Point clouds, range data |
| **RTX Radar** | RTX (ray-traced) | Custom radar profiles |
| **IMU** | Physics | Acceleration, angular velocity |
| **Contact Sensor** | Physics | Collision forces |
| **Effort Sensor** | Physics | Motor/actuator loads |
| **Articulation Force Sensor** | Physics | Joint forces/torques |
| **Proximity Sensor** | Physics | Distance detection |

Sensors support both GUI configuration and standalone Python scripting.

---

## 6. ROS 2 Integration

### Architecture

Isaac Sim provides a native **ROS 2 Bridge** with both interactive (GUI) and standalone Python workflows.

### Supported Message Types

- **Geometry:** Pose, Transform (TF2 broadcasting)
- **Sensor:** Image, CompressedImage, PointCloud2, Imu, LaserScan
- **Control:** JointState, JointTrajectory, Twist
- **Navigation:** Odometry, OccupancyGrid, Path

### OmniGraph Nodes for ROS 2

- Generic Publisher / Subscriber
- Generic Server / Client
- Clock publisher (simulation time sync)
- Real Time Factor (RTF) publisher
- Camera/Lidar/TF publishers
- Custom Python and C++ OmniGraph nodes

### Key Features

- **Clock synchronization:** ROS 2 Clock node for temporal alignment
- **QoS support:** Configurable reliability, durability, history policies
- **Namespace generation:** Automatic namespace for multi-robot deployments
- **Navigation:** Multi-robot ROS 2 Navigation with heightmap support
- **MoveIt 2:** Trajectory planning and manipulation integration
- **RL Policy via ROS 2:** Deploy trained policies through ROS 2 control loops

### Tutorials Available

- Camera publishing (RGB, depth)
- Lidar and point cloud publishing
- TF tree broadcasting
- Joint state publishing/subscribing
- Turtlebot simulation
- Multi-robot navigation
- MoveIt 2 integration

---

## 7. Reinforcement Learning

### Isaac Lab (Primary RL Framework)

Isaac Lab is the **recommended framework for robot learning** built on Isaac Sim. It supersedes and replaces:
- IsaacGymEnvs (deprecated)
- OmniIsaacGymEnvs (deprecated)
- Orbit (deprecated)

### Supported RL Libraries

| Library | Backend | Algorithms | Multi-GPU | Multi-Agent |
|---------|---------|-----------|-----------|-------------|
| **SKRL** | PyTorch, JAX | Extensive | Yes | Yes |
| **RSL-RL** | PyTorch | PPO, Distillation | Yes | No |
| **RL-Games** | PyTorch | PPO, SAC, A2C | Yes | No |
| **Stable-Baselines3** | PyTorch | Extensive | No | No |

### Training Commands

```bash
# Training with different RL libraries
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
python scripts/reinforcement_learning/sb3/train.py --task Isaac-Humanoid-v0 --max_iterations 500 --headless
```

### Performance (RTX 4090, 65.5M steps, Isaac-Humanoid-v0)

- RSL-RL: 198 seconds
- RL-Games / SKRL: 201 seconds
- Stable-Baselines3: 287 seconds

### Key Features

- GPU-accelerated parallel environments (thousands simultaneously)
- Multi-GPU and multi-node distributed training
- Procedural terrain generation
- Actuator dynamics simulation
- Sensor simulation at variable frequencies
- 30+ pre-built task environments
- 16+ pre-configured robot models

---

## 8. Domain Randomization

Domain randomization is done through USD attribute manipulation and/or Omniverse Replicator.

### Light Randomization

```python
import numpy as np
light = stage.GetPrimAtPath("/World/Light")
light.GetAttribute("xformOp:translate").Set(
    (np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(4, 6))
)
light.GetAttribute("inputs:intensity").Set(np.random.normal(25000, 5000))
light.GetAttribute("inputs:colorTemperature").Set(np.random.normal(6500, 1000))
```

### Material / Texture Randomization

```python
from pxr import UsdShade, Sdf

shader = UsdShade.Shader.Get(stage, "/Materials/MyMaterial/Shader")
shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
shader.GetInput("diffuse_texture").Set(texture_path)
shader.GetInput("reflection_roughness_constant").Set(np.random.uniform(0.0, 1.0))
shader.GetInput("metallic_constant").Set(np.random.uniform(0.0, 1.0))
shader.GetInput("texture_scale").Set((scale, scale))
```

### Physics Randomization

```python
from pxr import UsdPhysics, PhysxSchema

physics_material = UsdPhysics.MaterialAPI.Get(stage, "/Materials/PhysMat")
physics_material.CreateStaticFrictionAttr().Set(np.random.uniform(0.01, 1.0))
physics_material.CreateDynamicFrictionAttr().Set(np.random.uniform(0.01, 1.0))
physics_material.CreateRestitutionAttr().Set(np.random.uniform(0.0, 0.5))

rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, "/World/Object")
PhysxSchema.PhysxRigidBodyAPI(rigid_body_api).CreateAngularDampingAttr().Set(0.9)
```

### Pose Randomization

```python
from pxr import Gf

prim = stage.GetPrimAtPath("/World/Object")
prim.GetAttribute("xformOp:translate").Set(
    Gf.Vec3d(np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(0.5, 2))
)
prim.GetAttribute("xformOp:rotateXYZ").Set(
    Gf.Vec3f(np.random.uniform(0, 360), np.random.uniform(0, 360), np.random.uniform(0, 360))
)
```

### SimReady Asset Discovery

```python
import omni.simready.explorer
assets = omni.simready.explorer.find_assets(keywords=["box", "container"])
```

---

## 9. Replicator / Synthetic Data Generation

Omniverse Replicator (`omni.replicator.core`) provides the SDG pipeline.

### Core Components

| Component | Purpose |
|-----------|---------|
| **Annotators** | Extract labeled data (RGB, depth, segmentation, bounding boxes, normals) |
| **Writers** | Output data to disk in specific formats (BasicWriter, KITTI, COCO) |
| **Randomizers** | Programmatic domain randomization |
| **Orchestrator** | Coordinates randomization and data capture per frame |

### Available Annotators

- `rgb` -- RGB image
- `distance_to_camera` / `distance_to_image_plane` -- Depth maps
- `semantic_segmentation` -- Per-pixel semantic labels
- `instance_segmentation` -- Per-pixel instance labels
- `bounding_box_2d_tight` / `bounding_box_2d_loose` -- 2D bounding boxes
- `bounding_box_3d` -- 3D bounding boxes
- `normals` -- Surface normal maps
- `cross_correspondence` -- Cross-frame correspondence

### Basic SDG Workflow

```python
import omni.replicator.core as rep

# Create render product
camera = rep.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
render_product = rep.create.render_product(camera, (1024, 1024))

# Setup writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/path/to/output",
    rgb=True,
    distance_to_camera=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True,
)
writer.attach(render_product)

# Run capture with randomization
for i in range(num_frames):
    # Apply randomizations (see domain randomization section)
    await rep.orchestrator.step_async(rt_subframes=16)

await rep.orchestrator.wait_until_complete_async()
```

### Async Execution Pattern

```python
async def run_sdg(num_frames):
    backend = rep.backends.get("DiskBackend")
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.attach(render_product)
    
    for _ in range(num_frames):
        # Randomize scene
        await rep.orchestrator.step_async(rt_subframes=16)
    
    await rep.orchestrator.wait_until_complete_async()
```

### GUI Tools

- **Semantics Schema Editor:** Tools > Replicator > Semantics Schema Editor
- **Synthetic Data Visualizer:** Real-time sensor output in viewport
- **Synthetic Data Recorder:** Tools > Replicator > Synthetic Data Recorder
- **Replicator YAML:** Tools > Replicator > Replicator YAML (config-file-driven SDG)

---

## 10. Extensions & Omniverse Ecosystem

### How Isaac Sim Fits Into Omniverse

- **NVIDIA Omniverse** is the platform/framework
- **Omniverse Kit** is the application framework for building Omniverse apps
- **Isaac Sim** is a Kit-based reference application for robotics
- **Nucleus** is the optional asset management server (NOT required for Isaac Sim)
- **USD** is the universal scene format connecting everything

### Extension System

Isaac Sim is composed of modular extensions that can be enabled/disabled. Extensions are Python/C++ packages that add functionality.

### Key Isaac Sim Extensions

| Extension | Purpose |
|-----------|---------|
| `isaacsim.core.api` | Core robotics API wrappers |
| `isaacsim.core.experimental` | New-generation core API (recommended) |
| `isaacsim.core.prims` | Prim wrappers (Articulation, RigidPrim, etc.) |
| `isaacsim.asset.importer.urdf` | URDF robot import |
| `isaacsim.asset.importer.mjcf` | MJCF (MuJoCo) import |
| `isaacsim.robot.motion_generation` | Motion planning (RMPflow, Lula RRT) |
| `isaacsim.sensors` | Physics-based sensors |
| `omni.replicator.core` | Synthetic data generation |
| `omni.isaac.ros2_bridge` | ROS 2 integration |

### Creating Custom Extensions

- Use the **Extension Template Generator** for scaffolding
- Extensions live in the Omniverse extension search paths
- Hot-reloadable during development in the GUI

---

## 11. Common Workflows

### Loading a USD Scene

```python
import omni.usd
omni.usd.get_context().open_stage("path/to/scene.usd")
```

### Stepping Simulation (Standalone, Isaac Sim API)

```python
import isaacsim
from isaacsim.simulation_app import SimulationApp
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
simulation_app = SimulationApp({"headless": True})

# Setup scene...

# Simulation loop
while simulation_app.is_running():
    simulation_app.update()  # Step physics + render

simulation_app.close()
```

### Headless Mode

```bash
# Pip-installed Isaac Sim
isaacsim --headless

# Standalone script
python my_script.py --headless

# Container
./runheadless.sh -v
```

In Python:
```python
simulation_app = SimulationApp({"headless": True})
```

### Running from Terminal (Workstation Install)

```bash
# Linux
cd ~/isaacsim
./python.sh standalone_examples/tutorials/getting_started.py

# Windows
cd C:\isaacsim
python.bat standalone_examples\tutorials\getting_started.py
```

### Development Tools

- **Script Editor:** Built into Isaac Sim GUI for interactive Python
- **VS Code:** Debugging support with Isaac Sim
- **Jupyter Notebooks:** Interactive development with `SimulationApp`
- **Carb Settings:** Runtime configuration via `carb.settings`

### Reset/Cache Issues

```bash
# Reset user settings
./isaac-sim.sh --reset-user

# Clear shader/physics caches
./clear_caches.sh    # Linux
clear_caches.bat     # Windows
```

---

## 12. Isaac Lab (Formerly Orbit)

### Overview

Isaac Lab is a **GPU-accelerated framework for robot learning** built on Isaac Sim. It provides a unified interface for RL, imitation learning, and motion planning research.

- **GitHub:** https://github.com/isaac-sim/IsaacLab
- **Docs:** https://isaac-sim.github.io/IsaacLab/main/index.html
- **Current version:** v2.3.2
- **Requires:** Isaac Sim 5.1.0 (stable) | Python 3.11
- **License:** BSD-3-Clause

### Installation

```bash
# Recommended: pip Isaac Sim + source Isaac Lab
pip install isaacsim[all]==5.1.0 --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install  # Linux
# isaaclab.bat --install  # Windows
```

### Core Architecture

Isaac Lab provides two task design workflows:

#### Manager-Based Workflow (Modular)

Environment decomposed into component managers:

```python
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
```

#### Direct Workflow (Monolithic)

Single class implementing entire environment -- closer to IsaacGymEnvs pattern:

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg  # Multi-agent
```

### Complete Manager-Based RL Environment Example

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class EventCfg:
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

### Running an Isaac Lab Environment

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = CartpoleEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)

count = 0
while simulation_app.is_running():
    with torch.inference_mode():
        if count % 300 == 0:
            env.reset()
        actions = torch.randn_like(env.action_manager.action)
        obs, reward, terminated, truncated, info = env.step(actions)
        count += 1

env.close()
```

### Isaac Lab Simulation Setup

```python
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationCfg, SimulationContext
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
sim.reset()

while simulation_app.is_running():
    sim.step()

simulation_app.close()
```

### Spawning Prims in Isaac Lab

```python
import isaaclab.sim as sim_utils

# Ground plane
sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

# Light
cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

# Xform container
sim_utils.create_prim("/World/Objects", "Xform")

# Visual cone
cfg = sim_utils.ConeCfg(
    radius=0.15, height=0.5,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
)
cfg.func("/World/Objects/Cone", cfg, translation=(-1.0, 1.0, 1.0))

# Rigid body cone (with physics)
cfg_rigid = sim_utils.ConeCfg(
    radius=0.15, height=0.5,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
)
cfg_rigid.func("/World/Objects/ConeRigid", cfg_rigid, translation=(-0.2, 0.0, 2.0))

# USD file reference
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
cfg_usd = sim_utils.UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
)
cfg_usd.func("/World/Objects/Table", cfg_usd, translation=(0.0, 0.0, 1.05))
```

### Articulation Robot in Isaac Lab

```python
from isaaclab.assets import Articulation
from isaaclab_assets import CARTPOLE_CFG
import torch

# Configure
cartpole_cfg = CARTPOLE_CFG.copy()
cartpole_cfg.prim_path = "/World/Origin.*/Robot"
cartpole = Articulation(cfg=cartpole_cfg)

# After sim.reset():
# Read state
root_state = cartpole.data.default_root_state.clone()
joint_pos = cartpole.data.default_joint_pos.clone()
joint_vel = cartpole.data.default_joint_vel.clone()

# Set state
cartpole.write_root_pose_to_sim(root_state[:, :7])
cartpole.write_root_velocity_to_sim(root_state[:, 7:])
joint_pos += torch.rand_like(joint_pos) * 0.1
cartpole.write_joint_state_to_sim(joint_pos, joint_vel)
cartpole.reset()

# Control
efforts = torch.randn_like(cartpole.data.joint_pos) * 5.0
cartpole.set_joint_effort_target(efforts)
cartpole.write_data_to_sim()

# In sim loop
sim.step()
cartpole.update(sim_dt)
```

---

## 13. Key APIs and Their Import Paths

### Isaac Sim Core Imports

```python
# Simulation app
from isaacsim.simulation_app import SimulationApp

# Core Experimental API (recommended)
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, Cube, DistantLight
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.prims import XformPrim, RigidPrim, GeomPrim

# Core API (legacy)
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import SingleArticulation, Articulation
from isaacsim.core.utils.types import ArticulationAction

# Robot importers
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

# USD (from pxr)
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, PhysxSchema

# Omniverse utilities
import omni.usd
import omni.replicator.core as rep
```

### Isaac Lab Core Imports

```python
# App launcher
from isaaclab.app import AppLauncher

# Simulation
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils

# Assets
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject

# Scene
from isaaclab.scene import InteractiveSceneCfg

# Environments
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
import isaaclab.envs.mdp as mdp

# Managers
from isaaclab.managers import (
    RewardTermCfg,
    TerminationTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)

# Sensors
from isaaclab.sensors import Camera, ContactSensor, FrameTransformer, Imu, RayCaster

# Actuators
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, ActuatorNetMLPCfg

# Controllers
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

# Terrain
from isaaclab.terrains import TerrainImporterCfg

# Utilities
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# RL library wrappers
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.skrl import SkrlOnPolicyRunnerCfg
from isaaclab_rl.rl_games import RlGamesPpoRunnerCfg
from isaaclab_rl.sb3 import Sb3OnPolicyRunnerCfg

# Pre-configured robot assets
from isaaclab_assets import CARTPOLE_CFG, FRANKA_PANDA_CFG, ANYMALx_CFG
```

### Isaac Lab API Module Map

| Module | Description |
|--------|-------------|
| `isaaclab.app` | Application launcher and lifecycle |
| `isaaclab.actuators` | Actuator models (implicit, DC motor, network-based) |
| `isaaclab.assets` | Rigid objects and articulated robots |
| `isaaclab.controllers` | Motion controllers and IK solvers |
| `isaaclab.devices` | Teleoperation device interfaces |
| `isaaclab.envs` | Environment definitions (Manager-based, Direct) |
| `isaaclab.envs.mdp` | MDP components (rewards, observations, actions, terminations) |
| `isaaclab.managers` | Environment component managers |
| `isaaclab.markers` | UI visualization markers |
| `isaaclab.scene` | Interactive scene definition |
| `isaaclab.sensors` | Sensor implementations (Camera, IMU, Contact, RayCaster) |
| `isaaclab.sim` | Simulation context, spawners, schemas, converters |
| `isaaclab.sim.spawners` | Object spawners (shapes, lights, USD files) |
| `isaaclab.sim.schemas` | Physics schema helpers |
| `isaaclab.sim.converters` | Format converters (URDF, MJCF to USD) |
| `isaaclab.terrains` | Procedural terrain generation |
| `isaaclab.utils` | Helpers, math, IO, configuration |

---

## 14. Official Documentation URLs and Resources

### Primary Documentation

| Resource | URL |
|----------|-----|
| **Isaac Sim Docs** | https://docs.isaacsim.omniverse.nvidia.com/latest/index.html |
| **Isaac Lab Docs** | https://isaac-sim.github.io/IsaacLab/main/index.html |
| **Isaac Sim GitHub** | https://github.com/isaac-sim/IsaacSim |
| **Isaac Lab GitHub** | https://github.com/isaac-sim/IsaacLab |
| **NVIDIA Developer Portal** | https://developer.nvidia.com/isaac/sim |
| **NGC Container** | https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim |
| **PyPI Package** | https://pypi.org/project/isaacsim/ |

### Isaac Sim Documentation Sections

| Section | Path |
|---------|------|
| Installation | `/latest/installation/index.html` |
| System Requirements | `/latest/installation/requirements.html` |
| Pip Install | `/latest/installation/install_python.html` |
| Container Install | `/latest/installation/install_container.html` |
| Quick Tutorials | `/latest/introduction/quickstart_index.html` |
| Basic Usage Tutorial | `/latest/introduction/quickstart_isaacsim.html` |
| Python Scripting | `/latest/python_scripting/index.html` |
| Core API Overview | `/latest/python_scripting/core_api_overview.html` |
| Robot Simulation | `/latest/robot_simulation/index.html` |
| URDF Import | `/latest/importer_exporter/import_urdf.html` |
| Sensors | `/latest/sensors/index.html` |
| ROS 2 Tutorials | `/latest/ros2_tutorials/index.html` |
| Replicator / SDG | `/latest/replicator_tutorials/index.html` |
| Domain Randomization | `/latest/replicator_tutorials/tutorial_replicator_isaac_randomizers.html` |
| OmniGraph | `/latest/omnigraph/index.html` |
| Articulation Controller | `/latest/robot_simulation/articulation_controller.html` |

All paths are relative to `https://docs.isaacsim.omniverse.nvidia.com`.

### Isaac Lab Documentation Sections

| Section | Path |
|---------|------|
| Ecosystem Overview | `/main/source/setup/ecosystem.html` |
| Installation | `/main/source/setup/installation/index.html` |
| Core Concepts | `/main/source/overview/core-concepts/index.html` |
| Task Workflows | `/main/source/overview/core-concepts/task_workflows.html` |
| RL Frameworks | `/main/source/overview/reinforcement-learning/rl_frameworks.html` |
| API Reference | `/main/source/api/index.html` |
| Tutorials: Empty Scene | `/main/source/tutorials/00_sim/create_empty.html` |
| Tutorials: Spawn Prims | `/main/source/tutorials/00_sim/spawn_prims.html` |
| Tutorials: Articulation | `/main/source/tutorials/01_assets/run_articulation.html` |
| Tutorials: RL Env | `/main/source/tutorials/03_envs/create_manager_rl_env.html` |

All paths are relative to `https://isaac-sim.github.io/IsaacLab`.

### Community & Support

| Resource | URL |
|----------|-----|
| **NVIDIA Omniverse Discord** | https://discord.gg/nvidiaomniverse |
| **Isaac Sim Forum** | https://forums.developer.nvidia.com/c/isaac-sim/ |
| **Isaac Lab GitHub Discussions** | https://github.com/isaac-sim/IsaacLab/discussions |
| **Isaac Lab Paper** | arXiv:2511.04831 |

---

## Quick Reference: Minimal Working Examples

### Isaac Sim -- Hello World (Standalone)

```python
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import isaacsim
from isaacsim.simulation_app import SimulationApp

app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, Cube

stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane")
Cube(paths="/World/MyCube", positions=[0, 0, 1.0], sizes=0.5)

while app.is_running():
    app.update()

app.close()
```

### Isaac Lab -- Empty Simulation

```python
import argparse
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationCfg, SimulationContext

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sim = SimulationContext(SimulationCfg(dt=0.01))
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
sim.reset()

while simulation_app.is_running():
    sim.step()

simulation_app.close()
```

### Isaac Lab -- RL Training Loop

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv

env = ManagerBasedRLEnv(cfg=MyEnvCfg())
obs, info = env.reset()

for step in range(10000):
    with torch.inference_mode():
        actions = torch.randn(env.num_envs, env.action_space.shape[-1], device=env.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if terminated.any() or truncated.any():
            # Environments auto-reset in Isaac Lab
            pass

env.close()
```
