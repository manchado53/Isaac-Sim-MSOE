# IRA (Isaac Sim Replicator Agent) configs — "W3" fallback

These are **fallback scaffolding** for the MSOE people-in-building demo. They
invoke NVIDIA's full headless IRA pipeline (`sdg_scheduler.py` +
`isaacsim.exp.action_and_event_data_generation.base.kit`) against the
Innovation Center physics USD. Only reach for this path if the simpler
`simulations/environment/people_walking_anim_demo.py` (W2.5 — direct use of
`omni.anim.*` extensions) cannot produce a walking, non–T-posed character.

## Files

| File | Purpose |
|------|---------|
| `ground_floor_config.yaml` | IRA config. Points at `MSOEBuildingV4_physics.usd`, spawns 1 character, 1 camera, 450-frame sim (15 s @ 30 fps). |
| `ground_floor_commands.txt` | Character command file. `Character_0` walks a 3-waypoint path around the Event Floor (IC 100) with short idle pauses between legs. |
| `ground_floor_cameras.json` | Single camera placement matching `TEST_CAMERAS['G']` from `sim_utils/test_cameras.py` (top corner of event space, 12 mm wide-angle). |
| `../../simulations/environment/run_ira_ground_floor.sh` | Wrapper that invokes `~/isaacsim/tools/actor_sdg/sdg_scheduler.py` with the right flags. |
| `../../submit_ira_job.sh` | SLURM submission wrapper (partition=teaching, 1 GPU, 1 h, 32 GB). |

## How to launch

### Interactive (on a GPU node)

```bash
srun --partition=teaching --gres=gpu:1 --time=1:00:00 --pty bash
cd ~/isaacsim/isaac-sim-msoe
chmod +x simulations/environment/run_ira_ground_floor.sh   # once
./simulations/environment/run_ira_ground_floor.sh
```

### SLURM batch

```bash
cd ~/isaacsim/isaac-sim-msoe
chmod +x submit_ira_job.sh simulations/environment/run_ira_ground_floor.sh   # once
sbatch submit_ira_job.sh
squeue -u $USER
tail -f logs/ira_<JOBID>.out
```

Optional `sdg_scheduler.py` flags (pass through after the script name):
- `--save_usd` — save the fully-populated scene to `scene.usd` in the output dir
- `--no_random_commands` — honor `ground_floor_commands.txt` verbatim (already on by default in the launcher)
- `--debug_print` — enable IRA's debug log channel (already on)

## What to expect

If everything works:
1. IRA opens `MSOEBuildingV4_physics.usd` via `StageUtil.open_stage(...)`.
2. It **bakes a NavMesh** across the whole stage (this is the step most likely to fail — see "Blockers" below).
3. Spawns `Character_0` from the S3 character library (random pick from `Assets/Isaac/5.1/Isaac/People/Characters/`).
4. Places one camera at `(-42.15, 68.44, 16.44)` in stage feet.
5. Runs `ground_floor_commands.txt` — Idle → GoTo → Idle → GoTo → Idle → GoTo → Idle (~15 s total).
6. `IRABasicWriter` writes RGB frames + camera params to `~/ReplicatorResult/<timestamp>/`.

You will get **PNGs, not a video**. Stitching them into MP4 is a follow-up (ffmpeg
from `imageio-ffmpeg`, same pattern as `sim_utils/video_capture.py`).

## Known issues / blockers

### 1. NavMesh baking is REQUIRED — this is the biggest risk

Found in `extscache/isaacsim.replicator.agent.core-.../isaacsim/replicator/agent/core/simulation.py` around line 603:

```python
async def try_bake_navmesh():
    _inav = nav.acquire_interface()
    _inav.start_navmesh_baking_and_wait()
    navmesh = _inav.get_navmesh()
    if navmesh is None:
        carb.log_error(
            "NavMesh building failed. Please check whether the stage has a valid NavmeshVolume. "
            "Will not load assets to scene."
        )
        return   # <-- character & camera loading is skipped entirely
```

**There is no "disable NavMesh" flag.** If the building stage lacks a
`NavmeshVolume` prim covering the ground floor, IRA will abort asset loading
and produce no frames. The code explicitly swallows the failure and exits the
coroutine; the only signal is the `carb.log_error` in the log.

If that happens, mitigations are (in order of effort):
- **Add a NavmeshVolume** to the building USD: open `MSOEBuildingV4_physics.usd`
  in Isaac Sim UI, `Create → Navigation → NavMesh Volume`, size it to cover the
  ground floor, save. (Requires manual action, not headless-friendly.)
- **Pre-bake a NavMesh** programmatically using `omni.anim.navigation.core` API
  before invoking IRA, and reference the NavMesh prim in the scene.
- **Lower `agent_radius`** in nav settings so it has a chance on sparse geometry.
- Confirm the navmesh also covers a valid Z range — Revit feet plus stacked
  floors may confuse the Recast/Meshtools bake.

The existing physics pipeline (`setup_building_physics.py`) does **not** add a
NavmeshVolume. This is a prerequisite we have not yet tackled.

### 2. Stage-loading uses `StageUtil.open_stage(...)` under the hood

Our `feedback_isaac_sim_pitfalls` memory notes that `omni.usd.get_context().open_stage()`
can produce black renders for Revit USDs. IRA uses the same pattern. It may
work for `MSOEBuildingV4_physics.usd` because the physics pipeline already
cleaned up references, but verify the frames aren't all-black. If they are,
the workaround (reference-add into a fresh stage) is incompatible with IRA's
`load_scene_from_config_file` and would require patching the IRA extension.

### 3. Feet vs meters

- Building USD: `metersPerUnit = 0.3048` (Revit feet).
- Character USDs: meters.
- Camera coords in `ground_floor_cameras.json`: feet (stage units).
- GoTo waypoints in `ground_floor_commands.txt`: feet (stage units).

IRA respects `metersPerUnit` when instancing character references, so the
character should scale correctly. However the character's `walkSpeed` metadata
is in m/s and may look slow or fast relative to the feet-scale building.
Verify visually; if off, patch via `/persistent/exts/isaacsim.replicator.agent/
character_goto_*_distance` carb settings.

### 4. Auto-aimed cameras override our yaw/pitch

`sdg_scheduler.py:132` sets
`/persistent/exts/isaacsim.replicator.agent/aim_cameras_at_characters = true`
unconditionally. Our yaw/pitch in `ground_floor_cameras.json` will be
**overwritten at runtime** so the camera tracks `Character_0`. The (x, y, height)
is preserved. If you need a locked pose, patch `sdg_scheduler.py` or toggle
the setting before the run (would require a one-line patch — we are not
modifying upstream code here).

### 5. Output is PNG sequence, not video

The `IRABasicWriter` writes annotator frames per-camera. Convert to MP4 with:
```bash
ffmpeg -framerate 30 -i "$HOME/ReplicatorResult/<run>/Camera_0/rgb/rgb_%04d.png" \
       -c:v libx264 -pix_fmt yuv420p videos/ira_ground_floor.mp4
```

## Summary: which of the listed IRA keys matter

Confirmed from `data/config_file/config_file_default_values.yaml` and
`isaacsim/replicator/agent/core/simulation.py`:

| Key | Required | Notes |
|-----|----------|-------|
| `global.seed` | optional (auto-generated) | Int; pinned to 42 here for determinism. |
| `global.simulation_length` | yes | Frames, divided by 30 fps → duration. |
| `sensor.camera_num` | yes | Int. IRA spawns N cameras under `/World/Cameras`. |
| `sensor.lidar_num` | optional | Keep 0. |
| `scene.asset_path` | **yes** | Absolute path. Loaded via `StageUtil.open_stage`. |
| `character.asset_path` | optional | Empty string falls back to Isaac Sim S3 root. |
| `character.command_file` | yes | Relative to config file dir. |
| `character.num` | yes | We use 1 for baby-step demo. |
| `character.filters` | optional | Limit which sub-folders to sample from. |
| `robot.*` | required block | Zeros out; IRA still needs the section defined. |
| `replicator.writer` | yes | `IRABasicWriter` is fine for RGB-only output. |
| `replicator.parameters` | yes | Dict; all-false toggles keep output small. |

Unknown/not required:
- `event` — omitted (no incidents).
- `response` — omitted.
- `character.spawn_area`, `robot.spawn_area` — auto-derived from NavMesh when absent.
