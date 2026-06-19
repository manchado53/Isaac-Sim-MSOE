# Cracked-glass synthetic data generation

All Python + shell code for the cracked-glass anomaly-detection pipeline
lives under this directory. The image assets themselves (PNGs) stay in
`assets/materials/cracked_glass/` because USD material prims reference
them by path — the separation is deliberate.

## Directory layout

```
SyntheticDataGen/cracked_glass/
├── README.md                           # this file
├── textures/                           # PNG generators
│   ├── generate_textures.py            # shared helpers (Sobel, normal-from-height, ...)
│   ├── edge_crack.py                   # generates edge_v1 (stress crack)
│   ├── tempered_voronoi.py             # generates tempered_v5 (safety-glass dice)
│   └── generate_textures_diffusion.py  # generates v4 (impact spider-web, SDXL)
│
├── glass_anomaly.py                    # material authoring: OmniGlass + texture binding
│
├── cracked_glass_preview.py            # single-panel preview renderer
├── build_cracked_scene_usd.py          # bakes a full cracked USD (all floor-4 panels)
├── cracked_glass_floor4_stream.py      # WebRTC stream of the cracked scene
├── _open_cracked_stage.py              # bootstrap that opens the USD on `isaac-sim.streaming.sh --exec`
├── crack_parameter_sweep.py            # parameter-sweep renders (seed/hits/lighting/bump/panel)
│
├── cracked_glass_dataset.py            # legacy Replicator dataset (BasicWriter — see DATASET_PLAN.md)
├── diffusion_augment.py                # SDXL img2img post-processing for intact frames
│
└── submit_cracked_glass_dataset.sh     # SLURM submission for the dataset run
```

## Three canonical crack types

See `assets/materials/cracked_glass/README.md` for the full story. Short
version:

| Variant | Damage type | Generator |
|---|---|---|
| `edge_v1` | Stress crack radiating from panel corner | `textures/edge_crack.py` |
| `v4` | Impact spider-web | `textures/generate_textures_diffusion.py` |
| `tempered_v5` | Safety-glass dice tessellation | `textures/tempered_voronoi.py` |

## Pipeline stages — from PNG to a rendered panel

1. **Texture generation** (`textures/`) → writes `crack_{opacity,normal,roughness}_<v>.png` into `assets/materials/cracked_glass/`.
2. **Material authoring** (`glass_anomaly.py::create_cracked_glass_materials`) → builds a `UsdShade.Material` prim (OmniGlass MDL) that binds those PNGs to `normal_map_texture` / `roughness_texture` / `glass_color_texture`. Tint PNG is auto-baked on first call.
3. **Panel selection** (`glass_anomaly.py::enumerate_glass_prims`) → walks the Innovation Center USD, filters to `STATIC_GLAZING` prims passing the GL_1/Curtain name filters.
4. **Binding** (`build_cracked_scene_usd.py::bind_visual_material_deep`) → walks each panel down to the inner Mesh, unbinds its pre-existing material, binds the chosen crack material.
5. **Rendering** — `cracked_glass_preview.py` (single-panel), `build_cracked_scene_usd.py` (bake + export USD), `cracked_glass_floor4_stream.py` (live stream), or the dataset runner.

## Running things from here

All scripts still expect to be run from the **project root** (`~/isaac-sim-msoe/`) because they use `./python.sh` (Isaac Sim's bundled interpreter at `~/isaacsim/python.sh`):

```bash
cd ~/isaac-sim-msoe

# Regenerate the three canonical crack PNGs
python3 SyntheticDataGen/cracked_glass/textures/edge_crack.py --variant edge_v1 --n-roots 1 --seed 42
python3 SyntheticDataGen/cracked_glass/textures/tempered_voronoi.py --suffix _v5 --n-seeds 500 --line-width 6
python3 SyntheticDataGen/cracked_glass/textures/generate_textures_diffusion.py --variant v4   # slow, SDXL

# Single-panel preview render
~/isaacsim/python.sh SyntheticDataGen/cracked_glass/cracked_glass_preview.py \
    --variants edge_v1,v4,tempered_v5 --n-panels 3

# Build the exported cracked USD (all floor-4 panels cracked)
~/isaacsim/python.sh SyntheticDataGen/cracked_glass/build_cracked_scene_usd.py

# Live WebRTC stream of the cracked scene
~/isaacsim/python.sh SyntheticDataGen/cracked_glass/cracked_glass_floor4_stream.py

# Legacy dataset run (replaced soon — see DATASET_PLAN.md)
sbatch SyntheticDataGen/cracked_glass/submit_cracked_glass_dataset.sh --frames 500 --fraction 0.2
```

## Planned dataset collector

`assets/materials/cracked_glass/DATASET_PLAN.md` describes the new
collection pipeline modelled on Tanner's `door_static_collection.py`:
7 lighting presets × 3 camera angles × 100 panels × (intact / 3 crack
variants). That will land as `cracked_glass_collection.py` alongside
the legacy `cracked_glass_dataset.py`. See the plan for the full design.

## Dependencies on the rest of the repo

| Where | Why |
|---|---|
| `assets/materials/cracked_glass/*.png` | Source-of-truth texture PNGs (referenced by the materials) |
| `assets/environments/InnovationCenter/` | Building USD + `building_info.json` (floor bounds) |
| `sim_utils/building_physics.py` | `setup_physics_scene`, `_find_mesh_descendant` |
| `sim_utils/building_info.py` | `get_floor` — floor Z-range helper |
| `sim_utils/test_cameras.py` | `CAMERA_FOCAL_LENGTH`, `add_interior_lighting`, `add_camera_headlight` |

All scripts here add the project root to `sys.path` at startup so the
`sim_utils.*` imports resolve. Don't `cd` into this directory to run a
script — paths will break.
