# Cracked Glass Textures

PNG textures that drive the OmniGlass cracked-glass materials in
`sim_utils/glass_anomaly.py`. Used by `simulations/environment/cracked_glass_preview.py`,
`simulations/environment/build_cracked_scene_usd.py`, and the dataset
collection pipeline described in `DATASET_PLAN.md`.

## Three canonical variants

These three types cover the damage archetypes public safety flagged as the
ones seen on real MSOE campus glass. Only these three are used in production
renders; everything else has been moved to `archive/`.

| Variant | Damage type | Real-world analogue | Generator |
|---|---|---|---|
| `edge_v1` | **Stress crack** — single diagonal fracture from a panel edge, light branching, tapering tip | Thermal / installation stress on curtain-wall glazing | `edge_crack.py` (biased random walk with edge-corner bias) |
| `v4` | **Impact spider-web** — dense radial fracture from an impact point | Object strike (ball, thrown projectile) on single-pane | `generate_textures_diffusion.py` (SDXL diffusion + Sobel) |
| `tempered_v5` | **Safety-glass dice** — dense Voronoi crackle across the full panel | Tempered / safety glass after catastrophic failure | `tempered_voronoi.py` (Voronoi tessellation with 3×3 seed tiling for seamless UV) |

## Required files per variant

Each variant ships four PNG files at 2048 × 2048:

| File | Role |
|---|---|
| `crack_opacity_<v>.png` | Grayscale mask: **black** on cracks, **white** on intact glass. Source of truth; the other three are derived from this |
| `crack_normal_<v>.png` | Tangent-space normal map derived from the opacity via Sobel gradients (`opacity_to_normal` in `generate_textures.py`). DirectX orientation, `flip_tangent_v = True` in the material |
| `crack_roughness_<v>.png` | Inverse of opacity — **white** on cracks, **black** on intact. Drives OmniGlass `roughness_texture` so cracks scatter light (frosted look) while intact stays mirror-clear |
| `crack_tint_<v>.png` | Remap of opacity: cracks raised from 0 → 180 (≈70% transmission), intact stays 255. Drives OmniGlass `glass_color_texture` so cracks look like dark frosted smudges instead of pitch-black holes. Auto-generated at runtime by `glass_anomaly.py` if missing |

## Material pipeline

`sim_utils/glass_anomaly.py::create_cracked_glass_material` writes the
following OmniGlass inputs:

- `normal_map_texture` ← `crack_normal_<v>.png` (`flip_tangent_v=True`)
- `roughness_texture` ← `crack_roughness_<v>.png` (`roughness_texture_influence=1.0`)
- `glass_color_texture` ← `crack_tint_<v>.png` (auto-baked from opacity)
- `frosting_roughness = 0.03` (intact glass = near-mirror, cracks fully frosted by the texture override)
- `glass_ior = 1.491`, `thin_walled = True`, `glass_color = (0.82, 0.90, 0.94)`

This combination was empirically validated to make cracks visibly readable
under camera-headlight lighting. Validation across the six Tanner lighting
presets is the next open question before running the dataset collection —
see `DATASET_PLAN.md`.

## Regenerating textures

```bash
# Edge crack (stress fracture from a corner, 1 root, seed 42)
python3 edge_crack.py --variant edge_v1 --n-roots 1 --seed 42

# Tempered safety-glass (Voronoi, 500 seeds, 6px line width)
python3 tempered_voronoi.py --suffix _v5 --n-seeds 500 --line-width 6

# Impact spider-web — re-runs SDXL diffusion, slow (~5 min on T4):
python3 generate_textures_diffusion.py --variant v4
```

Each script writes all four `crack_{opacity,normal,roughness,tint}_<v>.png`
files. The `tint` is regenerated on-demand inside the material helper if
missing, so you never need to commit it explicitly.

## Archive

`archive/legacy_textures/` — retired variants from the iteration phase
(edge_v2/v3, hairline/hairline_{100,555,888}, impact, tempered_v1..v7
except v5, thermal, stress, sparse, s100/s2025/s9999, v1/v3, legacy
`crack_raw_*` diffusion intermediates, abandoned `crack_emissive_*`
from the self-illuminated-overlay experiment). Keep for reference and
A/B regressions; not used in any active pipeline.

`archive/legacy_generators/` — `generate_crack_types.py`,
`refine_crack_img2img.py`, `refine_crack_photos_batch.py`,
`tempered_from_photo.py`. Superseded by the three canonical generators.

`archive/debug/` — one-off debug PNGs from the Voronoi and edge-crack
tuning passes.

## Files in this directory

```
cracked_glass/
├── crack_opacity_edge_v1.png      edge_v1 source-of-truth
├── crack_opacity_tempered_v5.png
├── crack_opacity_v4.png
├── crack_normal_*.png             (3 files, one per variant)
├── crack_roughness_*.png          (3 files, one per variant)
├── crack_tint_*.png               (3 files — auto-baked, committed for convenience)
├── edge_crack.py                  generator for edge_v1
├── tempered_voronoi.py            generator for tempered_v5
├── generate_textures_diffusion.py generator for v4 (SDXL)
├── generate_textures.py           shared helpers (opacity_to_normal, etc.)
├── README.md                      this file
├── DATASET_PLAN.md                paused plan — dataset pipeline design
└── archive/                       retired iteration artefacts
```
