# Cracked Glass Textures

Authored PNG textures that drive the cracked-glass OmniPBR materials used by
`simulations/dataset/cracked_glass_dataset.py`.

## Expected files

Three variants, each consisting of a normal map and (optionally) an opacity mask:

| File | Purpose |
|------|---------|
| `crack_normal_v1.png` | Subtle radial crack (single impact point) — normal map |
| `crack_normal_v2.png` | Shattered / spider-web crack pattern — normal map |
| `crack_normal_v3.png` | Chipped edge / partial breakage — normal map |
| `crack_opacity_v1.png` | Matching alpha mask (white = intact, black = missing) |
| `crack_opacity_v2.png` | Matching alpha mask |
| `crack_opacity_v3.png` | Matching alpha mask |

All textures should be:

- **2K resolution** (`2048 x 2048`) — enough detail at TEST_CAMERA distances
  (~6–20 ft) without hammering VRAM.
- **Tileable** — glass prims vary in size; tiling avoids obvious seams.
- **DirectX normal orientation** (Y-up). The material sets
  `flip_tangent_v = True` to compensate for Omniverse's OpenGL default.

## Authoring notes

Quick paths we considered, in order of effort:

1. **CC0 crack pack** — several free sources (ambientCG, Poly Haven,
   OpenGameArt) publish tileable cracked-glass normal maps. Most direct
   route; check licensing (CC0 preferred).
2. **Hand-authored** — paint crack lines in grayscale, run a normal-map
   filter (GIMP's *Filters → Generic → Normal Map* or Krita's equivalent).
3. **Procedural** — Substance / Material Maker / Blender's shader nodes.
   Overkill for a first pass.

## Fallback behavior

If any file is missing, `sim_utils.glass_anomaly.create_cracked_glass_material`
still defines the OmniPBR material using its constant-opacity fallback. The
pipeline keeps running — cracked variants will just look tinted instead of
damaged — so the dataset script is safe to run before textures land.

## Source credits

- (add CC0 / CC-BY attribution here once textures are committed)
