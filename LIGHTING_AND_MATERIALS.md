# Lighting and Material Variation

Domain randomization for the door-state dataset. This doc explains **why** we
vary each axis, and **how** each is implemented in the Isaac Sim data-collection
scripts ([door_replicator_collection.py](simulations/robot/door_replicator_collection.py),
[c1_wall_material_test.py](simulations/robot/c1_wall_material_test.py)).

---

## All variation axes at a glance

| Axis | What changes | Technique | Applies to | Script |
|------|-------------|-----------|------------|--------|
| **Lighting** | Global illumination — color cast, contrast, shadow direction | 6 discrete indoor presets + per-cell jitter | All doors | `door_replicator_collection.py` |
| **Door material — baseline** | Door frame color (glass left as glass) | OmniPBR solid color on frame meshes only | All | `door_replicator_collection.py` |
| **Door material — fill_glass (A1)** | Door becomes a solid opaque panel | OmniPBR on frame + glass meshes | Curtain-wall & glass-panel doors | `door_replicator_collection.py` |
| **Door material — glass_variant (A3)** | Glass tint / frosting changes | OmniGlass on glass meshes only | Curtain-wall & glass-panel doors | `door_replicator_collection.py` |
| **Door material — textured (A4)** | Surface pattern on frame | Grayscale procedural texture + `diffuse_tint` via OmniPBR on frame | All | `door_replicator_collection.py` |
| **Door material — fill_textured (A1+A4)** | Solid-looking door with surface pattern | Tinted grayscale texture on frame + glass | Curtain-wall & glass-panel doors | `door_replicator_collection.py` |
| **Wall material (C1)** | Surrounding wall color changes | OmniPBR binding swap on nearby wall mesh prims | **Solid door families only** (`Type_F`, `Type_FG`, `Type_N`) | `c1_wall_material_test.py` |
| **Door angle** | Open/closed state | 4×4 xform matrix rewrite per mesh | All | `door_replicator_collection.py` |
| **Camera jitter** | Slight viewpoint variation | Per-episode random position + orientation offset | All | `door_replicator_collection.py` |

---

## Why we vary these axes at all

The fine-tuned Qwen2-VL model hit **96.3%** in-distribution accuracy but
**0%** on an unseen door (`Doors_13185454`). Error analysis traced this to the
model memorizing the specific visual appearance of the one door it was trained
on, rather than learning an abstract CLOSED-vs-OPEN concept. This is
consistent with the brittleness InspectVLM (2025) documented for VLMs on
industrial inspection tasks.

The fix is to force the model to see the same scene state (CLOSED / OPEN)
under enough visual conditions that the only signal correlated with the label
is the door pose itself — not pixel color, not specular highlight location,
not a particular texture. That means varying:

1. **Lighting** — the dominant source of global image statistics (color cast,
   contrast, where shadows fall). If every training frame has the same
   fluorescent daylight, the model treats that color distribution as part of
   the "CLOSED" fingerprint.
2. **Door material** — the dominant source of local door appearance. If every
   training frame has the same aluminum curtain-wall panel, the model
   associates "door" with that specific texture and fails on wood or painted
   doors.

These are the two axes the in-sim scene state can affect cheaply (no mesh
edits, no new assets). Camera pose is varied on a third axis (the arc grid
plus per-episode jitter) but that's a separate concern documented in the
collection scripts themselves.

---

## Lighting — six indoor presets

### Design rationale

We did not use HDRI swaps on the DomeLight. In the Kern Innovation Center's
enclosed interior corridors the ceiling is opaque and the DomeLight barely
contributes — an earlier attempt to add HDRI variety produced nearly
identical frames across presets. Instead, indoor lighting variety comes from
the **actual fixture configuration** inside the room.

We also removed an earlier "fill SphereLight" placed ~1.5m above the door.
It became a visible bright disc at the top of every frame plus a specular
hotspot on nearby red columns — a spurious visual feature that would
contaminate training. Job 242708's visual audit confirmed the artifact was
present in all 84 material × lighting cells. The fill's illumination is now
absorbed into boosted ceiling fixture intensities instead.

### The two light sources

Every preset drives the same two physical light sources, which are created
once at startup:

1. **One `UsdLux.DomeLight`** at `/World/DomeLight` — provides ambient fill.
   Each preset sets `dome_int` (intensity) and `dome_temp` (Kelvin).

2. **Four `UsdLux.RectLight` ceiling fixtures** at `/World/CeilingLight_0..3`
   — a 2×2 grid of 0.6m-square rectangles placed ~1.2m above the camera base
   Z, rotated 180° around X to face downward. Each slot can be toggled on or
   off independently per preset. The 2×2 layout sits above the camera's
   vertical FOV, so the fixtures illuminate the scene without appearing in
   frame.

### The six presets

| Preset               | Dome (int / K)  | Ceiling pattern | Ceiling (int / K) | Real-world analogue                         |
|----------------------|-----------------|-----------------|-------------------|---------------------------------------------|
| `bright_fluorescent` | 15,000 / 5500   | `(1,1,1,1)`     | 400,000 / 5000    | Standard daytime hallway, all fixtures on   |
| `daylight_flood`     | 50,000 / 6500   | `(1,0,1,0)`     | 230,000 / 6500    | Window-lit, cool, half ceiling fixtures off |
| `warm_evening`       |  8,000 / 3000   | `(1,1,1,1)`     | 200,000 / 2700    | After-hours tungsten, warm                  |
| `dim_night`          |  2,000 / 4000   | `(1,0,1,0)`     | 150,000 / 3500    | Overnight security lighting                 |
| `overcast_flat`      | 25,000 / 5000   | `(1,1,1,1)`     | 220,000 / 5000    | Even diffuse light, all ceilings dim        |
| `high_contrast`      |  2,000 / 5000   | `(0,1,0,0)`     | 700,000 / 5500    | One bright spotlight, dome near-dark        |

Presets are deliberately picked to span visually distinct regimes rather than
clustering around a single mean. `high_contrast` and `bright_fluorescent` are
designed to look obviously different to the eye, not marginally different.
This is closer to stratified domain variation than Gaussian noise
augmentation — the model sees six clearly separated points in lighting space,
and each point is individually repeated across every material and angle.

### Per-cell jitter

To prevent pixel-level memorization within a preset, a small jitter is added
each time the preset is applied:

- **±10%** on dome and ceiling intensity
- **±200K** on dome and ceiling color temperature

This is small enough that `dim_night` still looks dim and `bright_fluorescent`
still looks bright, but large enough that two frames in the same cell aren't
byte-identical. See `apply_lighting_condition()` in
[door_replicator_collection.py](simulations/robot/door_replicator_collection.py).

### Color temperature → RGB

The Kelvin values are converted to RGB via Tanner Helland's blackbody
approximation (`_color_temp_to_rgb()` in
[door_replicator_collection.py](simulations/robot/door_replicator_collection.py)),
then written to the light's `color` attribute. This is approximate but fast
and deterministic — good enough for the low-dimensional lighting axis.

---

## Materials — five techniques, pre-created materials, per-cell binding swap

Material variation is organised as **techniques**. Each technique targets a
specific subset of the door meshes (frame only, glass only, or both) with a
specific kind of shader (OmniPBR solid colour, OmniPBR textured, or
OmniGlass). All material prims are pre-created at startup; the per-cell
loop only flips `MaterialBindingAPI` relationships — it never mutates
shader inputs.

### Design rationale

We need the door to look like it could be any real door: aluminum curtain
wall, painted wood, fire-rated steel, tinted glass, textured panel, etc.
The cheapest way to do this without replacing meshes is to swap the bound
material on the existing door geometry.

Two failure modes ruled this out as trivial at first:

1. **OmniPBR shader cache.** Mutating a shared shader's
   `diffuse_color_constant` between renders does **not** invalidate OmniPBR's
   material cache in the RTX delegate. The first-set color sticks regardless
   of later writes. Jobs 242579/242581/242625 verified this experimentally.
   Lesson: never try to "reuse one shader and change its inputs."

2. **OmniPBR `CreateIdAttr` is not enough.** Calling
   `shader.CreateIdAttr("OmniPBR")` does **not** resolve to the real
   OmniPBR MDL module — the delegate substitutes a red debug material. The
   canonical binding path uses `SetSourceAsset("OmniPBR.mdl", "mdl")` +
   `SetSourceAssetSubIdentifier("OmniPBR", "mdl")` plus explicit `mdl`
   surface/displacement/volume output connections.

So the working pattern is: **pre-create one material prim per preset at
startup**, then per-cell do a binding swap — no shader mutation ever.

### Why a dispatcher, not a flat preset list

The first implementation only swapped OmniPBR materials on non-glass
meshes. Expanding beyond "recolour the frame" requires binding different
materials to different mesh subsets: A1 paints the glass *and* the frame,
A3 leaves the frame alone and swaps glass to OmniGlass, A4 uses a
procedural texture instead of a solid colour, and A1+A4 does both. The
clean way to express this is a `technique` field on every preset, read by
a dispatcher function that decides:

1. Which shader to pre-create (`OmniPBR` vs `OmniGlass`, with or without a
   `diffuse_texture`) — handled by `_make_omnipbr_material()` and
   `_make_omniglass_material()`.
2. Which mesh subset to bind it to — handled by
   `_bind_door_meshes(mat, include_frame, include_glass)`.

The dispatcher itself is `apply_door_material(spec)`.

### The five techniques

| Technique       | Shader     | Frame bound? | Glass bound? | What it tests                                               |
|-----------------|------------|--------------|--------------|-------------------------------------------------------------|
| `baseline`      | OmniPBR    | yes          | no           | Original behaviour: frame recolour, glass stays glass       |
| `fill_glass`    | OmniPBR    | yes          | **yes**      | Curtain-wall → solid opaque panel (A1)                       |
| `glass_variant` | OmniGlass  | no           | **yes**      | Glass colour / frosting variety, frame original (A3)        |
| `textured`      | OmniPBR    | yes          | no           | Procedural pattern on frame only (A4)                        |
| `fill_textured` | OmniPBR    | yes          | **yes**      | A1 + A4 combined: tinted pattern on both sides               |

`baseline` and `glass_variant` exist for completeness, but the
generalisation-interesting techniques for scaling up into training data
are **`fill_glass`**, **`textured`**, and **`fill_textured`** — visual
inspection of the 24-frame technique grid showed these produce the
cleanest appearance variation on the `Doors_13606713` curtain-wall door.

### Per-preset fields

Each entry in `DOOR_MATERIALS` carries:

- `name` — unique string, used as the USD prim path suffix and written to
  the NPZ `material_names` column
- `technique` — one of the five above, drives the dispatcher
- `roughness`, `metallic` — always present on OmniPBR presets
- `color` — Color3f, only for OmniPBR constant-colour presets
  (`baseline`, `fill_glass`)
- `texture` — pattern name (`brushed` / `grain` / `speckle`), only for
  `textured` and `fill_textured`
- `tint` — Color3f applied via `diffuse_tint`, only for textured variants
- `glass_color`, `ior`, `frosting`, `thin_walled` — OmniGlass inputs,
  only for `glass_variant`

### How the binding swap works

At startup, `_make_omnipbr_material()` and `_make_omniglass_material()`
each create one material prim per preset under `/World/Looks/DoorMat_*`.
For a **solid-colour OmniPBR** preset the shader inputs are:

```python
sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
sh.CreateInput("diffuse_color_constant",        ...).Set(Gf.Vec3f(*preset["color"]))
sh.CreateInput("reflection_roughness_constant", ...).Set(preset["roughness"])
sh.CreateInput("metallic_constant",             ...).Set(preset["metallic"])
mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
# ... plus displacement and volume outputs
```

For a **textured OmniPBR** preset (techniques `textured` and
`fill_textured`) the `diffuse_color_constant` branch is replaced with:

```python
sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
sh.CreateInput("project_uvw",     Sdf.ValueTypeNames.Bool).Set(True)
sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
sh.CreateInput("texture_scale",   Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2.0, 2.0))
sh.CreateInput("diffuse_tint",    Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*preset["tint"]))
```

`project_uvw=True` is essential: Revit-exported meshes have no UV
coordinates, so without triplanar projection the texture lookup degenerates
to a single pixel. `world_or_object=False` keeps projection in object space,
so the pattern follows the door when it rotates instead of sliding through
the geometry.

At each grid cell `apply_door_material(spec)` does:

```python
_restore_original_door_materials(include_frame=True, include_glass=True)
mat = _door_preset_materials[spec["name"]]
_bind_door_meshes(mat, include_frame=<per-technique>,
                       include_glass=<per-technique>)
```

and `_bind_door_meshes` walks `Usd.PrimRange(door_prim)`, checks
`_is_glass_mesh(p)`, and calls

```python
UsdShade.MaterialBindingAPI.Apply(p)
UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=strongerThanDescendants)
```

on the subset the technique targets.

**The dispatcher always restores both frame and glass first** — otherwise
a `fill_glass` cell followed by a `baseline` cell would leave the glass
painted, because the baseline path only touches the frame. Restoration
is cheap (an unbind + rebind round trip per mesh) and makes every cell
independent of the previous one.

This is a **binding relationship swap**, not a shader edit. The delegate
sees an entirely different material prim and picks up a different shader.
Cost per cell is O(meshes under the door) USD writes — no shader compile,
no material rebuild.

`strongerThanDescendants` is necessary because Revit exports frequently
bind materials at a nested scope, and without the stronger binding
strength your override gets out-prioritized by the original binding.

### Grayscale patterns + `diffuse_tint` — separating colour from pattern

For the `textured` and `fill_textured` techniques we want to explore
colour × pattern independently: "same red as the A1 fill, but with a
brushed surface", or "the same brushed pattern, tinted red vs white vs
near-black". OmniPBR supports this natively. It computes (see
`OmniPBRBase.mdl:386`):

```
tinted_diffuse_color = diffuse_reflection_color * diffuse_tint
```

where `diffuse_reflection_color` is sampled from `diffuse_texture` (or
falls back to `diffuse_color_constant` when no texture is bound), and
`diffuse_tint` multiplies on top. So if the texture is **grayscale and
bright** (mean ~0.85), the final surface colour is effectively the tint
value — with the texture's high-frequency variation carried on top.

`_generate_procedural_textures()` writes three 512×512 grayscale PNGs to
`data/<output_name>_textures/` at startup using NumPy + PIL:

| Pattern   | Generation (per-pixel intensity in [0, 1])                                    |
|-----------|-------------------------------------------------------------------------------|
| `brushed` | `0.85 + horizontal-row noise (±0.08) + per-pixel noise (±0.015)`              |
| `grain`   | `0.82 + 0.12·(sin(0.04·x) + sin(0.11·x + 1.7))/2 + per-pixel noise (±0.04)`   |
| `speckle` | `0.85 + per-pixel uniform noise (±0.05)`                                      |

Each pattern is written as a grayscale RGB (R=G=B) PNG so it round-trips
through the OmniPBR texture pipeline cleanly. The per-preset `tint` field
then drives the colour.

### The twelve presets in the current inspection grid

| Preset                   | Technique       | Pattern   | Colour / tint      | Notes                              |
|--------------------------|-----------------|-----------|--------------------|------------------------------------|
| `fill_red`               | `fill_glass`    | —         | (0.65, 0.10, 0.08) | A1 reference, flat red             |
| `fill_white`             | `fill_glass`    | —         | (0.90, 0.90, 0.88) | A1 reference, flat white           |
| `fill_wood`              | `fill_glass`    | —         | (0.55, 0.35, 0.20) | A1 reference, flat warm brown      |
| `tex_brushed`            | `textured`      | brushed   | (1.00, 1.00, 1.00) | A4 reference, frame only, no tint  |
| `tex_grain`              | `textured`      | grain     | (1.00, 1.00, 1.00) | A4 reference, frame only, no tint  |
| `tex_speckle`            | `textured`      | speckle   | (1.00, 1.00, 1.00) | A4 reference, frame only, no tint  |
| `fill_tex_red_brushed`   | `fill_textured` | brushed   | (0.80, 0.13, 0.10) | A1+A4 — red with brushed streaks   |
| `fill_tex_red_grain`     | `fill_textured` | grain     | (0.80, 0.13, 0.10) | A1+A4 — red with grain bands       |
| `fill_tex_white_brushed` | `fill_textured` | brushed   | (1.00, 1.00, 0.98) | A1+A4 — brushed-metal appearance   |
| `fill_tex_white_speckle` | `fill_textured` | speckle   | (1.00, 1.00, 0.98) | A1+A4 — painted-finish appearance  |
| `fill_tex_wood_grain`    | `fill_textured` | grain     | (0.70, 0.45, 0.25) | A1+A4 — wood-veneer appearance     |
| `fill_tex_dark_brushed`  | `fill_textured` | brushed   | (0.20, 0.20, 0.22) | A1+A4 — dark anodised appearance   |

The tint values for the `fill_textured` presets are deliberately ~20%
brighter than the corresponding A1 `color` values, because multiplying
by a grayscale texture (mean ~0.85) darkens the final surface. The
compensation gives visually comparable frames between e.g. `fill_red`
and `fill_tex_red_brushed` so the only apparent difference is the
surface pattern.

### OmniGlass variants (A3)

The `glass_variant` technique swaps the glass mesh binding to an
OmniGlass material instead of OmniPBR. `_make_omniglass_material()` sets:

```python
sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
sh.CreateInput("glass_color",        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*preset["glass_color"]))
sh.CreateInput("glass_ior",          Sdf.ValueTypeNames.Float  ).Set(preset["ior"])
sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float  ).Set(preset["frosting"])
sh.CreateInput("thin_walled",        Sdf.ValueTypeNames.Bool   ).Set(preset["thin_walled"])
```

`thin_walled=True` matches the geometry — curtain-wall glass panels are
single-sided planes, not solid volumes, and the thin-walled path avoids
the expensive two-sided refraction BSDF. `frosting_roughness` blurs the
glass: `glass_frosted` uses 0.8, the others 0.0. The three reference
presets were `glass_blue`, `glass_frosted`, `glass_bronze`; they are not
part of the current 12-preset inspection grid because inspection favored
the opaque techniques, but the code path still exists and can be
re-enabled by adding entries to `DOOR_MATERIALS`.

### Glass meshes — identification

`_is_glass_mesh()` filters children whose names start with `Glass_` or
contain `_GL_`, `glazing`, or `pane_`. These are the Revit export tags
for transparent glass panels. The dispatcher uses this predicate to
decide which subset of meshes a given technique targets — it is the
single source of truth for "is this mesh glass?".

### Restoring original bindings — split cache

The `baseline_original` no-op case requires restoring the Revit bindings
for both frame and glass. At startup we walk the door subtree once and
split the bindings into two caches:

```python
_original_frame_bindings = []
_original_glass_bindings = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh):
        continue
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    targets = list(rel.GetTargets()) if rel else []
    (_original_glass_bindings if _is_glass_mesh(p) else
     _original_frame_bindings).append((p, targets))
```

`_restore_bindings(cache)` does an unbind + rebind-targets round trip
per mesh. `_restore_original_door_materials(include_frame, include_glass)`
wraps it so the dispatcher can restore either half or both. Without this
split, a `fill_glass` cell would permanently corrupt the glass side
because the original-binding helper used to only know about the frame
meshes and couldn't put the glass back.

### Instanced doors need SetInstanceable(False)

Some Revit-exported doors (notably solid `Type_F_Door` variants and a subset
of curtain-wall exports) come through as **instanced prototype references**.
While instanced, their subtree is read-only — `MaterialBindingAPI.Bind()`
silently no-ops and `GetChildren()` walks won't yield the real meshes.

The fix is a single call at startup:

```python
if door_prim.IsInstance() or door_prim.IsInstanceable():
    door_prim.SetInstanceable(False)
```

This concretizes the subtree for the session so both the rotation-discovery
loop and the material override can see and mutate the actual mesh prims. No-op
on already-concrete doors.

---

## Wall material (C1) — surrounding wall color swap

### What it is and why

C1 changes the material of the wall surfaces that visually surround the door
in frame. The model currently always sees the same wall color/texture behind
and beside every door — if that wall is correlated with a specific door ID,
the model can use it as a spurious feature. Swapping the wall material breaks
that correlation without touching the door itself.

This is distinct from all the A-series techniques: C1 leaves the door at its
Revit-original appearance and only changes the background context.

### Which door families support C1

**C1 only works on solid door families** where the surrounding geometry lives
in the USD `Walls` category:

| Family | C1 possible? | Reason |
|--------|-------------|--------|
| `Type_F_Door` | **Yes** | Set into drywall/CMU wall — `Walls` prims present |
| `Type_FG_Door` | **Yes** | Same — glass panel in solid HM frame, real wall behind |
| `Type_N_Door` | **Yes** | Narrow solid door in wall |
| `Type_FG___New` (curtain wall) | **No** | The curtain wall system IS the wall — no `Walls` prims present |
| `Type_F___NEW` (curtain wall solid) | **No** | Same — curtain wall envelope |

Attempting C1 on curtain-wall doors yields 0 wall meshes and is silently
skipped. The only swappable adjacent geometry on curtain-wall doors is
`Curtain_Wall_Mullions` (the aluminum frame bars), which produces only
a subtle change because the mullions are narrow.

### How wall mesh discovery works

At startup, the script walks `/Geometry/Walls/` and reads the
**leaf mesh** `xformOp:transform` row-3 directly. This gives the world-space
translation baked in by the Revit exporter — the same approach that correctly
locates door hinges. The BBoxCache alternative was attempted but returns
prototype-space (local) coordinates for instanced wall prims, which are
offset from the correct floor level.

```python
walls_root = stage.GetPrimAtPath(f"{BUILDING_ROOT}/Geometry/Walls")
wall_meshes = []
for prim in Usd.PrimRange(walls_root):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    t = _xform_translation(prim)          # reads xformOp:transform row 3
    if t is None:
        continue
    tx, ty, tz = t
    if not (WALL_FLOOR_Z[0] <= tz <= WALL_FLOOR_Z[1]):  # floor z band in cm
        continue
    d = math.sqrt((tx-WALL_CENTRE_CM[0])**2 + (ty-WALL_CENTRE_CM[1])**2)
    if d > WALL_RADIUS_CM:               # 500 cm = 5 m search radius
        continue
    wall_meshes.append(prim)
```

`WALL_CENTRE_CM` is taken from the door's parent matrix translation (row 3).
`WALL_FLOOR_Z` is the expected z-range for the door's floor in cm (e.g.
`(1750, 2150)` for floor 3 ≈ 17.5–21.5 m).

### How the binding swap works

C1 uses the **identical binding-swap mechanism** as the door material
techniques — pre-create OmniPBR material prims at startup, swap
`MaterialBindingAPI` targets per cell, restore from cache. The only
difference is the target mesh set: `wall_meshes` instead of the door mesh
subtree.

```python
# Cache originals
_orig_wall_bindings = []
for p in wall_meshes:
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    _orig_wall_bindings.append((p, list(rel.GetTargets()) if rel else []))

# Per-cell: restore then bind
_restore(_orig_wall_bindings)
if mat_spec["color"] is not None:
    for mesh_prim in wall_meshes:
        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
```

The door is **never touched** in C1 — it stays at its Revit-original binding
throughout. This isolates the wall as the sole variation axis in each C1 cell.

### What was explored and ruled out

During development two other adjacent categories were tested:

- **`Curtain_Wall_Mullions`** — works on curtain-wall doors but produces
  only a subtle frame color change (narrow bars, small fraction of visible
  pixels). Not worth including as a production axis for curtain-wall doors.

- **`Curtain_Panels`** (glass panels in curtain wall) — applying opaque
  OmniPBR makes the glass wall look solid, which was judged too visually
  jarring and unlike any real building condition. Ruled out.

- **`Ceilings`** — not yet tested. May be added in a future iteration.

- **`Rooms` geometry** — Revit room boundary volumes. Untested; may represent
  visible wall faces for some floor plans.

---

## Putting it together — the per-cell grid loop

For each `(lighting, wall_material, door_material, angle)` cell the main loop does:

```python
apply_lighting_condition(lighting)   # write dome + 4 ceiling light attrs
apply_wall_material(wall_material)   # C1: swap wall mesh bindings (solid doors only)
apply_door_material(door_material)   # A-series: swap door mesh bindings
set_door_angle(angle_deg)            # rewrite N×4×4 xform matrices
apply_camera_jitter()                # (optional) small pose noise
for _ in range(WARMUP_STEPS):        # 10 steps — let renderer settle
    world.step(render=True)
rgb_annotator.get_data()             # Replicator readout path
```

All variation axes are just **USD attribute writes** — no shader
recompile, no material rebuild, no scene reload. The renderer picks up the
deltas on the next render step. Five render steps is enough for both the
light contribution and the binding swap to propagate before the annotator is
read. This is why the inspection-scale grid (84 cells) finishes in ~80 seconds
after the initial ~50-second scene load.

All the expensive work — prim creation, shader compile, material
instantiation, building USD load — happens exactly once at startup. The grid
loop itself is pure attribute writes plus a short render settle.

---

## Grid sizes and how they're chosen

| Mode                     | Lightings                   | Materials                 | Angles                | Cameras | Cells | Use                                         |
|--------------------------|-----------------------------|---------------------------|-----------------------|---------|-------|---------------------------------------------|
| Full training grid       | 6                           | 6 flat + 6 textured = 12  | 12 (full stratified)  | 3       | 2,592 | Production data collection (planned)       |
| Technique inspection     | 1 (`overcast_flat`)         | 12 (A1 + A4 + A1+A4)      | 2 (0°, 45°)           | 1       | 24    | Current `door_replicator_collection`       |
| Baseline-only            | 1 (`bright_fluorescent`)    | 1 (`baseline_original`)   | 12                    | 3       | 36    | Generalization-test doors                  |
| Material × lighting audit| 6                           | 12                        | 2 (0°, 45°)           | 1       | 144   | Full audit with the new technique grid     |

The technique-inspection grid deliberately uses a **single** lighting
preset (`overcast_flat`, an even white-balanced render) so the only
visual difference between frames is the material technique. That makes
comparing e.g. `fill_red` vs `fill_tex_red_brushed` trivial — the two
frames differ only in surface pattern. Once a technique passes visual
inspection, the material × lighting audit scales it back up against all
six lighting presets.

`ANGLE_GRID = [0.0, 45.0]` gives one clearly-CLOSED and one
clearly-OPEN frame per preset, enough to verify that the technique
doesn't break under rotation (e.g. a texture sliding through the mesh
when the door swings, which is what `world_or_object=False` protects
against).

---

## Known limitations

- **C1 (wall swap) only works on solid door families.** Curtain-wall doors
  (`Type_FG___New`, `Type_F___NEW`) have no `Walls` category prims — the
  curtain wall mullions and panels are the only swappable adjacent geometry,
  and those produce only a subtle change. C1 is most impactful on `Type_F`,
  `Type_FG`, and `Type_N` families where traditional wall surfaces surround
  the door frame.
- **Glass panels stay glass under `baseline` and `textured`.** By design for
  those two techniques — they only touch the frame. Curtain-wall doors get
  limited appearance variation from them because most of the visible surface
  is glass. Use `fill_glass`, `fill_textured`, or `glass_variant` when you
  need to change the glass side too.
- **Grayscale textures + tint are lit-independent only at the diffuse term.**
  Roughness and metallic are constants per preset, so two `fill_textured`
  presets that share a pattern but differ in tint also end up sharing
  specular highlight shape. If we later need per-pattern specular variation,
  the path is either (a) per-preset roughness / metallic already supported,
  or (b) add a `reflectionroughness_texture` input for a spatially varying
  rough map.
- **Triplanar projection is approximate near sharp edges.** `project_uvw=True`
  works well on the flat curtain-wall panels but seams would show on a
  door with aggressive bevels. No curtain-wall door in the current dataset
  has these, but solid `Type_F_Door` variants might — revisit if the
  textured techniques get extended to solid doors.
- **Ceiling light layout is fixed per scene.** The 2×2 grid position is
  anchored to the door center, so moving to a different door requires
  re-placing the ceiling fixtures. Currently done implicitly because the
  lights are defined relative to `DOOR_CENTER_XY` + `CAMERA_Z_BASE` at
  startup.
- **Lighting is discrete, not continuous.** Six presets is coarse compared
  to a continuous sampler. This was a deliberate tradeoff for interpretability
  — you can look at a frame and say "that's `dim_night`" — but the generalization
  curve will plateau if the gap between presets is too large. If 6 presets
  are not enough, the fix is more presets (spanning new regimes), not finer
  interpolation between existing ones.
- **Color temperature conversion is approximate.** Tanner Helland's blackbody
  approximation is good enough for training data but will not match a
  reference photometer. Not a concern for sim-to-real transfer because the
  real-world evaluation uses its own camera white balance.
