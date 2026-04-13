# Lighting and Material Variation

Domain randomization for the door-state dataset. This doc explains **why** we
vary lighting and door materials the way we do, and **how** the variation is
implemented in the Isaac Sim data-collection scripts
([door_static_collection.py](simulations/robot/door_static_collection.py),
[door_replicator_collection.py](simulations/robot/door_replicator_collection.py)).

Both scripts share the same presets and the same per-cell application logic —
the Replicator script is just a re-routing of the frame-readout path through
`rep.AnnotatorRegistry` instead of `Camera.get_rgba()`.

---

## Why we vary these two axes at all

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
byte-identical. See `apply_lighting_condition()`
at [door_replicator_collection.py:394](simulations/robot/door_replicator_collection.py#L394).

### Color temperature → RGB

The Kelvin values are converted to RGB via Tanner Helland's blackbody
approximation (`_color_temp_to_rgb()` at
[door_replicator_collection.py:371](simulations/robot/door_replicator_collection.py#L371)),
then written to the light's `color` attribute. This is approximate but fast
and deterministic — good enough for the low-dimensional lighting axis.

---

## Materials — six pre-created OmniPBR presets plus "original"

### Design rationale

We need the door to look like it could be any real door: aluminum curtain
wall, painted wood, fire-rated steel, etc. The cheapest way to do this
without replacing meshes is to swap the bound material on the existing door
geometry.

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

### The seven presets

| Preset                | Diffuse RGB              | Roughness | Metallic | Intent                           |
|-----------------------|--------------------------|-----------|----------|----------------------------------|
| `brushed_aluminum`    | (0.75, 0.75, 0.78)       | 0.35      | 0.85     | Silver curtain-wall frame         |
| `painted_white`       | (0.90, 0.90, 0.88)       | 0.70      | 0.05     | Institutional white door         |
| `dark_anodized`       | (0.12, 0.12, 0.15)       | 0.25      | 0.80     | Black anodized metal             |
| `wood_veneer`         | (0.55, 0.35, 0.20)       | 0.75      | 0.02     | Warm wood-finish door            |
| `red_fire_door`       | (0.65, 0.10, 0.08)       | 0.60      | 0.10     | Fire-rated steel door            |
| `beige_institutional` | (0.72, 0.68, 0.58)       | 0.65      | 0.08     | Muted neutral panel              |
| `original`            | — (sentinel)             | —         | —        | Restore Revit's original binding |

Presets are chosen to span the color-roughness-metallic cube with visually
distinct corners rather than cluster near the curtain-wall defaults. Vivid
red, warm brown, and near-black give the model obviously different door
appearances under every lighting preset.

### How the binding swap works

At startup (`_door_preset_materials` dict in
[door_replicator_collection.py:258](simulations/robot/door_replicator_collection.py#L258)):

```python
for preset in DOOR_MATERIALS:
    if preset["color"] is None:        # the "original" sentinel
        continue
    mat = UsdShade.Material.Define(stage, f"/World/Looks/DoorMat_{preset['name']}")
    sh  = UsdShade.Shader.Define(stage, f"/World/Looks/DoorMat_{preset['name']}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant",        ...).Set(Gf.Vec3f(*preset["color"]))
    sh.CreateInput("reflection_roughness_constant", ...).Set(preset["roughness"])
    sh.CreateInput("metallic_constant",             ...).Set(preset["metallic"])
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    # ... plus displacement and volume outputs
```

At each grid cell (`apply_door_material()` at
[door_replicator_collection.py:436](simulations/robot/door_replicator_collection.py#L436)):

```python
mat = _door_preset_materials[spec["name"]]
strong = UsdShade.Tokens.strongerThanDescendants
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh) or _is_glass_mesh(p):
        continue
    UsdShade.MaterialBindingAPI.Apply(p)
    UsdShade.MaterialBindingAPI(p).Bind(mat, bindingStrength=strong)
```

This is a **binding relationship swap**, not a shader edit. The delegate sees
an entirely different material prim and picks up a different shader. Cost per
cell is O(meshes under the door) USD writes — no shader compile, no material
rebuild.

`strongerThanDescendants` is necessary because Revit exports frequently bind
materials at a nested scope, and without the stronger binding strength your
override gets out-prioritized by the original binding.

### Glass meshes are skipped

`_is_glass_mesh()` at
[door_replicator_collection.py:346](simulations/robot/door_replicator_collection.py#L346)
filters out children whose names start with `Glass_` or contain `_GL_`,
`glazing`, or `pane_`. These are the Revit export tags for transparent
glass panels. Tinting them with an opaque OmniPBR color kills transparency
and looks worse than leaving them alone.

**Consequence**: on mostly-glass curtain-wall doors like `Doors_13606713`,
the material swap only affects the aluminum frame panels — the glass stays
glass. That's expected and correct. For fully-solid doors (e.g.
`Doors_Type_F_Door` variants) the swap covers the entire door.

### Restoring the "original" material

The `original` preset is not a no-op. At startup we cache every non-glass
mesh's direct-binding targets:

```python
_original_mat_bindings = []
for p in Usd.PrimRange(door_prim):
    if not p.IsA(UsdGeom.Mesh) or _is_glass_mesh(p):
        continue
    rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel()
    targets = list(rel.GetTargets()) if rel else []
    _original_mat_bindings.append((p, targets))
```

When the `original` cell is applied, `_restore_original_door_materials()`
replays an unbind + rebind-targets round trip per mesh. Without this cache
you'd permanently corrupt the scene after the first non-original cell and
"original" would actually mean "whatever was bound last."

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

## Putting it together — the per-cell grid loop

For each `(lighting, material, angle)` cell the main loop does:

```python
apply_lighting_condition(lighting)   # write dome + 4 ceiling light attrs
apply_door_material(material)        # swap MaterialBindingAPI targets
set_door_angle(angle_deg)            # rewrite N×4×4 xform matrices
apply_camera_jitter()                # (optional) small pose noise
for _ in range(WARMUP_STEPS):        # 5 steps — let renderer settle
    world.step(render=True)
rgb_annotator.get_data()             # Replicator readout path
```

All three variation axes are just **USD attribute writes** — no shader
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

| Mode                    | Lightings | Materials | Angles                  | Cells | Use                                   |
|-------------------------|-----------|-----------|-------------------------|-------|---------------------------------------|
| Full training grid      | 6         | 7         | 12 (full stratified)    | 504   | Production data collection            |
| Visual inspection       | 6         | 7         | 2 (0°, 45°)             | 84    | Current `door_replicator_collection`  |
| Baseline-only           | 1         | 1         | 12                      | 12    | Generalization-test doors             |
| Material × lighting audit| 6        | 7         | 2 (0°, 45°)             | 84    | `--sanity-grid` in static script      |

The 504-cell full grid is the real training dataset shape per door, giving
balanced coverage across every combination. The 84-cell inspection grid
(`ANGLE_GRID = [0.0, 45.0]` in the Replicator script) is a deliberately small
fraction of that, sized to be visually auditable one JPEG at a time — any
material or lighting bug shows up in a single browse of the dump directory.

---

## Known limitations

- **Glass panels don't recolor.** By design. Curtain-wall doors get limited
  appearance variation from material swaps because most of their visible
  surface is glass.
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
