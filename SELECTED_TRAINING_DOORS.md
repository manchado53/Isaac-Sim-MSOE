# Selected Training Doors

Final 10 doors for multi-door dataset expansion, signed off after visual review
of the 60-JPEG lighting audit (`data/lighting_test/`, job 243457) across all 6
lighting presets. Source of truth for the per-door config (hinge, camera pose,
`angle_sign`, `mesh_count`) is the `DOORS` list in
[simulations/robot/lighting_test_all_doors.py](simulations/robot/lighting_test_all_doors.py).

**8 training + 2 holdout.** Holdouts never appear in training frames — they
exist to measure out-of-distribution generalization (unseen door instance and
unseen swing direction).

> **Note — Type_F_Sidelite dropped earlier:** The 3 original Type_F_Sidelite
> doors (Doors_8285993, Doors_8431775, Doors_9499392) were removed from
> consideration because the Revit export merged the door leaf and frame into a
> single `Paint` mesh with no intermediate Xform, so rotating the leaf dragged
> the frame with it. They were replaced by doors from other families.

---

## Train / Holdout Split

| Role | Door ID | Family | Floor | Approach | mesh_count | Hinge space | angle_sign | Note |
|---|---|---|---|---|---|---|---|---|
| **TRAIN** | Doors_13606466 | CW_F_NEW | 4 | X | 1 | world | +1 | |
| **TRAIN** | Doors_13606056 | CW_F_NEW | 3 | X | 1 | world | +1 | |
| **TRAIN** | Doors_3135827 | CW_FG_New | 3 | Y | 2 | world | +1 | |
| **TRAIN** | Doors_12302795 | CW_FG_New | 2 | X | 2 | world | +1 | |
| **TRAIN** | Doors_8286057 | Type_F | 2 | Y | 0 (instanced) | local | +1 | Inward |
| **TRAIN** | Doors_7171063 | Type_F | 5 | X | 0 (instanced) | local | −1 | **Outward-swinging** (42″ wide); gives training the outward-swing pattern |
| **TRAIN** | Doors_12590968 | Type_N | 3 | Y | 0 (instanced) | local | +1 | Parent row1 `[0,-1,0,0]` |
| **TRAIN** | Doors_10649156 | Type_FG | 3 | X | 0 (instanced) | local | −1 | Only Type_FG in training; parent maps local Y = +world X |
| **HOLDOUT** | Doors_13182085 | CW_FG_New | 4 | X | 2 | world | +1 | Unseen CW_FG_New instance — clean instance-level generalization probe |
| **HOLDOUT** | Doors_1915613 | Type_F | 5 | Y | 0 (instanced) | local | +1 | Inward Type_F instance probe; parent `[-1,-1]` rotation |

---

## Family Coverage

| Family | Full Revit Name | Description | Training Count | Holdout Count | Floors |
|---|---|---|---|---|---|
| CW_F_NEW | RS___Curtain_Wall___Type_F___NEW | Curtain-wall opaque door | 2 | 0 | 3, 4 |
| CW_FG_New | RS___Curtain_Wall___Type_FG___New | Curtain-wall glass door | 2 | 1 | train 2, 3 / holdout 4 |
| Type_F | RS___Type_F_Door | Standard solid HM swing door | 2 | 1 | train 2, 5 / holdout 5 |
| Type_FG | RS___Type_FG_Door | Glass-panel swing door | 1 | 0 | 3 |
| Type_N | RS___Type_N_Door | Narrow swing door | 1 | 0 | 3 |

**Every family is represented in training.** The two holdouts test clean
instance-level generalization within seen families (rather than fully-unseen-
family generalization, which would be a confounded test — failure couldn't be
attributed to instance-level vs family-level gaps). Both holdouts pick a door
whose family and hinge-geometry class already appear in training.

**Floor coverage:** training doors span floors 2, 3, 4, and 5. Floor 1 is not
represented because it's largely a lobby/atrium without the interior corridor
doors this dataset targets.

---

## Swing Direction Rationale

**Definitions** (from the robot's perspective standing in the corridor):
- **Inward** — door leaf swings away from the robot into the room.
- **Outward** — door leaf swings toward the robot into the corridor.

**7 of 8 training doors swing inward; 1 swings outward (Doors_7171063).**
Interior corridor doors in commercial buildings almost universally swing into
the room, both by convention and by fire/life-safety codes that keep egress
corridors clear, so the inward majority matches the real-world distribution
the robot will encounter during patrol of the Kern Engineering Innovation
Center. The single outward training example is deliberate: it gives the model
at least one exposure to the visual pattern where an open leaf sweeps *toward*
the camera, so the learned representation can include "door as foreground
obstacle" rather than only "door as side-of-frame frame element."

**Both holdouts swing inward** (Doors_1915613 Type_F and Doors_13182085
CW_FG_New). The holdout split tests *instance-level* generalization within
patterns the training distribution has seen — not fully out-of-distribution
swing directions. An outward-swinging holdout with zero outward training
examples would conflate two failure modes (never-seen visual pattern vs
never-seen instance), making a holdout failure impossible to attribute.
Keeping the one outward example in training resolves that confound.

---

## Rotation Mechanics — World vs Local

Two code paths exist in `door_replicator_collection.py` and
`lighting_test_all_doors.py`, controlled by `mesh_count`:

- **`mesh_count > 0` → world-space rotation.** Walk `door_prim.GetChildren()`,
  read each leaf mesh's `xformOp:transform` row-3 (world translation in cm),
  rewrite the 4×4 matrix around the world hinge. Used for curtain-wall doors
  (`CW_F_NEW`, `CW_FG_New`) where the leaf meshes are concrete prims with
  real world-space transforms.
- **`mesh_count == 0` → local-space rotation.** Walk `Usd.PrimRange(door_prim)`
  recursively (because the subtree is an instanced prototype), read each
  mesh's local `xformOp:transform`, rewrite around a local hinge at
  `(+half_width, 0)`. Used for all HM-frame solid doors (`Type_F`, `Type_FG`,
  `Type_N`). Requires `SetInstanceable(False)` first to expose the prototype
  subtree for mutation.

The `angle_sign` field flips the rotation direction when the door's parent
matrix has a negative determinant that would otherwise swing the leaf the
wrong way. Verified per-door by the hinge diagnostic scripts.

---

## Per-Door Details

All `hinge_x_cm` / `hinge_y_cm` / `cam_pos` / `cam_yaw_deg` / `door_center_m`
values below are copied verbatim from `lighting_test_all_doors.py`. The door
center is `cam_pos` projected 2 m along the look direction and is used to
anchor the 4-slot ceiling rect-light array above the door.

### 1 — Doors_13606466 · CW_F_NEW · Floor 4 · TRAIN

Curtain-wall opaque door.

| Property | Value |
|---|---|
| Prim path | `…/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606466` |
| Family | RS___Curtain_Wall___Type_F___NEW |
| Floor | 4 |
| Approach axis | X |
| `mesh_count` | 1 |
| Hinge (world cm) | (−4836.01, 6170.58) |
| Camera pos (m) | (−46.360, 62.163, 23.704) |
| Camera yaw | 180° |
| Door centre (m) | (−48.360, 62.163, 23.704) |
| `angle_sign` | +1 |

---

### 2 — Doors_10649156 · Type_FG · Floor 3 · TRAIN

Glass-panel swing door (solid HM frame with a glass vision lite). Instanced
prototype — rotation uses local-space 4×4 matrix after `SetInstanceable(False)`.

> **`angle_sign = −1`** (verified by hinge test job 243071). Positive angle
> drove the door toward the camera. The parent matrix for this instance maps
> local Y = +world X, opposite to the other `Type_*` instances. Negating the
> angle corrects the swing direction so the leaf opens inward.

The **only** Type_FG door in the roster. Promoted from holdout to training so
that the `Type_FG` family is represented in the training distribution —
otherwise a holdout-only family confounds instance-level vs family-level
generalization on the holdout eval.

| Property | Value |
|---|---|
| Prim path | `…/RS___Type_FG_Door/NEW___HM_Frame___36_x96/Doors_10649156` |
| Family | RS___Type_FG_Door |
| Floor | 3 |
| Approach axis | X |
| `mesh_count` | 0 (instanced) |
| Hinge (local cm) | (+50.80, 0.00) |
| Camera pos (m) | (−14.295, 80.200, 19.437) |
| Camera yaw | 0° |
| Door centre (m) | (−12.295, 80.200, 19.437) |
| `angle_sign` | −1 |

---

### 3 — Doors_3135827 · CW_FG_New · Floor 3 · TRAIN

Curtain-wall glass door.

| Property | Value |
|---|---|
| Prim path | `…/RS___Curtain_Wall___Type_FG___New/New/Doors_3135827` |
| Family | RS___Curtain_Wall___Type_FG___New |
| Floor | 3 |
| Approach axis | Y |
| `mesh_count` | 2 |
| Hinge (world cm) | (1724.33, 5878.48) |
| Camera pos (m) | (17.701, 56.807, 19.437) |
| Camera yaw | 90° |
| Door centre (m) | (17.701, 58.807, 19.437) |
| `angle_sign` | +1 |

---

### 4 — Doors_12302795 · CW_FG_New · Floor 2 · TRAIN

Curtain-wall glass door. Prim lives under the `New___GL1_1` subfamily, not
`New` — a Revit variant marker for a different glazing SKU. Visually
indistinguishable from the other two CW_FG_New doors.

| Property | Value |
|---|---|
| Prim path | `…/RS___Curtain_Wall___Type_FG___New/New___GL1_1/Doors_12302795` |
| Family | RS___Curtain_Wall___Type_FG___New |
| Floor | 2 |
| Approach axis | X |
| `mesh_count` | 2 |
| Hinge (world cm) | (−5925.83, 5774.34) |
| Camera pos (m) | (−57.258, 58.251, 14.561) |
| Camera yaw | 180° |
| Door centre (m) | (−59.258, 58.251, 14.561) |
| `angle_sign` | +1 |

---

### 5 — Doors_13606056 · CW_F_NEW · Floor 3 · TRAIN

Curtain-wall opaque door. Second `CW_F_NEW` in training — gives the only
opaque-curtain-wall family two different floors of appearance variation.

| Property | Value |
|---|---|
| Prim path | `…/RS___Curtain_Wall___Type_F___NEW/New/Doors_13606056` |
| Family | RS___Curtain_Wall___Type_F___NEW |
| Floor | 3 |
| Approach axis | X |
| `mesh_count` | 1 |
| Hinge (world cm) | (−4747.75, 6958.30) |
| Camera pos (m) | (−45.477, 70.040, 19.437) |
| Camera yaw | 180° |
| Door centre (m) | (−47.477, 70.040, 19.437) |
| `angle_sign` | +1 |

---

### 6 — Doors_8286057 · Type_F · Floor 2 · TRAIN

Standard solid HM swing door. Instanced prototype — rotation uses local-space
4×4 matrix after `SetInstanceable(False)`.

| Property | Value |
|---|---|
| Prim path | `…/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_8286057` |
| Family | RS___Type_F_Door |
| Floor | 2 |
| Approach axis | Y |
| `mesh_count` | 0 (instanced) |
| Hinge (local cm) | (+50.80, 0.00) |
| Camera pos (m) | (17.443, 79.341, 14.764) |
| Camera yaw | 270° |
| Door centre (m) | (17.443, 77.341, 14.764) |
| `angle_sign` | +1 |

---

### 7 — Doors_7171063 · Type_F · Floor 5 · TRAIN

Standard solid HM swing door, 42″ wide (all other Type_F doors in the roster
are 36″). Instanced prototype.

**Outward-swinging.** `angle_sign = −1` rotates the leaf into the corridor
toward the camera rather than into the room — the only outward-swinging
training door. Promoted from holdout so the training distribution contains
at least one instance of the "open leaf as foreground obstacle" visual
pattern; otherwise the model could only learn the inward-swing pattern where
an open door reveals a clear passage.

| Property | Value |
|---|---|
| Prim path | `…/RS___Type_F_Door/NEW___HM_Frame___42_x96/Doors_7171063` |
| Family | RS___Type_F_Door |
| Floor | 5 |
| Approach axis | X |
| `mesh_count` | 0 (instanced) |
| Hinge (local cm) | (+58.42, 0.00) |
| Camera pos (m) | (−49.316, 49.892, 27.972) |
| Camera yaw | 0° |
| Door centre (m) | (−47.316, 49.892, 27.972) |
| `angle_sign` | −1 (outward) |

---

### 8 — Doors_12590968 · Type_N · Floor 3 · TRAIN

Narrow swing door. Instanced prototype. Parent matrix row1 `[0, −1, 0, 0]`
(Y-axis flipped).

| Property | Value |
|---|---|
| Prim path | `…/RS___Type_N_Door/NEW___HM_Frame___36_x96/Doors_12590968` |
| Family | RS___Type_N_Door |
| Floor | 3 |
| Approach axis | Y |
| `mesh_count` | 0 (instanced) |
| Hinge (local cm) | (+50.80, 0.00) |
| Camera pos (m) | (17.878, 50.941, 19.437) |
| Camera yaw | 90° |
| Door centre (m) | (17.878, 52.941, 19.437) |
| `angle_sign` | +1 |

---

### 9 — Doors_13182085 · CW_FG_New · Floor 4 · HOLDOUT

Curtain-wall glass door. Vanilla `New/` subfamily — same Revit SKU as the
other two CW_FG_New training doors (Doors_3135827, Doors_12302795 lives under
`New___GL1_1`, so this holdout matches floor-3 Doors_3135827 exactly at the
prim-path level).

Clean **instance-level generalization probe within a seen family.** The
CW_FG_New family is represented in training by 2 doors on floors 2 and 3;
this holdout sits on floor 4, so the split also holds out one floor of the
family. Failure here would indicate the model memorised per-instance surface
cues (glass reflections, adjacent mullion pattern) rather than learning a
reusable CW_FG_New open/closed representation.

| Property | Value |
|---|---|
| Prim path | `…/RS___Curtain_Wall___Type_FG___New/New/Doors_13182085` |
| Family | RS___Curtain_Wall___Type_FG___New |
| Floor | 4 |
| Approach axis | X |
| `mesh_count` | 2 |
| Hinge (world cm) | (−4142.56, 8214.32) |
| Camera pos (m) | (−39.426, 82.600, 23.704) |
| Camera yaw | 180° |
| Door centre (m) | (−41.426, 82.600, 23.704) |
| `angle_sign` | +1 |

---

### 10 — Doors_1915613 · Type_F · Floor 5 · HOLDOUT

Standard solid HM swing door (36″ wide), inward-swinging. Instanced prototype.
Parent matrix `[-1, -1]` rotation — positive `angle_sign` opens the leaf
into the room, verified by the hinge diagnostic.

**Clean instance-level generalization probe within the seen Type_F family.**
Type_F is represented in training by Doors_8286057 (floor 2, inward) and
Doors_7171063 (floor 5, outward 42″). This holdout is floor 5 inward 36″ —
same swing direction as Doors_8286057 and same floor as Doors_7171063, but a
new instance with a different camera pose. A failure here would indicate the
model memorised per-instance surface cues or pose-specific lighting rather
than learning a reusable inward Type_F open/closed representation.

| Property | Value |
|---|---|
| Prim path | `…/RS___Type_F_Door/NEW___HM_Frame___36_x96/Doors_1915613` |
| Family | RS___Type_F_Door |
| Floor | 5 |
| Approach axis | Y |
| `mesh_count` | 0 (instanced) |
| Hinge (local cm) | (+50.80, 0.00) |
| Camera pos (m) | (−58.753, 65.219, 27.972) |
| Camera yaw | 270° |
| Door centre (m) | (−58.753, 63.219, 27.972) |
| `angle_sign` | +1 |

---

## Changes From Previous Selection

The earlier 10-door roster documented in this file has been superseded.
Summary of swaps (reasons recorded from visual review of
`data/lighting_test/` and `data/door_candidates/`):

| Action | Door | Reason |
|---|---|---|
| Replaced | Doors_13607620 (CW_F_NEW, floor 2) → **Doors_13606466** (CW_F_NEW, floor 4) | Better camera framing on the new door during lighting review |
| Replaced | Doors_4078243 (Type_F, floor 5, sideN) → **Doors_1915613** (Type_F, floor 5, sideP) | Initially added as a Type_F trainer on the opposite side of the floor from Doors_7171063 (later re-assigned to holdout; see bottom row of this table) |
| Removed | Doors_10649710 (Type_FG, floor 3) | Family visually cluttered under curtain-wall adjacency; moved Type_FG to holdout-only |
| Removed | Doors_2429516 (Type_N, floor 4) | Reduces Type_N training count to 1; made room for the two CW additions |
| Added | **Doors_12302795** (CW_FG_New, floor 2, sideP) | Third CW_FG_New, gives the family coverage across floors 2/3/4 |
| Added | **Doors_13606056** (CW_F_NEW, floor 3, sideP) | Second CW_F_NEW, gives the opaque-curtain-wall family two floors of variation |
| Swapped | **Doors_10649156** (Type_FG) train ↔ **Doors_13182085** (CW_FG_New, floor 4) holdout | Every family now has a training instance. Type_FG holdout-only would confound instance-level vs family-level generalization on the eval. CW_FG_New has 3 candidates, so demoting one to holdout still leaves 2 training instances across floors 2 and 3. |
| Swapped | **Doors_7171063** (Type_F, floor 5, outward 42″) train ↔ **Doors_1915613** (Type_F, floor 5, inward 36″) holdout | Put the outward-swing visual pattern into training — previously it appeared only on the holdout, which would conflate "never-seen swing direction" with "never-seen instance" on any holdout failure. Demoting Doors_1915613 (inward) keeps Type_F training on two different floors (2 and 5) and both holdouts now swing inward, so neither holdout tests a pattern the training set never saw. |

**Consequences for training distribution:**
- CW family training count: 4 (2 CW_F_NEW + 2 CW_FG_New) → **4 of 8** training doors are curtain-wall.
- Solid-door training count: **4 of 8** (2 Type_F + 1 Type_FG + 1 Type_N).
- **Every family appears in training.** No fully-unseen family — both holdouts are seen-family instance probes.
- Type_FG training count: 1 (Doors_10649156).
- Type_N training count: 1 (Doors_12590968) — remains the smallest-represented family.
- Training swing-direction split: **7 inward + 1 outward** (Doors_7171063 is the sole outward trainer).
- Both holdouts swing inward — no swing-direction confound on the eval.

This balances the training distribution between curtain-wall glass and solid
HM-frame regimes (4 of each), represents every door family in training, and
ensures each holdout failure can be attributed to instance-level
generalization rather than a visual pattern the training set never saw.
