# Plan: Analyze MSOEBuildingTake2.usd and Switch Pipeline If Better

## Context
We've been using `TestBuilding.usd` (19MB) for occupancy map generation, but it references 2,176 external prop files (interior walls, doors, partitions) that we don't have. This means our occupancy maps are mostly open space with only the exterior shell. A new file `MSOEBuildingTake2.usd` (35MB, nearly 2x bigger) was added — initial analysis shows more interior geometry keywords (rooms, doors, walls) and fewer external references, suggesting props may be embedded directly.

## Goal
Determine if MSOEBuildingTake2.usd has embedded interior geometry (solving the missing props problem), and if so, switch the occupancy map pipeline to use it and regenerate all outputs.

## Steps

### Step 1: Run USD analysis script (DONE)
Created `test_building/analyze_usd.py` — runs in Isaac Sim and compares both files:
- Total prim count and mesh count
- Number of missing/unresolved external references
- Mesh names categorized: walls, doors, floors, HVAC, structural, other
- Bounding box comparison

**Run with:** `sbatch submit_isaac_job.sh test_building/analyze_usd.py`

### Step 2: Review analysis results
- Check `test_building/output/usd_comparison.json` and job log
- Confirm which file has more meshes and fewer missing references

### Step 3: If MSOEBuildingTake2.usd is better — update the pipeline
- Update `USD_PATH` in `generate_occupancy_map.py` to point to `MSOEBuildingTake2.usd`
- May need to re-tune `SKIP_PREFIXES` based on new mesh names
- May need to re-tune floor Z slices if the geometry differs
- Re-run `find_floor_levels.py` first to get correct Z levels

### Step 4: Regenerate occupancy maps
- `sbatch submit_isaac_job.sh test_building/generate_occupancy_map.py`
- Verify the new maps show interior walls/rooms (higher occupied %)

### Step 5: Rebuild navigation graphs
- `python3 test_building/build_nav_graph.py`
- Verify edges don't cross walls, graph reflects actual room layout

### Step 6: Update pipeline.md
- Document the USD file switch and results comparison

## Files involved
- `test_building/analyze_usd.py` — comparison script (runs in Isaac Sim)
- `test_building/generate_occupancy_map.py` — change `USD_PATH` (line 36)
- `test_building/find_floor_levels.py` — change USD path
- `test_building/build_nav_graph.py` — no changes needed (reads .npy files)
- `test_building/pipeline.md` — update documentation

## Verification
1. Analysis script output shows MSOEBuildingTake2.usd has more meshes AND fewer missing references
2. New occupancy maps show higher occupied % (more walls detected)
3. Nav graph PNGs show edges following actual corridors between rooms
4. No edges crossing walls in the visualization
