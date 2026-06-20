# Plan: Updated full paper draft (WACV 2027 Applications)

## What we do
Take the old MICS2026 PDF as the skeleton, but rewrite the two parts that changed
most, and update the abstract/intro/related-work/conclusion to match. Write one
full `docs/paper/paper.md` (Markdown, venue-neutral formatting, WACV-length aim).

## Big changes vs the old PDF
1. **Navigation: RL is OUT, Nav2 is IN.**
   - Old §3.4 "RL Navigation Training Infrastructure" + §4.3 "RL validation" get
     replaced by a Nav2 local-navigation section + a Nav2/route results section.
   - Keep occupancy-map + nav-graph + TSP, but say the graph is placed with
     **NVIDIA SWAGGER** (per your call), and the robot drives the route with
     **ROS 2 Nav2** (SmacPlanner2D global + DWB local + AMCL, Jazzy), LiDAR for
     obstacle avoidance, single NavigateThroughPoses goal over the TSP route.
   - RL work noted only as "earlier approach / archived" (1-2 sentences), not a
     results section.

2. **VLM: bigger and current.**
   - Model is **Qwen3-VL-2B-Instruct** (not Qwen2), LoRA + vision-projector
     co-training, 1.62% params.
   - **Two anomaly classes now: open doors + cracked glass** (4-way label set:
     door open/closed, glass cracked/intact).
   - New numbers from Tanner's branch (`origin/tanner_vlm`):
     - Zero-shot model comparison table (Qwen3-VL 69.3%, Qwen2-VL 62.1%,
       Qwen2.5-VL-3B 64.9%, -7B 68.7%, InternVL3-2B 62.9%, SmolVLM2 62.1%).
     - Fine-tuned: door val 97.4%, glass val 97.2% (cracked 98.8 / intact 95.6).
     - Held-out synthetic doors 90.2% (vs 84.1% zero-shot).
     - Real-world Diercks Hall doors 93.0% — the generalization test.
   - Honest framing of the old single-door brittleness (0% on unseen door) as the
     motivation for the multi-door (8 train + 2 holdout, many families) + glass
     + domain-randomization expansion.
   - Cracked-glass pipeline: OmniGlass normal-map swap, 3 crack styles
     (tempered Voronoi, SDXL impact, edge), Replicator semantic-seg labels,
     visual-only, ~1013 panels.
   - Onboard camera: first-person RGB 640×480, 20 fps, 5 mm.

3. **Scaling / scope:** six-building digital campus, >3,500 GPU jobs,
   >2,000 GPU-hours on Rosie.

## Section layout (paper.md)
- Title + author block + abstract (reuse current abstract, light sync)
- 1 Introduction (new contributions list: Nav2, 2-anomaly VLM)
- 2 Related Work (BIM nav + Isaac/Isaac Lab + classical planning/Nav2 + VLM/LoRA + model comparison)
- 3 System Architecture
  - 3.1 Building the Dataset (Revit → trimmed BIM)
  - 3.2 Scene Preparation (BIM→USD, Isaac Sim, physics/doors)
  - 3.3 Occupancy Map, Navigation Graph (SWAGGER) & Route Planning (Dijkstra/TSP)
  - 3.4 Local Navigation with ROS 2 Nav2  ← replaces RL
  - 3.5 Vision-Language Perception (Qwen3-VL, doors + glass)
- 4 Evaluation
  - 4.1 Scene Fidelity
  - 4.2 Navigation Graph Coverage
  - 4.3 Navigation & Route Execution (Nav2)  ← replaces RL results
  - 4.4 Vision-Language Perception Evaluation (baseline table + fine-tuned + real-world)
- 5 Discussion (5.1 Limitations, 5.2 Sim-to-Real, 5.3 Future Work)
- 6 Conclusion
- References (carry old refs; add SWAGGER, Nav2, Qwen3-VL, classical TSP/Dijkstra)

## Conventions
- Every number traces to a repo file / eval JSON / the deck — no invented stats.
- `[FIGURE: ...]` and `[TODO: ...]` placeholders where an asset or a not-yet-
  recorded metric is needed (e.g. exact Nav2 patrol completion video stats).
- Keep all numbers in SI/meters; note metersPerUnit only if needed.
- Markdown now; convert to the WACV template later.

## Out of scope (this pass)
- No LaTeX/template conversion, no figure generation, no committing to main.
- Won't merge Tanner's data files; just cite their numbers in prose/tables.
