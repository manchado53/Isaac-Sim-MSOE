# WACV 2027 paper — LaTeX build

Generated from `../paper.md`. Two-column CVF/WACV template (the official
`cvpr.sty` author kit, with the banner set to WACV 2027 via `\confName`/`\confYear`
in `paper.tex`).

## Build
Needs a TeX Live install with `pdflatex` + `bibtex` (present on Rosie login nodes):

    make            # builds paper.pdf
    make clean      # removes LaTeX intermediates

## Files
- `paper.tex`  — the paper (sections, tables, figure placeholders)
- `paper.bib`  — references (a few have TODO notes: Nav2, SWAGGER, Qwen3-VL)
- `cvpr.sty`, `ieeenat_fullname.bst` — CVF author-kit style + bib style
- `paper.pdf`  — built output (8 pp)

## Switches (top of paper.tex)
- `\usepackage{cvpr}`           → named draft (current)
- `\usepackage[review]{cvpr}`   → anonymous review build (line numbers, paper ID)

## Figures
All five figures are inserted and built from repo data by the scripts in
`../figures_src/` (outputs land in `figures/`). Regenerate with:

    for f in overview navgraph nav2 confusion synthdata; do \
      python3 ../figures_src/make_$f.py; done

- `fig_overview`  — pipeline block diagram (Fig. 1)
- (`innovation_center_exterior.png` — the building / twin, Fig. 2)
- `fig_navgraph`  — floor-4 occupancy + nav graph + TSP/Dijkstra route (Fig. 3)
- `fig_nav2`      — REAL follow + onboard frames of the Nav2-driven robot at a
  floor-4 patrol waypoint (Fig. 4). Frames are extracted from the tracked demo
  cache `door_demo_cache_238942.npz` into `figures/nav2_{follow,onboard}.png` by
  `make_nav2.py::regenerate()`; delete those PNGs to force re-extraction.
- `fig_synthdata` — intact vs. 3 crack-style renders from the twin (Fig. 5)
- `fig_confusion` — door/glass/real-world confusion matrices (Fig. 6)

`fig_confusion` reconstructs counts from the reported per-class accuracies: the
real-world (Diercks) split is exact; the two synthetic splits assume balanced
classes (this reproduces the paper's overall accuracies). Caption states this.

## Pending assets (on another machine)
- **Fig. 5 (synthetic glass)** — swap the tight `panel00` crops for real high-res
  heavy-shatter frames from `datasets/cracked_glass/<run>/rgb/`, then re-run
  `../figures_src/make_synthdata.py`.
- **Fig. 6 (confusion)** — the door/glass held-out cell *counts* are reconstructed
  from a balanced-class assumption (exact in method since the eval stratifies
  50/50; ±1 from rounded per-class %). Drop in `data/finetuned_eval_results.json`
  for exact integers, then re-run `../figures_src/make_confusion.py`.

## Still to do
- **Quantitative Nav2 route-execution numbers** (Sec. 4.3) — completion rate,
  path length vs. graph-optimal, recovery counts, obstacle-avoidance success.
  The bracketed TODO is gone; the text now says this is ongoing work.
- `fig_nav2` now uses REAL rendered sensor frames from a Nav2-driven floor-4 run
  (robot at a door-inspection waypoint), not a synthetic costmap render. It shows
  arrival/inspection, not local obstacle avoidance — the avoidance behaviour is
  documented via the Nav2 parameter table (Tab. 3) and text. A real RViz/Foxglove
  costmap capture could still be added if/when one is recorded.
- Real Diercks Hall example frames could be added to `fig_confusion` if photos
  become available.
- Confirm Nav2 / SWAGGER / Qwen3-VL citations in `paper.bib`.
