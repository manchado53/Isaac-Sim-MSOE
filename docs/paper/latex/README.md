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

## Still to do (also flagged inline as `[TODO]` / `[FIGURE]`)
- Insert the 5 figures (overview, navgraph+route, Nav2 obstacle avoidance,
  Diercks Hall confusion matrices/examples, Dijkstra path).
- Quantitative Nav2 route-execution numbers in Sec. 4.3.
- Confirm Nav2 / SWAGGER / Qwen3-VL citations in `paper.bib`.
