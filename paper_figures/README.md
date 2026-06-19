# Paper Figures

Curated, paper-ready images for the project writeup. These were copied out of
gitignored working directories (`datasets/`, `images/`) so they are preserved in
git and the remote. The bulk dataset (~4.4 GB of raw frames) stays gitignored.

| Folder | File(s) | Suggested role | Original source |
|--------|---------|----------------|-----------------|
| `taxonomy/` | `cracked_glass_variants.png` | Damage taxonomy: stress / impact spider-web / tempered dice | `images/charts/` |
| `anomaly_comparison/` | `panel00_{intact,impact,stress,edge,thermal,tempered}.png` | Intact vs. 5 anomaly archetypes on the *same* panel | `datasets/cracked_glass/preview_impact_stress_edge_thermal_tempered/` |
| `hero/` | `hero_cracked_panel.jpg` | Hero full-frame photorealistic shattered-panel render | `datasets/cracked_glass/smoke_test/` |
| `references/` | `impact_pexels.jpg`, `tempered_safety1.jpg`, `edge_thermal_break.jpg` | Real-world reference photos for real-vs-synthetic comparison | `assets/materials/cracked_glass/references/` |
| `textures/` | `crack_{normal,opacity,emissive}_v4.png` | PBR material maps — how cracks are rendered | `assets/materials/cracked_glass/` |
| `diversity/` | `panel_x_variant.png` | Dataset diversity: panel × crack-variant contact sheet | `datasets/cracked_glass/sweep_preview_20260418_173834/contact_sheets/` |
| `methods/` | `y_flip_explainer.png` | Appendix: occupancy-map Y-orientation convention | `images/nav_debug/` |
