# Rebuilding the paper figures (tools/figures)

The paper has 15 figures: Figure 1 and Figures A10-A11 are TikZ drawings; the rest are
rendered to vector PDF by the `fig*.py` scripts from the experiment dumps in
`repro/json/`, resolved through `repro/canonical.json`. No new experiments are run.

| Script | Inputs (canonical blocks) | Paper figure |
|---|---|---|
| `fig00_readme_summary.py` | `x7bench`, `x35`, `x36seq`, `mk1` | README summary (not a paper figure) |
| `fig02_crossgroup.py` | `e16v2`, `e22` | Figure A1 |
| `fig04_matrix_pareto.py` | `run_main` (E1), `e38` | Figure 2 |
| `fig05_sample_efficiency.py` | `run_main` (E4), `e39` | Figure A3 |
| `fig08_imu.py` | `x5` | Figure A5 |
| `fig09_node_capacity.py` | `run_main` (E13), `e31`, `e31b`, `e31c`, `e32`, `e33`, `e40` | Figure A2 |
| `fig10_x10_platforms.py` | `x10` | Figure A6 |
| `fig12_noise_extrap.py` | `run_main` (E3.2, E3.4) | Figure A4 |
| `fig13_learned_law.py` | `x7bench`, `x35`, `x22`, `x13`, `x14`, `g2bc_fig`, `s3`, `x34`, `s1`, `x32` | Figure 3 |
| `fig14_law_diagnostics.py` | `x16iso`, `s1`, `s2`, `x11` | Figure A9 |
| `fig15_new_platforms.py` | `mk1`, `rc1` | Figure A8 |
| `fig16_kinova_seeds.py` | `x32` | Figure A7 |
| `fig17_convergence.py` | `x46` | Figure 4 |
| `figstyle.py` | — | shared style/palette (helper) |

Stack: Python + matplotlib + SciencePlots (`science`, `no-latex`), PDF output only.

```bash
python tools/figures/fig13_learned_law.py   # example
```

Output: PDFs next to the manuscript when its `figures/` directory is present on
disk, otherwise into `figures/` in this repository. `figstyle.py` holds the shared
style (single font, size, Okabe--Ito palette).
