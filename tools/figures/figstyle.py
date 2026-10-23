#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared style and palette for all paper figures.

Usage in a figure script:

    from figstyle import apply, PALETTE
    apply()

Style: scienceplots "science" (journal) with sans-serif, a light grid,
spines without top/right, and uniform font sizes.
Palette: Okabe--Ito (colorblind-safe), consistent across all figures.
"""
import matplotlib.pyplot as plt

# Okabe--Ito, colorblind-safe
PALETTE = {
    "ours":       "#D55E00",  # LawKAN (ours)
    "base":       "#0072B2",  # main opponent
    "base2":      "#009E73",  # second opponent / structured
    "grey":       "#8C8C8C",  # neutral / dense / small MLP
    "nodeonly":   "#E69F00",  # node-only
    "integrator": "#000000",  # training-free integrator
}


def apply():
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.95",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "savefig.bbox": "tight",
    })


# Figures are authored at their final physical width (COLUMN_W inches = \columnwidth),
# so the point sizes below are the *nominal* sizes. In LaTeX each figure is then placed
# at DISPLAY_SCALE * \columnwidth, which scales everything uniformly, so every figure
# still shows identical text/marker sizes (nominal * DISPLAY_SCALE on the page).
COLUMN_W = 6.5
DISPLAY_SCALE = 0.75

