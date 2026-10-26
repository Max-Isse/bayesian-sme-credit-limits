"""Shared plot style so every figure in the project looks the same."""

import matplotlib.pyplot as plt

# Fixed colour per role. Colours follow the thing they represent, never its rank.
COLORS = {
    "hierarchical": "#2a78d6",  # blue
    "no_pooling": "#eb6834",  # orange
    "complete_pooling": "#1baf7a",  # aqua
    "truth": "#0b0b0b",
    "observed": "#e34948",  # red
    "muted": "#898781",
    "band": "#e1e0d9",
}

PRIOR_COLORS = ["#eb6834", "#eda100", "#2a78d6"]  # vague, Normal(0,1.5), chosen


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "axes.titlecolor": "#0b0b0b",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "xtick.color": "#898781",
            "ytick.color": "#898781",
            "xtick.labelcolor": "#52514e",
            "ytick.labelcolor": "#52514e",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
