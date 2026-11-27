"""Shared figure helpers (spec §8.11). Plan 3 adds the evaluation and event plots.

Two rules for every figure in the project:
  - inline in notebooks at 100 dpi (keeps the committed notebooks small);
  - saved to results/figures/<name>.png at 150 dpi (good enough for the report, still small).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from brent import paths

SAVE_DPI = 150


def setup_style() -> None:
    """One consistent look: 100-dpi inline figures, light grid, no top/right frame lines."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    })


def save_fig(fig, name: str) -> Path:
    """Save `fig` as results/figures/<name>.png at 150 dpi and return the path."""
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / f"{name}.png"
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    return out
