"""Consistent, colour-blind-checked matplotlib styling for all project figures.

The categorical palette below was validated for colour-vision deficiency
(adjacent-pair CVD Delta-E >= 8). Colours are assigned to *methods* by a fixed
mapping so a method keeps its colour in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: Neutral ink for truth/observed data, and a muted grey for context.
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"


def setup(dpi: int = 110) -> None:
    """Apply the project style (thin lines, recessive grid, no top/right spines)."""
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "font.size": 10,
        }
    )


def color_for(name: str, names: list[str]) -> str:
    """Colour of ``name`` given the fixed list ``names`` (never cycles past 8)."""
    i = names.index(name)
    return PALETTE[i] if i < len(PALETTE) else MUTED


def save(fig, name: str, folder: str | Path = "reports/figures") -> Path:
    """Save a figure as PNG in ``reports/figures`` and return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


__all__ = ["PALETTE", "INK", "MUTED", "setup", "color_for", "save", "plt"]
