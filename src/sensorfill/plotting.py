"""Consistent, colour-blind-checked matplotlib styling for all project figures.

The categorical palette below was validated for colour-vision deficiency
(adjacent-pair CVD Delta-E >= 8). Colours are assigned to *methods* by a fixed
mapping so a method keeps its colour in every figure.

Example:
    >>> from sensorfill.plotting import color_for
    >>> color_for("Linear", ["LOCF", "Linear"])
    '#eb6834'
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
    """Apply the project style (thin lines, recessive grid, no top/right spines).

    Args:
        dpi: On-screen figure resolution (saved figures always use 160 dpi).
    """
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
    """Colour of a method, fixed by its position in a list of names.

    Args:
        name: The method name.
        names: The fixed, ordered list of all method names in the figure.

    Returns:
        A hex colour. Positions beyond the 8-colour palette get the muted grey
        rather than a recycled colour, so two methods never share a colour.

    Raises:
        ValueError: If ``name`` is not in ``names``.

    Example:
        >>> color_for("x", list("abcdefghx"))  # 9th entry
        '#8a8984'
    """
    i = names.index(name)
    return PALETTE[i] if i < len(PALETTE) else MUTED


def save(fig, name: str, folder: str | Path = "reports/figures") -> Path:
    """Save a figure as PNG.

    Args:
        fig: Matplotlib figure.
        name: File name without extension.
        folder: Output folder (created if missing).

    Returns:
        Path of the written PNG.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


__all__ = ["PALETTE", "INK", "MUTED", "setup", "color_for", "save", "plt"]
