from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#6f6e6a"
GRID = "#e3e2de"
AXIS = "#b9b8b3"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = SERIES
RED = "#e34948"
VIOLET = "#4a3aa7"
NEUTRAL = "#f0efec"

SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

SEQ_CMAP = LinearSegmentedColormap.from_list("cabcast_blue", SEQUENTIAL_STEPS)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cabcast_div", ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL, "#f2a3a2", "#e34948", "#8f2322"]
)


def apply_style() -> None:
    mpl.use("Agg", force=False)
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.edgecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
        }
    )


def title_block(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, pad=32 if subtitle else 12)
    if subtitle:
        ax.text(
            0.0,
            1.014,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.5,
            color=TEXT_MUTED,
            va="bottom",
            ha="left",
        )


def caption(fig, text: str) -> None:
    fig.text(0.008, -0.012, text, fontsize=8.6, color=TEXT_MUTED, ha="left", va="top")


def finish(fig, path, caption_text: str | None = None):
    if caption_text:
        caption(fig, caption_text)
    fig.savefig(path)
    plt.close(fig)
    return path
