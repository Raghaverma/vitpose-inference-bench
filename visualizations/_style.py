"""Shared visual language for every figure in visualizations/.

Palette is the first three slots of the dataviz skill's validated categorical
theme (blue/orange/aqua) -- documented to clear every adjacent AND all-pairs
CVD/contrast gate in both light and dark mode as a set of three, so backend
identity (PyTorch / ONNX faithful / ONNX+ORT-fusion) stays distinguishable
without relying on a legend alone. Assigned in FIXED order across every
figure -- PyTorch is always BLUE, ONNX-faithful is always ORANGE, ONNX+ORT is
always AQUA -- never re-cycled per chart, so the same color means the same
backend everywhere in the report. TensorRT (unmeasured) deliberately does NOT
get a fourth categorical hue: it's muted grey + dashed, a non-color encoding,
because it isn't a real data series yet.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# Chart chrome & ink -- light mode only (these are static PNGs committed to a
# git repo / README, not an interactive artifact with a dark-mode toggle).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS_GOOD = "#0ca30c"

# Categorical slots 1-4, fixed assignment by backend identity. The dataviz
# skill's palette validates this fixed order for ADJACENT-pair safety in bar/
# line charts specifically (the stricter all-pairs cap that rules out a 4th
# hue only applies to scatter/bubble/small-multiples forms) -- so slot 4
# (yellow) is safe to add here as long as bars stay in this slot order,
# which BACKEND_ORDER below enforces.
BLUE = "#2a78d6"       # PyTorch FP16 -- always
ORANGE = "#eb6834"     # ONNX Runtime, faithful graph -- always
AQUA = "#1baf7a"       # ONNX Runtime, ORT graph optimizations -- always
YELLOW = "#eda100"     # TensorRT FP16 -- always
UNMEASURED_GREY = "#c3c2b7"  # TensorRT INT8 / anything not yet measured -- dashed, no fill

BACKEND_ORDER = ["pytorch", "onnx_faithful", "onnx_optimized", "tensorrt_fp16"]
BACKEND_COLORS = {
    "pytorch": BLUE,
    "onnx_faithful": ORANGE,
    "onnx_optimized": AQUA,
    "tensorrt_fp16": YELLOW,
}
BACKEND_LABELS = {
    "pytorch": "PyTorch FP16",
    "onnx_faithful": "ONNX Runtime\n(faithful graph)",
    "onnx_optimized": "ONNX Runtime\n(+ORT fusion)",
    "tensorrt_fp16": "TensorRT FP16",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axes(ax, ygrid: bool = True) -> None:
    """Recessive grid/axes: hairline gridlines behind the data, muted spines,
    ticks only where the data needs them."""
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    if ygrid:
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
