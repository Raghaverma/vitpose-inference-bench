#!/usr/bin/env python3
"""System architecture -- docs/images/architecture.png.

    python -m visualizations.architecture

Two parts in one diagram: the top-down pose pipeline (input -> YOLOv8s ->
crop/preprocess -> ViTPose++-L -> postprocess -> decoded skeleton), and the
optimization branch rooted at ViTPose++-L -- the single stage this whole
project targets (see docs/images/stage_breakdown.png: everything else in the
pipeline is a small, fixed cost by comparison).

PyTorch FP16, ONNX, and (once built) TensorRT FP16 are measured branches --
real numbers, read from results/*.json. TensorRT INT8 is drawn as a real
part of the plan -- box sized, positioned, and labeled -- but in a visibly
distinct dashed/unfilled style with no numeric value on it at all, because
it hasn't been measured yet. There is no code path in this script that could
put a number on that box: _data.NOT_YET_MEASURED is never passed through
_data.numeric(), so a plottable value for it literally cannot be constructed
here by accident. This is also the first real test of that design promise --
TensorRT FP16 used to be unmeasured too; adding its real number below was a
data change (loading results/tensorrt/fp16.json), not a rewrite of this
script's layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import (
    NOT_YET_MEASURED,
    REPO_ROOT,
    load_stage0,
    load_stage2,
    load_stage3_fp16,
    numeric,
)
from visualizations._diagram import Edge, Node, render_diagram
from visualizations._style import (
    AQUA,
    BLUE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORANGE,
    YELLOW,
    apply_style,
)

OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "architecture.png"


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    apply_style()
    stage0 = load_stage0()
    stage2 = load_stage2()
    runs = stage2["benchmark"]["runs"]
    stage3 = load_stage3_fp16()

    pytorch_ms = numeric(stage0["mean_latency_ms"], "pytorch baseline")
    faithful_ms = numeric(runs["unoptimized_graph"]["mean_ms"], "onnx faithful")
    optimized_ms = numeric(runs["ort_default_optimizations"]["mean_ms"], "onnx optimized")
    faithful_speedup = pytorch_ms / faithful_ms
    optimized_speedup = pytorch_ms / optimized_ms

    if stage3 is not None:
        trt_fp16_ms = numeric(stage3["benchmark"]["mean_ms"], "tensorrt fp16")
        trt_fp16_speedup = pytorch_ms / trt_fp16_ms
        trt_fp16_node = Node("trt_fp16", "TensorRT FP16", rank=7, lane=-0.9, style="measured",
                              color=YELLOW, sublabel=f"{trt_fp16_ms:.2f} ms -- {trt_fp16_speedup:.2f}x")
        trt_router_label = "TensorRT"
        edge_to_fp16_style = "default"
    else:
        # TensorRT FP16 genuinely does not exist yet -- NOT_YET_MEASURED is
        # never passed through numeric(), so there is no value to put here.
        _unused = NOT_YET_MEASURED
        trt_fp16_node = Node("trt_fp16", "TensorRT FP16", rank=7, lane=-0.9, style="unmeasured")
        trt_router_label = "NEXT: TensorRT"
        edge_to_fp16_style = "unmeasured"

    nodes = [
        # Main top-down pose pipeline (the spine).
        Node("input", "Input image", rank=0, lane=0),
        Node("yolo", "YOLOv8s\nperson detection", rank=1, lane=0),
        Node("crop", "Crop +\npreprocessing", rank=2, lane=0),
        Node("vitpose", "ViTPose++-L", rank=3, lane=0, style="boundary", box_w=2.1),
        Node("postprocess", "Postprocessing", rank=4, lane=0),
        Node("skeleton", "Decoded\nskeleton", rank=5, lane=0),

        # Optimization branch, rooted at ViTPose++-L.
        Node("pytorch", "PyTorch FP16", rank=4, lane=1.6, style="measured", color=BLUE,
             sublabel=f"{pytorch_ms:.2f} ms -- BASELINE"),
        Node("onnx", "ONNX", rank=4, lane=-1.6),
        Node("onnx_faithful", "Faithful graph", rank=5, lane=-0.9, style="measured", color=ORANGE,
             sublabel=f"{faithful_ms:.2f} ms -- {faithful_speedup:.2f}x"),
        Node("onnx_optimized", "ORT Optimized", rank=5, lane=-2.3, style="measured", color=AQUA,
             sublabel=f"{optimized_ms:.2f} ms -- {optimized_speedup:.2f}x"),
        Node("next_trt", trt_router_label, rank=6, lane=-1.6, box_w=2.1,
             style="default" if stage3 is not None else "unmeasured"),
        trt_fp16_node,
        Node("trt_int8", "TensorRT INT8", rank=7, lane=-2.3, style="unmeasured"),
    ]
    edges = [
        Edge("input", "yolo"), Edge("yolo", "crop"), Edge("crop", "vitpose"),
        Edge("vitpose", "postprocess"), Edge("postprocess", "skeleton"),
        Edge("vitpose", "pytorch"), Edge("vitpose", "onnx"),
        Edge("onnx", "onnx_faithful"), Edge("onnx", "onnx_optimized"),
        Edge("onnx_faithful", "next_trt"), Edge("onnx_optimized", "next_trt"),
        Edge("next_trt", "trt_fp16", style=edge_to_fp16_style),
        Edge("next_trt", "trt_int8", style="unmeasured"),
    ]

    fig, ax = plt.subplots(figsize=(17, 8.5))
    render_diagram(ax, nodes, edges, dx=2.5, dy=1.05)

    # Optimization-boundary callout on the ViTPose++-L box.
    vx, vy = 3 * 2.5, 0
    ax.annotate("Optimization\nboundary", xy=(vx, vy + 0.32), xytext=(vx - 0.3, vy + 1.55),
                fontsize=9, ha="center", color=INK_SECONDARY, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=1.0,
                                 linestyle=(0, (2, 2))))

    ax.set_title("ViTPose++-L Inference Optimization -- System Architecture & Progress",
                  fontsize=15, fontweight="bold", color=INK_PRIMARY, pad=18)

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, markersize=11,
                    label="Measured -- PyTorch FP16"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markersize=11,
                    label="Measured -- ONNX faithful graph"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=AQUA, markersize=11,
                    label="Measured -- ONNX + ORT fusion"),
    ]
    if stage3 is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=YELLOW, markersize=11,
                        label="Measured -- TensorRT FP16"))
    legend_elements.append(
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
                    markeredgecolor=INK_MUTED, markersize=11, linestyle="none",
                    label="Not yet measured -- TensorRT INT8"))
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9.5, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[architecture] wrote {path}")
