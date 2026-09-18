#!/usr/bin/env python3
"""PyTorch vs ONNX Runtime vs TensorRT throughput -- docs/images/throughput_comparison.png.

    python -m visualizations.throughput

Uses the FPS values already computed and stored at measurement time
(results/baseline/l4_fp16.json's throughput_fps, results/onnx/benchmark.json's
runs.*.fps, results/tensorrt/fp16.json's benchmark.fps if it exists) rather
than recomputing 1000/mean_ms from rounded latency figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, load_stage0, load_stage2, load_stage3_fp16
from visualizations._style import BACKEND_COLORS, BACKEND_LABELS, INK_PRIMARY, apply_style, style_axes

OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "throughput_comparison.png"


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    apply_style()
    stage0 = load_stage0()
    stage2 = load_stage2()
    runs = stage2["benchmark"]["runs"]
    stage3 = load_stage3_fp16()

    labels = ["pytorch", "onnx_faithful", "onnx_optimized"]
    fps_values = [
        stage0["throughput_fps"],
        runs["unoptimized_graph"]["fps"],
        runs["ort_default_optimizations"]["fps"],
    ]
    if stage3 is not None:
        labels.append("tensorrt_fp16")
        fps_values.append(stage3["benchmark"]["fps"])

    colors = [BACKEND_COLORS[k] for k in labels]
    x_labels = [BACKEND_LABELS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6.4 + (1.4 if stage3 else 0), 4.6))
    bars = ax.bar(x_labels, fps_values, color=colors, width=0.55, zorder=3)
    style_axes(ax)

    for bar, fps in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, fps + 1.5, f"{fps:.1f} FPS",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK_PRIMARY)

    ax.set_ylabel("Throughput (FPS, batch=1) -- higher is better")
    ax.set_title("Inference Backend Throughput -- ViTPose++-L, NVIDIA L4, FP16")
    ax.set_ylim(0, max(fps_values) * 1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[throughput] wrote {path}")
