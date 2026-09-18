#!/usr/bin/env python3
"""PyTorch vs ONNX Runtime vs TensorRT latency -- docs/images/latency_comparison.png.

    python -m visualizations.latency

Reads results/baseline/l4_fp16.json (PyTorch), results/onnx/benchmark.json
(ONNX Runtime, both the faithful-graph and ORT-optimized runs), and
results/tensorrt/fp16.json (TensorRT FP16, if it exists -- Stage 3 is
optional for this figure; earlier stages must still render correctly before
it lands). Speedups are computed from the loaded values at plot time, never
hardcoded, so the annotation can never drift from what's actually in the JSON.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, load_stage0, load_stage2, load_stage3_fp16
from visualizations._style import BACKEND_COLORS, BACKEND_LABELS, INK_PRIMARY, apply_style, style_axes

OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "latency_comparison.png"


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    apply_style()
    stage0 = load_stage0()
    stage2 = load_stage2()
    runs = stage2["benchmark"]["runs"]
    stage3 = load_stage3_fp16()

    labels = ["pytorch", "onnx_faithful", "onnx_optimized"]
    values_ms = [
        stage0["mean_latency_ms"],
        runs["unoptimized_graph"]["mean_ms"],
        runs["ort_default_optimizations"]["mean_ms"],
    ]
    if stage3 is not None:
        labels.append("tensorrt_fp16")
        values_ms.append(stage3["benchmark"]["mean_ms"])

    colors = [BACKEND_COLORS[k] for k in labels]
    x_labels = [BACKEND_LABELS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6.4 + (1.4 if stage3 else 0), 4.6))
    bars = ax.bar(x_labels, values_ms, color=colors, width=0.55, zorder=3)
    style_axes(ax)

    for bar, ms in zip(bars, values_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, ms + 0.35, f"{ms:.2f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK_PRIMARY)

    pytorch_ms = values_ms[0]
    for bar, ms, key in zip(bars[1:], values_ms[1:], labels[1:]):
        speedup = pytorch_ms / ms
        ax.text(bar.get_x() + bar.get_width() / 2, ms / 2, f"{speedup:.2f}x",
                ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    ax.set_ylabel("Mean latency (ms) -- lower is better")
    ax.set_title("Inference Backend Latency -- ViTPose++-L, NVIDIA L4, FP16, batch=1")
    ax.set_ylim(0, max(values_ms) * 1.2)

    caption = ("\"Faithful graph\" = optimize=False export, ORT session optimizations disabled.\n"
               "\"+ORT fusion\" = ORT's own default graph optimizations enabled at session creation --\n"
               "this is a separate transformation from the ONNX export itself.")
    if stage3 is not None:
        caption += ("\nTensorRT FP16 = strongly-typed engine built from the same faithful ONNX graph "
                     "(99.4% of layers\nactually run in fp16 -- see results/tensorrt/engine_metadata.json).")
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="#52514e")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[latency] wrote {path}")
