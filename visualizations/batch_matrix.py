#!/usr/bin/env python3
"""Stage 4 Experiment 4A -- docs/images/throughput_vs_batch.png and
docs/images/latency_vs_batch.png: three curves (PyTorch / ONNX Runtime /
TensorRT), all measured with GENUINELY DISTINCT crops per batch slot.

    python -m visualizations.batch_matrix

Two SEPARATE figures, same reasoning as visualizations/batch_scaling.py:
latency-per-batch and throughput move in different-looking directions as
batch size grows, and a dual-axis chart would misleadingly imply a
relationship between two different y-scales.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, load_batch_matrix
from visualizations._style import AQUA, BLUE, INK_PRIMARY, INK_SECONDARY, YELLOW, apply_style, style_axes

THROUGHPUT_OUTPUT = REPO_ROOT / "docs" / "images" / "throughput_vs_batch.png"
LATENCY_OUTPUT = REPO_ROOT / "docs" / "images" / "latency_vs_batch.png"

BACKENDS = [
    ("pytorch", "PyTorch FP16", BLUE),
    ("onnxruntime", "ONNX Runtime (+ORT fusion)", AQUA),
    ("tensorrt", "TensorRT FP16", YELLOW),
]


def _series(cells: list[dict], backend: str, field: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for cell in cells:
        stats = cell.get(backend)
        if stats is not None:
            xs.append(cell["batch"])
            ys.append(stats[field])
    return xs, ys


def generate_throughput(output_path: Path = THROUGHPUT_OUTPUT) -> Path:
    apply_style()
    data = load_batch_matrix()
    cells = data["cells"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, label, color in BACKENDS:
        xs, ys = _series(cells, key, "fps")
        if xs:
            ax.plot(xs, ys, marker="o", markersize=6, color=color, linewidth=2.2, label=label, zorder=3)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5, color=color, fontweight="bold")
    style_axes(ax)
    all_batches = sorted({c["batch"] for c in cells})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("Throughput vs. Batch Size -- ViTPose++-L, NVIDIA L4, FP16")
    ax.legend(loc="upper left", fontsize=9)

    fig.text(0.5, -0.02,
              "All backends measured with genuinely distinct crops per batch slot, not repeated "
              "copies -- see golden/build_distinct_batch.py.",
              ha="center", va="top", fontsize=8, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_latency(output_path: Path = LATENCY_OUTPUT) -> Path:
    apply_style()
    data = load_batch_matrix()
    cells = data["cells"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, label, color in BACKENDS:
        xs, ys = _series(cells, key, "mean_ms")
        if xs:
            ax.plot(xs, ys, marker="o", markersize=6, color=color, linewidth=2.2, label=label, zorder=3)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5, color=color, fontweight="bold")
    style_axes(ax)
    all_batches = sorted({c["batch"] for c in cells})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency per batch (ms)")
    ax.set_title("Latency vs. Batch Size -- ViTPose++-L, NVIDIA L4, FP16")
    ax.legend(loc="upper left", fontsize=9)

    fig.text(0.5, -0.02,
              "One forward-pass call per batch -- latency is for the WHOLE batch, not per image. "
              "TensorRT's speedup over PyTorch shrinks as batch size grows (see README).",
              ha="center", va="top", fontsize=8, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate() -> list[Path]:
    return [generate_throughput(), generate_latency()]


if __name__ == "__main__":
    for path in generate():
        print(f"[batch_matrix] wrote {path}")
