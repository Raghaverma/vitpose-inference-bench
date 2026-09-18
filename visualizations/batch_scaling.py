#!/usr/bin/env python3
"""Batch-size scaling -- docs/images/batch_scaling_latency.png and
docs/images/batch_scaling_throughput.png.

    python -m visualizations.batch_scaling

Two SEPARATE figures, deliberately not subplots: latency-per-batch and
total throughput move in opposite-looking directions as batch size grows
(latency up, throughput up faster), and collapsing them into one dual-axis
chart is exactly the kind of chart that misleads by implying a relationship
between two different y-scales. Source: results/raw/stage1_report.json's
batch_sweep (ViTPose++-L only, PyTorch FP16 -- see stage1_benchmark.py's
docstring for why the batch entries are B copies of one real crop, not B
distinct people).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, assert_batch_sweep_complete, load_stage1
from visualizations._style import BLUE, INK_PRIMARY, INK_SECONDARY, apply_style, style_axes

REQUIRED_BATCH_SIZES = [1, 2, 4, 8, 16]
LATENCY_OUTPUT = REPO_ROOT / "docs" / "images" / "batch_scaling_latency.png"
THROUGHPUT_OUTPUT = REPO_ROOT / "docs" / "images" / "batch_scaling_throughput.png"


def _load_sweep():
    stage1 = load_stage1()
    return assert_batch_sweep_complete(stage1, REQUIRED_BATCH_SIZES)


def generate_latency(output_path: Path = LATENCY_OUTPUT) -> Path:
    apply_style()
    sweep = _load_sweep()
    sizes = [e["batch_size"] for e in sweep]
    latencies = [e["mean_ms"] for e in sweep]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(sizes, latencies, marker="o", markersize=7, color=BLUE, linewidth=2.2, zorder=3)
    style_axes(ax)
    for x, y in zip(sizes, latencies):
        ax.annotate(f"{y:.1f} ms", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold", color=INK_PRIMARY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency per batch (ms)")
    ax.set_title("ViTPose++-L Latency vs. Batch Size -- PyTorch FP16, NVIDIA L4")
    ax.set_ylim(0, max(latencies) * 1.25)

    fig.text(0.5, -0.02, "One forward-pass call per batch -- latency is for the WHOLE batch, not per image.",
              ha="center", va="top", fontsize=8.5, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_throughput(output_path: Path = THROUGHPUT_OUTPUT) -> Path:
    apply_style()
    sweep = _load_sweep()
    sizes = [e["batch_size"] for e in sweep]
    fps = [e["fps"] for e in sweep]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(sizes, fps, marker="o", markersize=7, color=BLUE, linewidth=2.2, zorder=3)
    style_axes(ax)
    for x, y in zip(sizes, fps):
        ax.annotate(f"{y:.0f} FPS", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold", color=INK_PRIMARY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("ViTPose++-L Throughput vs. Batch Size -- PyTorch FP16, NVIDIA L4")
    ax.set_ylim(0, max(fps) * 1.25)

    fig.text(0.5, -0.02, "Higher batch size raises per-batch latency (previous figure) but "
                          "raises total throughput faster -- the two do not move together.",
              ha="center", va="top", fontsize=8.5, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate() -> list[Path]:
    return [generate_latency(), generate_throughput()]


if __name__ == "__main__":
    for path in generate():
        print(f"[batch_scaling] wrote {path}")
