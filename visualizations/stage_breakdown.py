#!/usr/bin/env python3
"""Stage 1 pipeline breakdown -- docs/images/stage_breakdown.png -- and the
benchmark methodology diagram -- docs/images/methodology.png.

    python -m visualizations.stage_breakdown

Both figures are built from results/raw/stage1_report.json. They're grouped
in one script because they're both "how Stage 1 was measured" figures over
the same source data, even though the README places them in different
sections (Baseline, and Benchmark Methodology respectively).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, load_stage1
from visualizations._diagram import Edge, Node, render_diagram
from visualizations._style import BASELINE, BLUE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, apply_style, style_axes
from stage1_benchmark import STAGE_SUM_TOLERANCE

STAGE_ORDER = ["yolo_detect", "preprocess", "vitpose_inference", "postprocess"]
STAGE_LABELS = {
    "yolo_detect": "YOLOv8s\ndetection",
    "preprocess": "Crop +\npreprocess",
    "vitpose_inference": "ViTPose++-L\ninference",
    "postprocess": "Postprocessing",
}

BREAKDOWN_OUTPUT = REPO_ROOT / "docs" / "images" / "stage_breakdown.png"
METHODOLOGY_OUTPUT = REPO_ROOT / "docs" / "images" / "methodology.png"


def generate_stage_breakdown(output_path: Path = BREAKDOWN_OUTPUT) -> Path:
    apply_style()
    stage1 = load_stage1()
    stages = stage1["stage_split"]["stages"]

    labels = [STAGE_LABELS[k] for k in STAGE_ORDER]
    values = [stages[k]["mean_ms"] for k in STAGE_ORDER]
    colors = [BLUE if k == "vitpose_inference" else "#cfd3ce" for k in STAGE_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    style_axes(ax)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=INK_PRIMARY)

    total = sum(values)
    vitpose_share = stages["vitpose_inference"]["mean_ms"] / total
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Stage 1 End-to-End Latency Breakdown -- NVIDIA L4")
    ax.set_ylim(0, max(values) * 1.25)
    fig.text(0.5, -0.02,
              f"ViTPose++-L accounts for {vitpose_share:.0%} of the measured single-image, "
              f"batch=1 pipeline latency ({total:.1f} ms total) -- the dominant stage, and the "
              f"one this project's optimization work targets.",
              ha="center", va="top", fontsize=8.5, color=INK_SECONDARY, wrap=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_methodology(output_path: Path = METHODOLOGY_OUTPUT) -> Path:
    apply_style()
    stage1 = load_stage1()
    noise_floor_ms = stage1["noise_floor"]["mean_ms"]
    sync_drift = stage1["stage_split"]["sync_additivity_rel_diff"]

    steps = ["Model\nload", "CUDA\ninit", "Warmup", "GPU\nsync", "Timed\niterations",
             "GPU\nsync", "P50/P95/P99\n/ mean", "VRAM\nmeasurement"]
    nodes = [Node(id=f"s{i}", label=label, rank=i, lane=0, style="default")
             for i, label in enumerate(steps)]
    nodes[4].sublabel = f"noise floor {noise_floor_ms:.3f} ms"
    nodes[6].sublabel = f"sync-additivity drift {sync_drift:.1%} (tol. {STAGE_SUM_TOLERANCE:.0%})"
    edges = [Edge(src=f"s{i}", dst=f"s{i+1}") for i in range(len(steps) - 1)]

    fig, ax = plt.subplots(figsize=(15.5, 3.2))
    render_diagram(ax, nodes, edges, dx=2.9, dy=1.0)
    ax.set_title("Benchmark Methodology -- Validated, Not Just a Python Timer", loc="center",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, pad=10)

    ok = stage1["stage_split"]["sync_additivity_ok"]
    assert ok, "sync-additivity check failed -- this figure's claim of a validated harness would be false"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate() -> list[Path]:
    return [generate_stage_breakdown(), generate_methodology()]


if __name__ == "__main__":
    for path in generate():
        print(f"[stage_breakdown] wrote {path}")
