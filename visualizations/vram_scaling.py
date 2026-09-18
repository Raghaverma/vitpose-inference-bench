#!/usr/bin/env python3
"""Stage 4 Experiment 4B -- docs/images/vram_scaling.png: how TensorRT's own
VRAM breaks down across batch sizes.

    python -m visualizations.vram_scaling

Deliberately TensorRT-only, not a 3-backend comparison: PyTorch's
peak_vram_mb (torch.cuda.max_memory_allocated), ONNX Runtime's and
TensorRT's device_vram_mb (torch.cuda.mem_get_info snapshots) are three
DIFFERENT accounting regimes (see backends/onnxruntime.py and
backends/tensorrt.py's own comments) -- plotting them on one shared axis
would imply a comparability that isn't real. TensorRT's own three figures
(engine_file_size_mb, activation_workspace_mb, device_vram_mb), read from
one consistent measurement path across all 5 batch sizes, are the one
apples-to-apples-within-itself VRAM story this repo can currently tell.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT, load_batch_matrix
from visualizations._style import AQUA, BLUE, INK_SECONDARY, YELLOW, apply_style, style_axes

OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "vram_scaling.png"


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    apply_style()
    data = load_batch_matrix()
    cells = [c for c in data["cells"] if c.get("tensorrt")]
    batches = [c["batch"] for c in cells]
    engine_mb = [c["tensorrt"]["engine_file_size_mb"] for c in cells]
    workspace_mb = [c["tensorrt"]["activation_workspace_mb"] for c in cells]
    device_mb = [c["tensorrt"]["device_vram_mb"] for c in cells]

    fig, (ax_workspace, ax_device) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax_workspace.plot(batches, workspace_mb, marker="o", markersize=6, color=YELLOW,
                       linewidth=2.2, zorder=3)
    for x, y in zip(batches, workspace_mb):
        ax_workspace.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8, fontweight="bold")
    style_axes(ax_workspace)
    ax_workspace.set_xscale("log", base=2)
    ax_workspace.set_xticks(batches)
    ax_workspace.set_xticklabels([str(b) for b in batches])
    ax_workspace.set_xlabel("Batch size")
    ax_workspace.set_ylabel("Activation workspace (MB)")
    ax_workspace.set_title("Scales with batch", fontsize=11, fontweight="bold", loc="left")

    ax_device.plot(batches, [engine_mb[0]] * len(batches), marker="s", markersize=5, color=BLUE,
                    linewidth=1.6, linestyle=(0, (4, 2)), label="Engine weights (file size)", zorder=3)
    ax_device.plot(batches, device_mb, marker="o", markersize=6, color=AQUA, linewidth=2.2,
                    label="Steady-state device VRAM", zorder=3)
    for x, y in zip(batches, device_mb):
        ax_device.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8, fontweight="bold", color=AQUA)
    style_axes(ax_device)
    ax_device.set_xscale("log", base=2)
    ax_device.set_xticks(batches)
    ax_device.set_xticklabels([str(b) for b in batches])
    ax_device.set_xlabel("Batch size")
    ax_device.set_ylabel("VRAM (MB)")
    ax_device.set_title("Weights stay flat; total VRAM barely moves", fontsize=11,
                          fontweight="bold", loc="left")
    ax_device.legend(loc="center right", fontsize=8.5)

    fig.suptitle("TensorRT FP16 VRAM Breakdown vs. Batch Size -- ViTPose++-L, NVIDIA L4",
                  fontsize=13.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.04,
              "TensorRT-only: PyTorch/ONNX Runtime VRAM use different accounting regimes "
              "(see backends/*.py) and aren't plotted on this axis for that reason.\n"
              "Activation workspace (what TensorRT itself declares it needs) is the part that "
              "actually scales with batch -- engine weights and the CUDA context's fixed\n"
              "overhead dominate steady-state VRAM at every batch size tested here.",
              ha="center", va="top", fontsize=8, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[vram_scaling] wrote {path}")
