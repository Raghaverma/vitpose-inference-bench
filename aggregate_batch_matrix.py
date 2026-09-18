#!/usr/bin/env python3
"""Stage 4 Experiment 4A/4B: aggregate the per-batch, per-backend benchmark
files into one results/batch_matrix.json comparison table.

    python aggregate_batch_matrix.py

Reads only files this project already wrote (backends/pytorch.py,
backends/onnxruntime.py, backends/tensorrt.py's per-batch outputs) --
invents nothing, recomputes nothing. Every cell that isn't present on disk
stays absent in the output (None), never filled with a guess.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BATCH_SIZES = [1, 2, 4, 8, 16]


def suffix(b: int) -> str:
    return "" if b == 1 else f"_b{b}"


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    cells = []
    for b in BATCH_SIZES:
        pt = load_json(REPO_ROOT / "results" / "baseline" / f"distinct_batch_b{b}.json")
        ort = load_json(REPO_ROOT / "results" / "onnx" / f"benchmark{suffix(b)}.json")
        trt = load_json(REPO_ROOT / "results" / "tensorrt" / f"fp16{suffix(b)}.json")

        pt_stats = pt["benchmark"] if pt else None
        ort_stats = ort["runs"]["ort_default_optimizations"] if ort else None
        trt_stats = trt["benchmark"] if trt else None
        trt_vram = trt["engine_metadata"]["vram"] if trt else None

        cell = {
            "batch": b,
            "pytorch": ({
                "mean_ms": pt_stats["mean_ms"], "fps": pt_stats["fps"],
                "p50_ms": pt_stats["p50_ms"], "p95_ms": pt_stats["p95_ms"], "p99_ms": pt_stats["p99_ms"],
                "peak_vram_mb": pt_stats["peak_vram_mb"],
            } if pt_stats else None),
            "onnxruntime": ({
                "mean_ms": ort_stats["mean_ms"], "fps": ort_stats["fps"],
                "p50_ms": ort_stats["p50_ms"], "p95_ms": ort_stats["p95_ms"], "p99_ms": ort_stats["p99_ms"],
                "device_vram_mb": ort_stats["device_vram_mb"],
            } if ort_stats else None),
            "tensorrt": ({
                "mean_ms": trt_stats["mean_ms"], "fps": trt_stats["fps"],
                "p50_ms": trt_stats["p50_ms"], "p95_ms": trt_stats["p95_ms"], "p99_ms": trt_stats["p99_ms"],
                "device_vram_mb": trt_stats["device_vram_mb"],
                "engine_file_size_mb": trt_vram["engine_file_size_mb"],
                "activation_workspace_mb": trt_vram["activation_workspace_mb"],
            } if trt_stats else None),
        }
        if pt_stats and trt_stats:
            cell["tensorrt_speedup_vs_pytorch"] = pt_stats["mean_ms"] / trt_stats["mean_ms"]
        if ort_stats and trt_stats:
            cell["tensorrt_speedup_vs_ort"] = ort_stats["mean_ms"] / trt_stats["mean_ms"]
        cells.append(cell)

        pt_fps = f"{pt_stats['fps']:.1f}" if pt_stats else "--"
        ort_fps = f"{ort_stats['fps']:.1f}" if ort_stats else "--"
        trt_fps = f"{trt_stats['fps']:.1f}" if trt_stats else "--"
        speedup = f"{cell.get('tensorrt_speedup_vs_pytorch', 0):.2f}x" if "tensorrt_speedup_vs_pytorch" in cell else "--"
        print(f"batch={b:2d}  PyTorch {pt_fps:>7s} FPS  ORT {ort_fps:>7s} FPS  "
              f"TensorRT {trt_fps:>7s} FPS  TRT/PyTorch {speedup}")

    output = {"note": "Every backend measured with GENUINELY DISTINCT crops per batch slot "
                        "(golden/build_distinct_batch.py), never .repeat() -- see backends/pytorch.py, "
                        "backends/onnxruntime.py, backends/tensorrt.py. Each cell was gated on a "
                        "per-slot cross-slot-correctness check before being recorded here.",
               "cells": cells}
    output_path = REPO_ROOT / "results" / "batch_matrix.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
