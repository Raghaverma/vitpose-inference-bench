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
        int8 = load_json(REPO_ROOT / "results" / "tensorrt" / f"int8{suffix(b)}.json")

        pt_stats = pt["benchmark"] if pt else None
        ort_stats = ort["runs"]["ort_default_optimizations"] if ort else None
        trt_stats = trt["benchmark"] if trt else None
        trt_vram = trt["engine_metadata"]["vram"] if trt else None
        int8_stats = int8["benchmark"] if int8 else None
        int8_paired = int8.get("paired_fp16") if int8 else None
        int8_kp = int8["real_crop_correctness"]["keypoint_err_crop_px"] if int8 else None

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
            # Stage 6. Timed on real crops at dataset_index 0 (its validated expert); latency of
            # this graph doesn't depend on content, but its accuracy claim covers expert 0 only.
            "tensorrt_int8": ({
                "mean_ms": int8_stats["mean_ms"], "fps": int8_stats["fps"],
                "p50_ms": int8_stats["p50_ms"], "p95_ms": int8_stats["p95_ms"], "p99_ms": int8_stats["p99_ms"],
                "device_vram_mb": int8_stats["device_vram_mb"],
                "engine_file_size_mb": int8["engine_metadata"]["vram"]["engine_file_size_mb"],
                "activation_workspace_mb": int8["engine_metadata"]["vram"]["activation_workspace_mb"],
                "real_crop_median_err_px": int8_kp["median"], "real_crop_p95_err_px": int8_kp["p95"],
                "paired_speedup_vs_fp16": int8_paired["speedup_vs_fp16"] if int8_paired else None,
                "paired_fp16_median_ms": int8_paired["fp16_median_ms"] if int8_paired else None,
                "paired_int8_median_ms": int8_paired["int8_median_ms"] if int8_paired else None,
            } if int8_stats else None),
        }
        if pt_stats and trt_stats:
            cell["tensorrt_speedup_vs_pytorch"] = pt_stats["mean_ms"] / trt_stats["mean_ms"]
        if ort_stats and trt_stats:
            cell["tensorrt_speedup_vs_ort"] = ort_stats["mean_ms"] / trt_stats["mean_ms"]
        if pt_stats and int8_stats:
            cell["tensorrt_int8_speedup_vs_pytorch"] = pt_stats["mean_ms"] / int8_stats["mean_ms"]
        cells.append(cell)

        pt_fps = f"{pt_stats['fps']:.1f}" if pt_stats else "--"
        ort_fps = f"{ort_stats['fps']:.1f}" if ort_stats else "--"
        trt_fps = f"{trt_stats['fps']:.1f}" if trt_stats else "--"
        int8_fps = f"{int8_stats['fps']:.1f}" if int8_stats else "--"
        speedup = f"{cell.get('tensorrt_speedup_vs_pytorch', 0):.2f}x" if "tensorrt_speedup_vs_pytorch" in cell else "--"
        paired = (f"{int8_paired['speedup_vs_fp16']:.2f}x" if int8_paired else "--")
        print(f"batch={b:2d}  PyTorch {pt_fps:>7s} FPS  ORT {ort_fps:>7s} FPS  "
              f"TensorRT {trt_fps:>7s} FPS  TRT/PyTorch {speedup}  INT8 {int8_fps:>7s} FPS  "
              f"INT8/FP16 (paired) {paired}")

    output = {"note": "Every backend measured with GENUINELY DISTINCT crops per batch slot "
                        "(golden/build_distinct_batch.py), never .repeat() -- see backends/pytorch.py, "
                        "backends/onnxruntime.py, backends/tensorrt.py. Each cell was gated on a "
                        "per-slot cross-slot-correctness check before being recorded here. "
                        "tensorrt_int8 (Stage 6) is gated on the 400 held-out real crops instead "
                        "(calibration/real_corpus.py) and timed on real crops at dataset_index 0; "
                        "its paired_speedup_vs_fp16 is measured against the FP16 engine in the same "
                        "process, the ratio to use rather than dividing across the two columns.",
               "cells": cells}
    output_path = REPO_ROOT / "results" / "batch_matrix.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
