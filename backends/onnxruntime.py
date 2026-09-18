#!/usr/bin/env python3
"""ONNX Runtime CUDA EP backend for the exported ViTPose++-L graph.

    python -m backends.onnxruntime

Gate D: benchmark ONNX Runtime's CUDAExecutionProvider against the frozen
PyTorch FP16 baseline (results/baseline/l4_fp16.json), using the exact same
sync-bracketed wall-clock timing helper (baseline.time_calls) so both
numbers come from one methodology, not two similar-looking ones -- see
time_calls()'s docstring for why that's a raw-wall-clock helper and not
paired torch.cuda.Event timestamps.

NOTE on invocation: this module is named onnxruntime.py, which shadows the
real `onnxruntime` package by filename. Run it as `python -m
backends.onnxruntime` (absolute imports resolve correctly as a package
module) -- running it directly as `python backends/onnxruntime.py` puts
this file's own directory on sys.path first and `import onnxruntime` inside
it would import itself instead of the real package.

Stage 4 update: `--batch-size B` (B>1) benchmarks the batch-B static export
(results/onnx/vitpose_plus_l_b{B}.onnx, built by conversion/export_onnx.py
--batch-size B from GENUINELY DISTINCT crops -- see
golden/build_distinct_batch.py). Before any latency number is trusted,
verify_batch_correctness() checks every slot's output against that exact
slot's own standalone golden -- catching the same cross-batch-element
contamination risk backends/tensorrt.py's identical check guards against.
Dynamic batch itself was never fixed (torch.export still specializes the
batch dim to a constant inside the backbone's windowed attention -- see
conversion/export_onnx.py's docstring); this works around that with 5
separate static per-batch exports instead, same as the TensorRT engines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from baseline import REPO_ROOT, time_calls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-path", type=Path, default=None,
                    help="Defaults to results/onnx/vitpose_plus_l[_b{B}].onnx.")
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--baseline-json", type=Path,
                    default=REPO_ROOT / "results" / "baseline" / "l4_fp16.json")
    p.add_argument("--output", type=Path, default=None,
                    help="Defaults to results/onnx/benchmark[_b{B}].json.")
    args = p.parse_args()

    suffix = "" if args.batch_size == 1 else f"_b{args.batch_size}"
    if args.onnx_path is None:
        args.onnx_path = REPO_ROOT / "results" / "onnx" / f"vitpose_plus_l{suffix}.onnx"
    if args.output is None:
        args.output = REPO_ROOT / "results" / "onnx" / f"benchmark{suffix}.json"
    return args


def verify_batch_correctness(session: ort.InferenceSession, pixel_values: np.ndarray,
                              dataset_index: np.ndarray, golden_slots: np.ndarray,
                              max_abs_error_threshold: float = 0.02) -> None:
    """Run the full batch once (plain session.run(), not the IOBinding perf
    path) and check EVERY slot's output against that exact slot's own
    standalone PyTorch golden -- the check a repeated-copy batch cannot do."""
    outputs = session.run(["heatmaps"], {"pixel_values": pixel_values, "dataset_index": dataset_index})
    batch_output = outputs[0].astype(np.float32)
    B = pixel_values.shape[0]
    if batch_output.shape[0] != B:
        raise SystemExit(f"[onnxruntime backend] session returned batch dim "
                          f"{batch_output.shape[0]}, expected {B}.")
    results = []
    for i in range(B):
        max_abs_err = float(np.abs(batch_output[i] - golden_slots[i].astype(np.float32)).max())
        results.append(max_abs_err)
        if max_abs_err >= max_abs_error_threshold:
            print(f"[onnxruntime backend] SLOT {i} FAILED cross-slot correctness check: "
                  f"max_abs_error {max_abs_err:.6f} >= {max_abs_error_threshold}")
    failed = [i for i, e in enumerate(results) if e >= max_abs_error_threshold]
    if failed:
        raise SystemExit(f"[onnxruntime backend] REFUSING to benchmark batch={B}: "
                          f"{len(failed)}/{B} slot(s) failed cross-slot correctness.")
    print(f"[onnxruntime backend] batch={B} cross-slot correctness: {B}/{B} slots OK "
          f"(max_abs_error range {min(results):.6f}-{max(results):.6f})")


def make_session(onnx_path: Path, graph_optimize: bool) -> ort.InferenceSession:
    # ort.InferenceSession applies its OWN session-level graph optimizations
    # (node fusion, constant folding) by default -- ORT_ENABLE_ALL -- which is
    # a completely separate knob from conversion/export_onnx.py's
    # OnnxConfig(optimize=False). Leaving ORT's default on would silently mix
    # "raw engine overhead vs PyTorch eager" with "gains from ORT's own graph
    # fusion" into one number, defeating the export's whole point of isolating
    # transformations one at a time. Default here matches that: disabled,
    # so this first Gate D number is the faithful-graph-vs-PyTorch-eager
    # comparison the earlier gates were built around.
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if graph_optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
    actual = session.get_providers()
    if actual[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider not active (got {actual}) -- "
                            f"benchmarking on CPU would not be a fair comparison "
                            f"against the PyTorch CUDA baseline.")
    return session


def make_io_binding(session: ort.InferenceSession, pixel_values: np.ndarray,
                     dataset_index: np.ndarray, device_id: int = 0) -> ort.IOBinding:
    """GPU-resident inputs bound once and reused across iterations, mirroring
    that pixel_values is already GPU-resident when PyTorch's benchmark() times
    it -- without this, ONNX Runtime would additionally pay a host-to-device
    copy every timed iteration that the PyTorch path never pays, understating
    ONNX Runtime's real inference-only latency."""
    binding = session.io_binding()
    pv_ort = ort.OrtValue.ortvalue_from_numpy(pixel_values, "cuda", device_id)
    di_ort = ort.OrtValue.ortvalue_from_numpy(dataset_index, "cuda", device_id)
    binding.bind_ortvalue_input("pixel_values", pv_ort)
    binding.bind_ortvalue_input("dataset_index", di_ort)
    binding.bind_output("heatmaps", "cuda", device_id)
    # keep references alive -- OrtValue GPU buffers must outlive the session.run calls
    binding._keep_alive = (pv_ort, di_ort)
    return binding


def run_onnxruntime_benchmark(onnx_path: Path, pixel_values: np.ndarray,
                               dataset_index: np.ndarray, num_iters: int,
                               warmup: int, graph_optimize: bool = False) -> dict:
    session = make_session(onnx_path, graph_optimize)
    binding = make_io_binding(session, pixel_values, dataset_index)

    call_fn = lambda: session.run_with_iobinding(binding)
    stats = time_calls(call_fn, batch_size=int(pixel_values.shape[0]), device="cuda",
                        num_iters=num_iters, warmup=warmup)

    # Sampled AFTER warmup + the timed loop above, i.e. once the session's CUDA
    # arena has already reached its steady-state footprint -- this is a
    # "resident at measurement time" figure, not a peak-during-execution one.
    # ONNX Runtime's CUDA arena allocates via cudaMalloc directly, NOT through
    # PyTorch's caching allocator, so torch.cuda.max_memory_allocated() (what
    # baseline.benchmark() uses for the PyTorch row) would report ~0 for this
    # session regardless of its real footprint. mem_get_info() asks the driver
    # for actual device memory in use, which is the only accounting regime
    # that sees both allocators -- but it is NOT directly comparable to
    # PyTorch's peak_vram_mb (different accounting regime, see module docstring).
    _free, total = torch.cuda.mem_get_info()
    stats["device_vram_mb"] = (total - _free) / 2**20
    stats["vram_measurement_method"] = "cuda.mem_get_info snapshot after warmup, not a true peak -- see comment"
    stats["execution_provider"] = session.get_providers()[0]
    return stats


def main() -> None:
    args = parse_args()
    B = args.batch_size
    if B == 1:
        person_crop = np.load(args.golden_dir / "person_crop.npy").astype(np.float16)
        pixel_values = person_crop
        dataset_index = np.array([0], dtype=np.int64)
        golden_slots = np.load(args.golden_dir / "pytorch_fp16_output.npy")
    else:
        distinct_crops = np.load(args.golden_dir / "distinct_crops.npy").astype(np.float16)
        distinct_env = json.loads((args.golden_dir / "distinct_batch_env.json").read_text())
        per_slot_dataset_index = [s["dataset_index"] for s in distinct_env["slots"][:B]]
        pixel_values = distinct_crops[:B]
        dataset_index = np.array(per_slot_dataset_index, dtype=np.int64)
        golden_slots = np.load(args.golden_dir / "distinct_pytorch_fp16_outputs.npy")[:B]

    print(f"[onnxruntime backend] session providers available: {ort.get_available_providers()}")
    pt = json.loads(args.baseline_json.read_text()) if (B == 1 and args.baseline_json.is_file()) else None

    runs = {}
    for label, graph_optimize in [("unoptimized_graph", False), ("ort_default_optimizations", True)]:
        print(f"[onnxruntime backend] benchmarking {args.onnx_path.name} "
              f"(graph_optimize={graph_optimize}, {args.num_iters} iters, warmup={args.warmup})...")
        session = make_session(args.onnx_path, graph_optimize)
        verify_batch_correctness(session, pixel_values, dataset_index, golden_slots)
        del session  # run_onnxruntime_benchmark() creates its own -- don't hold two
                      # sessions' CUDA arenas alive at once, or the VRAM figure below
                      # measures "two sessions' footprint," not one.
        stats = run_onnxruntime_benchmark(args.onnx_path, pixel_values, dataset_index,
                                           args.num_iters, args.warmup, graph_optimize)
        print(f"[onnxruntime backend]   mean {stats['mean_ms']:.2f}ms  "
              f"p50 {stats['p50_ms']:.2f}ms  p95 {stats['p95_ms']:.2f}ms  "
              f"p99 {stats['p99_ms']:.2f}ms  fps {stats['fps']:.1f}  "
              f"device_vram {stats['device_vram_mb']:.0f}MB")
        if pt is not None:
            stats["mean_latency_speedup_vs_pytorch"] = pt["mean_latency_ms"] / stats["mean_ms"]
            print(f"[onnxruntime backend]   vs PyTorch FP16 baseline ({pt['mean_latency_ms']:.2f}ms): "
                  f"{stats['mean_latency_speedup_vs_pytorch']:.2f}x mean-latency speedup")
        elif B > 1:
            print(f"[onnxruntime backend]   skipping PyTorch speedup comparison at batch={B} -- "
                  f"no same-batch-size PyTorch baseline exists yet.")
        runs[label] = stats

    print("[onnxruntime backend] the two runs above isolate ORT's own graph fusion from raw "
          "engine overhead -- 'unoptimized_graph' is the one comparable to the faithful export "
          "Gates A-C validated; 'ort_default_optimizations' is what a real deployment would "
          "actually run. VRAM figures use a different accounting regime than PyTorch's "
          "peak_vram_mb -- don't diff them directly (see run_onnxruntime_benchmark's comment).")

    report = {
        "backend": "ONNX Runtime",
        "onnxruntime_version": ort.__version__,
        "precision": "FP16",
        "onnx_path": str(args.onnx_path),
        "runs": runs,
        "pytorch_baseline": ({
            "mean_latency_ms": pt["mean_latency_ms"],
            "throughput_fps": pt["throughput_fps"],
            "peak_vram_mb": pt["peak_vram_mb"],
        } if pt is not None else None),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[onnxruntime backend] wrote {args.output}")


if __name__ == "__main__":
    main()
