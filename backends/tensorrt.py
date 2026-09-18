#!/usr/bin/env python3
"""TensorRT FP16 backend for the exported ViTPose++-L engine.

    python -m backends.tensorrt

Gate D equivalent for TensorRT: benchmark the built engine against the
frozen PyTorch baseline, using the same sync-bracketed wall-clock timing
helper (baseline.time_calls) that backends/onnxruntime.py already used, so
all three backends share one timing methodology.

NOTE on invocation, same issue as backends/onnxruntime.py: this module is
named tensorrt.py, which shadows the real `tensorrt` package by filename.
Run it as `python -m backends.tensorrt`, never `python backends/tensorrt.py`
directly (which would put this file's own directory on sys.path first and
make `import tensorrt` inside it import itself).

Engines are never loaded blind: verify_manifest() below diffs the engine's
build manifest (results/tensorrt/engine_metadata.json) against the live
environment field-by-field before deserializing anything, and refuses with
the exact mismatched field named rather than surfacing TensorRT's opaque
"engine plan was not created with this version" deserialization error. An
engine that fails this check is deleted and rebuilt, never patched or
debugged in place (see conversion/build_engine.py) -- it was never meant to
be a repairable artifact, only a disposable, locked compilation of the
manifest's onnx_sha256 for one specific hardware/software environment.

Stage 4 update: `--batch-size B` (B>1) benchmarks the batch-B engine using
golden/distinct_crops.npy[:B] -- GENUINELY DISTINCT crops with a mixed
dataset_index per slot, never `.repeat()`. Before any latency number is
trusted, verify_batch_correctness() runs the full batch once and checks
EVERY slot's output against that exact slot's own standalone golden
(golden/distinct_pytorch_fp16_outputs.npy[i]) -- catching cross-batch-
element contamination (the MoE dataset_index broadcast or windowed-
attention reshape math mixing information between slots) that a
repeated-copy batch is structurally blind to. A batch size whose engine
doesn't even bind to that batch dimension, or whose output shape doesn't
match, fails loudly here rather than silently benchmarking something else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from baseline import REPO_ROOT, time_calls

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_TO_TORCH_DTYPE = {
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


def verify_manifest(metadata_path: Path) -> dict:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"\n\nMissing {metadata_path}. Build the engine first:\n\n"
            f"    python -m conversion.build_engine --mode tuned\n")
    manifest = json.loads(metadata_path.read_text())
    if not manifest.get("benchmarkable", False):
        raise SystemExit(
            f"[tensorrt backend] {metadata_path} is from a '{manifest.get('mode')}' build, "
            f"which is not benchmarkable by design (minimal tactic search -- see "
            f"conversion/build_engine.py). Rebuild with --mode tuned before benchmarking.")

    live = {
        "tensorrt_version": trt.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    }
    mismatches = [f"{k}: manifest={manifest['env'][k]!r} != running={v!r}"
                  for k, v in live.items() if manifest["env"].get(k) != v]
    if mismatches:
        raise SystemExit(
            f"[tensorrt backend] REFUSING to load {manifest['engine_path']} -- built for a "
            f"different environment than the one currently running:\n  " + "\n  ".join(mismatches) +
            f"\n\nDelete and rebuild: python -m conversion.build_engine --mode tuned")

    engine_path = Path(manifest["engine_path"])
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"[tensorrt backend] {metadata_path} references {engine_path}, which doesn't exist "
            f"locally (.engine files are gitignored, never committed -- see conversion/build_engine.py's "
            f"docstring). Rebuild: python -m conversion.build_engine --mode tuned")
    return manifest


def load_engine(engine_path: Path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {engine_path} -- likely a version/ABI mismatch "
                            f"that verify_manifest() should have already caught.")
    return engine


def io_tensor_names(engine) -> tuple[list[str], list[str]]:
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
    return inputs, outputs


def run_inference(engine, context, inputs: dict[str, torch.Tensor], stream: int) -> dict[str, torch.Tensor]:
    """inputs: {tensor_name: torch CUDA tensor}. Returns {tensor_name: torch CUDA tensor}."""
    input_names, output_names = io_tensor_names(engine)
    outputs = {}
    for name in input_names:
        context.set_tensor_address(name, inputs[name].data_ptr())
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(name)]
        outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        context.set_tensor_address(name, outputs[name].data_ptr())
    context.execute_async_v3(stream)
    torch.cuda.synchronize()
    return outputs


def verify_batch_correctness(engine, context, stream: torch.cuda.Stream, pixel_values: torch.Tensor,
                              dataset_index: torch.Tensor, golden_slots: np.ndarray,
                              max_abs_error_threshold: float = 0.02) -> list[dict]:
    """Run the full batch once and check EVERY slot's output against that
    exact slot's own standalone PyTorch golden. This is the check a
    repeated-copy batch cannot do: if slot i's output here doesn't match
    golden_slots[i], something mixed information between batch elements."""
    with torch.cuda.stream(stream):
        outputs = run_inference(engine, context,
                                 {"pixel_values": pixel_values, "dataset_index": dataset_index},
                                 stream.cuda_stream)
    batch_output = outputs["heatmaps"].float().cpu().numpy()
    B = pixel_values.shape[0]
    if batch_output.shape[0] != B:
        raise SystemExit(f"[tensorrt backend] engine returned batch dim {batch_output.shape[0]}, "
                          f"expected {B} -- refusing to benchmark a shape mismatch.")
    results = []
    for i in range(B):
        max_abs_err = float(np.abs(batch_output[i] - golden_slots[i].astype(np.float32)).max())
        ok = max_abs_err < max_abs_error_threshold
        results.append({"slot": i, "max_abs_error": max_abs_err, "ok": ok})
        if not ok:
            print(f"[tensorrt backend] SLOT {i} FAILED cross-slot correctness check: "
                  f"max_abs_error {max_abs_err:.6f} >= {max_abs_error_threshold} -- "
                  f"possible cross-batch-element contamination.")
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise SystemExit(f"[tensorrt backend] REFUSING to benchmark batch={B}: "
                          f"{len(failed)}/{B} slot(s) failed cross-slot correctness "
                          f"(see printed detail above).")
    print(f"[tensorrt backend] batch={B} cross-slot correctness: {B}/{B} slots OK "
          f"(max_abs_error range {min(r['max_abs_error'] for r in results):.6f}"
          f"-{max(r['max_abs_error'] for r in results):.6f})")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--engine-metadata", type=Path, default=None,
                    help="Defaults to results/tensorrt/engine_metadata[_b{B}].json.")
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--baseline-json", type=Path,
                    default=REPO_ROOT / "results" / "baseline" / "l4_fp16.json")
    p.add_argument("--onnx-benchmark-json", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "benchmark.json")
    p.add_argument("--output", type=Path, default=None,
                    help="Defaults to results/tensorrt/fp16[_b{B}].json.")
    args = p.parse_args()

    suffix = "" if args.batch_size == 1 else f"_b{args.batch_size}"
    if args.engine_metadata is None:
        args.engine_metadata = REPO_ROOT / "results" / "tensorrt" / f"engine_metadata{suffix}.json"
    if args.output is None:
        args.output = REPO_ROOT / "results" / "tensorrt" / f"fp16{suffix}.json"
    return args


def main() -> None:
    args = parse_args()
    manifest = verify_manifest(args.engine_metadata)
    if manifest["batch"] != args.batch_size:
        raise SystemExit(f"[tensorrt backend] {args.engine_metadata} was built for batch="
                          f"{manifest['batch']}, but --batch-size={args.batch_size} was requested.")
    engine = load_engine(Path(manifest["engine_path"]))
    context = engine.create_execution_context()
    # TensorRT warns that the default stream forces it to add its own extra
    # cudaStreamSynchronize() calls internally -- a dedicated non-default
    # stream avoids that overhead so the measured latency reflects TensorRT's
    # own execution, not an artifact of which stream we handed it.
    trt_stream = torch.cuda.Stream()

    B = args.batch_size
    if B == 1:
        person_crop = np.load(args.golden_dir / "person_crop.npy").astype(np.float16)
        pixel_values = torch.from_numpy(person_crop).cuda()
        dataset_index = torch.zeros((1,), dtype=torch.int64, device="cuda")
        golden_slots = np.load(args.golden_dir / "pytorch_fp16_output.npy")
    else:
        distinct_crops = np.load(args.golden_dir / "distinct_crops.npy").astype(np.float16)
        distinct_env = json.loads((args.golden_dir / "distinct_batch_env.json").read_text())
        per_slot_dataset_index = [s["dataset_index"] for s in distinct_env["slots"][:B]]
        pixel_values = torch.from_numpy(distinct_crops[:B]).cuda()
        dataset_index = torch.tensor(per_slot_dataset_index, dtype=torch.int64, device="cuda")
        golden_slots = np.load(args.golden_dir / "distinct_pytorch_fp16_outputs.npy")[:B]
    inputs = {"pixel_values": pixel_values, "dataset_index": dataset_index}

    verify_batch_correctness(engine, context, trt_stream, pixel_values, dataset_index, golden_slots)

    def call_fn():
        with torch.cuda.stream(trt_stream):
            run_inference(engine, context, inputs, trt_stream.cuda_stream)

    print(f"[tensorrt backend] benchmarking batch={B} ({args.num_iters} iters, warmup={args.warmup})...")
    stats = time_calls(call_fn, batch_size=manifest["batch"], device="cuda",
                        num_iters=args.num_iters, warmup=args.warmup)

    free, total = torch.cuda.mem_get_info()
    stats["device_vram_mb"] = (total - free) / 2**20
    stats["vram_measurement_method"] = "cuda.mem_get_info snapshot after warmup, not a true peak"
    print(f"[tensorrt backend] mean {stats['mean_ms']:.2f}ms  p50 {stats['p50_ms']:.2f}ms  "
          f"p95 {stats['p95_ms']:.2f}ms  p99 {stats['p99_ms']:.2f}ms  fps {stats['fps']:.1f}  "
          f"device_vram {stats['device_vram_mb']:.0f}MB")

    report = {"backend": "TensorRT", "precision": manifest["precision_requested"],
               "engine_metadata": manifest, "benchmark": stats}

    # Only compare against baselines measured at the SAME batch size -- Stage
    # 0's PyTorch baseline and Stage 2's ORT benchmark are both batch=1 only.
    # Dividing a batch=B TensorRT latency by a batch=1 PyTorch/ORT number
    # would silently conflate "TensorRT is faster" with "batching amortizes
    # launch overhead," which is a different claim (see Stage 1's batch
    # sweep: latency-per-batch already goes UP with B for every backend).
    if B == 1 and args.baseline_json.is_file():
        pt = json.loads(args.baseline_json.read_text())
        speedup = pt["mean_latency_ms"] / stats["mean_ms"]
        report["pytorch_baseline"] = {"mean_latency_ms": pt["mean_latency_ms"]}
        report["mean_latency_speedup_vs_pytorch"] = speedup
        print(f"[tensorrt backend] vs PyTorch FP16 baseline ({pt['mean_latency_ms']:.2f}ms): "
              f"{speedup:.2f}x mean-latency speedup")
    elif B > 1:
        print(f"[tensorrt backend] skipping PyTorch/ORT speedup comparison at batch={B} -- "
              f"no same-batch-size PyTorch/ORT baseline exists yet (see results/batch_matrix.json "
              f"once the full matrix is built).")
    if B == 1 and args.onnx_benchmark_json.is_file():
        ort = json.loads(args.onnx_benchmark_json.read_text())
        ort_optimized_ms = ort["runs"]["ort_default_optimizations"]["mean_ms"]
        speedup_vs_ort = ort_optimized_ms / stats["mean_ms"]
        report["onnx_ort_optimized_baseline"] = {"mean_latency_ms": ort_optimized_ms}
        report["mean_latency_speedup_vs_ort_optimized"] = speedup_vs_ort
        print(f"[tensorrt backend] vs ONNX Runtime +ORT fusion ({ort_optimized_ms:.2f}ms): "
              f"{speedup_vs_ort:.2f}x mean-latency speedup -- this is the number that answers "
              f"'what does TensorRT add on top of ONNX Runtime's own graph optimization,' "
              f"not just 'is TensorRT faster than PyTorch eager.'")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[tensorrt backend] wrote {args.output}")


if __name__ == "__main__":
    main()
