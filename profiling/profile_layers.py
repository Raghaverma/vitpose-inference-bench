#!/usr/bin/env python3
"""Stage 5: per-layer GPU time from TensorRT's own IProfiler -- the ground
truth for "which layers dominate," with zero new installs.

    python -m profiling.profile_layers --batch-size 1
    python -m profiling.profile_layers --batch-size 16

`nsys`/`ncu` are not installed on this machine (would need sudo + a new
NVIDIA apt repo + a 500MB+ download -- a real infrastructure decision, not
taken here). TensorRT's own `IProfiler` interface needs none of that: it's
part of the `tensorrt` package already installed and in use since Stage 3.
Subclass it, attach via `context.profiler`, and TensorRT calls back with
every layer's own measured GPU time on every `execute_v2()` call --
real numbers from the exact validated engine, not a third-party guess.

Two honesty notes, found empirically:
  1. These are TensorRT's own POST-FUSION LAYER names (e.g. a fused
     Conv+Add+Relu, a Myelin-backend attention block, a Reformat node) --
     NOT raw SM kernel names. That finer granularity needs `ncu`'s hardware
     counters, which this script doesn't have and doesn't pretend to.
  2. Attaching a profiler forces TensorRT to insert per-layer stream
     synchronization internally, which measurably changes the engine's own
     latency versus an unprofiled run. This script's absolute numbers will
     NOT match results/tensorrt/fp16[_b{B}].json or
     profiling/results/deployment_stages_b{B}.json -- only the RELATIVE
     layer-to-layer proportions are meaningful, and the report says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from backends.tensorrt import load_engine, verify_manifest
from baseline import REPO_ROOT


class LayerTimeProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.total_ms: dict[str, float] = {}
        self.call_count: dict[str, int] = {}

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.total_ms[layer_name] = self.total_ms.get(layer_name, 0.0) + ms
        self.call_count[layer_name] = self.call_count.get(layer_name, 0) + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--num-iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "profiling" / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    B = args.batch_size
    suffix = "" if B == 1 else f"_b{B}"

    manifest = verify_manifest(REPO_ROOT / "results" / "tensorrt" / f"engine_metadata{suffix}.json")
    engine = load_engine(Path(manifest["engine_path"]))
    context = engine.create_execution_context()
    layer_profiler = LayerTimeProfiler()
    context.profiler = layer_profiler

    if B == 1:
        crop = np.load(args.golden_dir / "person_crop.npy").astype(np.float16)
        dataset_index_np = np.array([0], dtype=np.int64)
    else:
        distinct = np.load(args.golden_dir / "distinct_crops.npy").astype(np.float16)
        env = json.loads((args.golden_dir / "distinct_batch_env.json").read_text())
        crop = distinct[:B]
        dataset_index_np = np.array([s["dataset_index"] for s in env["slots"][:B]], dtype=np.int64)

    pixel_values = torch.from_numpy(crop).cuda()
    dataset_index = torch.from_numpy(dataset_index_np).cuda()

    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                     if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    tensors = {"pixel_values": pixel_values, "dataset_index": dataset_index}
    for name in input_names:
        context.set_tensor_address(name, tensors[name].data_ptr())
    output_buffers = {}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        output_buffers[name] = torch.empty(shape, dtype=torch.float16, device="cuda")
        context.set_tensor_address(name, output_buffers[name].data_ptr())

    # execute_v2's `bindings` list is a DIFFERENT binding mechanism than
    # set_tensor_address (found empirically: execute_v2(bindings=[]) raised
    # "bindings != nullptr" -- it doesn't fall back to addresses set via
    # set_tensor_address the way execute_async_v3 does). IProfiler is a
    # context-level callback, not tied to one execute method, so reuse
    # execute_async_v3 on a real stream instead, matching backends/
    # tensorrt.py's already-working binding path exactly.
    stream = torch.cuda.Stream()
    print(f"[profile_layers] warmup ({args.warmup} iters, batch={B})...")
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    # Reset counters -- warmup runs (cuDNN/cuBLAS algo selection settling)
    # shouldn't pollute the reported per-layer means.
    layer_profiler.total_ms.clear()
    layer_profiler.call_count.clear()

    print(f"[profile_layers] profiled run ({args.num_iters} iters, batch={B})...")
    with torch.cuda.stream(stream):
        for _ in range(args.num_iters):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    mean_ms = {name: total / args.num_iters for name, total in layer_profiler.total_ms.items()}
    total_mean = sum(mean_ms.values())
    ranked = sorted(mean_ms.items(), key=lambda kv: kv[1], reverse=True)

    print(f"[profile_layers] {len(ranked)} layers reported, sum of per-layer means: {total_mean:.3f}ms "
          f"(profiler overhead means this will NOT match the unprofiled engine-only latency)")
    print(f"[profile_layers] top {args.top_n} layers by mean time:")
    top_n = ranked[:args.top_n]
    for name, ms in top_n:
        calls_per_inference = layer_profiler.call_count[name] / args.num_iters
        print(f"[profile_layers]   {ms:7.4f}ms/inference  ({ms/total_mean:5.1%})  "
              f"{calls_per_inference:.1f} calls/inference  {name}")

    report = {
        "batch": B,
        "label": "TensorRT IProfiler per-LAYER time (post-fusion layer names, not raw SM kernels)",
        "caveat": "profiler attachment perturbs absolute latency -- only relative layer proportions "
                   "are meaningful; NOT comparable to results/tensorrt/fp16 or deployment_stages numbers",
        "num_iters": args.num_iters,
        "total_layers": len(ranked),
        "sum_of_layer_means_ms": total_mean,
        "top_layers": [{"name": name, "mean_ms_per_inference": ms, "share_of_total": ms / total_mean,
                          "calls_per_inference": layer_profiler.call_count[name] / args.num_iters}
                        for name, ms in ranked],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"layer_profile_b{B}.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"[profile_layers] wrote {output_path}")


if __name__ == "__main__":
    main()
