#!/usr/bin/env python3
"""Stage 5: the H2D / TensorRT-execution / D2H / postprocess breakdown --
measured honestly, on a call that actually contains all four stages.

    python -m profiling.profile_deployment_stages --batch-size 1
    python -m profiling.profile_deployment_stages --batch-size 16

This is NOT the same measurement as results/tensorrt/fp16[_b{B}].json.
Checked empirically before writing a line of this script: backends/
tensorrt.py's benchmarked call (`run_inference`) takes an input tensor
that's already GPU-resident (built once, before the timed loop) and
returns output that stays on the GPU -- it does zero host<->device copies
and zero postprocessing. Bracketing "H2D copy" / "D2H copy" / "postprocess"
NVTX-style ranges around that existing loop would make three of the four
stages read ~0ms at every batch size and explain nothing, while an
additivity check against it would pass trivially (the timed region never
changed). That's a real result -- it's exactly why Stage 4's headline
5.07ms number is an "engine-only" figure, not a deployment-realistic one --
but it means the 4-stage breakdown the spec actually wants has to be a
SEPARATE, honestly-different, explicitly-labeled measurement: a genuine
pinned-host-memory tensor is copied to the device fresh every iteration,
a genuine device-to-host copy happens after execute_async_v3, and the real
processor.post_process_pose_estimation() decode runs on the result.

Same additivity discipline as stage1_benchmark.py's run_stage_split_benchmark
(STAGE_SUM_TOLERANCE): sum(4 stage means) is checked against an
independently-measured, no-intermediate-sync wall clock for this SAME
widened call before any percentage breakdown is trusted.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from backends.tensorrt import load_engine, run_inference, verify_manifest
from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint

STAGE_SUM_TOLERANCE = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "profiling" / "results")
    return p.parse_args()


def load_batch_inputs(golden_dir: Path, B: int) -> tuple[np.ndarray, np.ndarray, list]:
    if B == 1:
        crop = np.load(golden_dir / "person_crop.npy").astype(np.float16)
        dataset_index = np.array([0], dtype=np.int64)
        env = json.loads((golden_dir / "env.json").read_text())
        box_xywh_per_slot = [env["box_xywh"]]
    else:
        distinct = np.load(golden_dir / "distinct_crops.npy").astype(np.float16)
        env = json.loads((golden_dir / "distinct_batch_env.json").read_text())
        crop = distinct[:B]
        dataset_index = np.array([s["dataset_index"] for s in env["slots"][:B]], dtype=np.int64)
        # Synthetic slots (1-15) have no real detection box -- reuse slot 0's
        # real box for all slots here, since this script times POSTPROCESS
        # COST (how long does decoding B heatmaps take), not per-slot
        # correctness (already proven in Stage 4). A wrong-but-consistent
        # box changes nothing about how long the decode math takes to run.
        golden_env = json.loads((golden_dir / "env.json").read_text())
        box_xywh_per_slot = [golden_env["box_xywh"]] * B
    return crop, dataset_index, box_xywh_per_slot


def percentiles(values_ms: list[float]) -> dict:
    n = len(values_ms)
    s = sorted(values_ms)
    return {
        "mean_ms": statistics.mean(values_ms),
        "std_ms": statistics.pstdev(values_ms),
        "p50_ms": statistics.median(values_ms),
        "p95_ms": s[max(0, int(0.95 * n) - 1)],
        "p99_ms": s[max(0, int(0.99 * n) - 1)],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def main() -> None:
    args = parse_args()
    B = args.batch_size
    suffix = "" if B == 1 else f"_b{B}"

    manifest = verify_manifest(REPO_ROOT / "results" / "tensorrt" / f"engine_metadata{suffix}.json")
    engine = load_engine(Path(manifest["engine_path"]))
    context = engine.create_execution_context()
    trt_stream = torch.cuda.Stream()

    crop, dataset_index_np, box_xywh_per_slot = load_batch_inputs(args.golden_dir, B)

    # Genuine pinned host memory -- this is what makes the H2D copy below a
    # real one (a non-pinned tensor's .cuda(non_blocking=True) silently
    # becomes a blocking copy under the hood; pinned memory is what actually
    # enables the async DMA transfer the "H2D copy" label claims to measure).
    host_input = torch.from_numpy(crop).pin_memory()
    host_dataset_index = torch.from_numpy(dataset_index_np).pin_memory()
    device_input = torch.empty_like(host_input, device="cuda")
    device_dataset_index = torch.empty_like(host_dataset_index, device="cuda")

    checkpoint = resolve_checkpoint(None)
    print(f"[profile_deployment_stages] loading processor for postprocessing (batch={B})...")
    processor, _model = load_pose_model(checkpoint, "cpu", torch.float16)

    def h2d():
        device_input.copy_(host_input, non_blocking=True)
        device_dataset_index.copy_(host_dataset_index, non_blocking=True)

    def exec_trt():
        return run_inference(engine, context, {"pixel_values": device_input,
                                                 "dataset_index": device_dataset_index},
                              trt_stream.cuda_stream)

    def postprocess(host_heatmaps: torch.Tensor):
        from types import SimpleNamespace
        outputs = SimpleNamespace(heatmaps=host_heatmaps.float())
        return processor.post_process_pose_estimation(outputs, boxes=[[b] for b in box_xywh_per_slot])

    def full_call():
        h2d()
        outputs = exec_trt()
        host_heatmaps = outputs["heatmaps"].cpu()
        return postprocess(host_heatmaps)

    print(f"[profile_deployment_stages] warmup ({args.warmup} iters)...")
    with torch.cuda.stream(trt_stream):
        for _ in range(args.warmup):
            full_call()
    torch.cuda.synchronize()

    print(f"[profile_deployment_stages] timed run, batch={B} ({args.num_iters} iters)...")
    e_h2d_start, e_h2d_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e_exec_end = torch.cuda.Event(enable_timing=True)
    e_d2h_end = torch.cuda.Event(enable_timing=True)

    per_stage = {"h2d": [], "exec": [], "d2h": [], "postprocess": []}
    for _ in range(args.num_iters):
        torch.cuda.synchronize()
        with torch.cuda.stream(trt_stream):
            e_h2d_start.record(trt_stream)
            h2d()
            e_h2d_end.record(trt_stream)
            outputs = exec_trt()
            e_exec_end.record(trt_stream)
            host_heatmaps = outputs["heatmaps"].cpu()  # D2H -- blocks until landed
            e_d2h_end.record(trt_stream)
        torch.cuda.synchronize()
        per_stage["h2d"].append(e_h2d_start.elapsed_time(e_h2d_end))
        per_stage["exec"].append(e_h2d_end.elapsed_time(e_exec_end))
        per_stage["d2h"].append(e_exec_end.elapsed_time(e_d2h_end))

        t0 = time.perf_counter()
        postprocess(host_heatmaps)
        per_stage["postprocess"].append((time.perf_counter() - t0) * 1000)

    # Independently-measured wall clock: the SAME widened call, timed as one
    # sync-bracketed block with no intermediate syncs -- this is what the
    # additivity check is actually checked against, not a re-derivation of
    # the stage sum itself.
    wall_clock_ms = []
    for _ in range(args.num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(trt_stream):
            full_call()
        torch.cuda.synchronize()
        wall_clock_ms.append((time.perf_counter() - t0) * 1000)

    stage_stats = {name: percentiles(v) for name, v in per_stage.items()}
    pipeline_sum_mean = sum(s["mean_ms"] for s in stage_stats.values())
    wall_clock_mean = statistics.mean(wall_clock_ms)
    rel_diff = abs(pipeline_sum_mean - wall_clock_mean) / wall_clock_mean

    print(f"[profile_deployment_stages] sync-additivity check: sum(stages)={pipeline_sum_mean:.3f}ms "
          f"vs wall_clock={wall_clock_mean:.3f}ms (rel diff {rel_diff:.1%}, tolerance {STAGE_SUM_TOLERANCE:.0%})")
    ok = rel_diff <= STAGE_SUM_TOLERANCE
    print("[profile_deployment_stages] " + ("OK" if ok else "FAILED -- breakdown below is NOT trustworthy"))

    for name, stats in stage_stats.items():
        share = stats["mean_ms"] / pipeline_sum_mean
        print(f"[profile_deployment_stages]   {name:12s} mean {stats['mean_ms']:7.3f}ms  "
              f"p95 {stats['p95_ms']:7.3f}ms  ({share:.1%} of sum)")

    engine_only_path = REPO_ROOT / "results" / "tensorrt" / f"fp16{suffix}.json"
    engine_only_mean = None
    if engine_only_path.is_file():
        engine_only_mean = json.loads(engine_only_path.read_text())["benchmark"]["mean_ms"]
        print(f"[profile_deployment_stages] for comparison, results/tensorrt/fp16{suffix}.json's "
              f"ENGINE-ONLY mean_ms (no H2D/D2H/postprocess in that timed region): {engine_only_mean:.3f}ms")

    report = {
        "batch": B,
        "label": "deployment-realistic (real pinned-memory H2D copy, real D2H copy, real postprocess)",
        "not_the_same_measurement_as": str(engine_only_path) if engine_only_path.is_file() else None,
        "engine_only_mean_ms_for_reference": engine_only_mean,
        "stages": stage_stats,
        "pipeline_sum_mean_ms": pipeline_sum_mean,
        "wall_clock_mean_ms": wall_clock_mean,
        "sync_additivity_rel_diff": rel_diff,
        "sync_additivity_ok": ok,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"deployment_stages_b{B}.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"[profile_deployment_stages] wrote {output_path}")


if __name__ == "__main__":
    main()
