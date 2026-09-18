#!/usr/bin/env python3
"""Stage 1: isolate ViTPose++-L inference from the pipeline, benchmark it
properly, and freeze a golden numerical reference for later backends.

    python stage1_benchmark.py --image samples/sample.jpg

Reuses Stage 0's model/detector loading, detection, and pose-decoding code
from baseline.py unchanged -- this script only adds timing/golden-reference
machinery around it. It produces four things:

1. results/raw/stage1_report.json
   - end-to-end pipeline broken into YOLO / preprocess / ViTPose / postprocess
     stages (mean/p50/p95/p99), plus a sync-additivity check (per-stage sync
     timing summed vs. a single no-intermediate-sync wall-clock timing --
     catches a mis-placed synchronize() silently pushing latency into the
     wrong stage's bucket)
   - a noise-floor calibration (empty CUDA sync loop) so sub-millisecond
     stages (crop+preprocess) can be read against the instrument's own
     resolution instead of misread as real signal
   - the ViTPose-only benchmark (mean/p50/p95/p99/min/max/fps/vram) at each
     swept batch size (1/2/4/8/16), reusing baseline.benchmark() unchanged
     with pixel_values.repeat(B, 1, 1, 1) on the one real detected crop --
     this measures forward-pass compute/VRAM scaling under a repeated input,
     not a varied-input serving benchmark (see README)
2. golden/person_crop.npy, golden/pytorch_fp16_output.npy
   - the fixed model input and raw (pre-decode, pre-float-cast) fp16 heatmap
     output, proven bit-stable across --golden-runs repeated forward passes
     under pinned cuDNN determinism before being trusted and saved
3. golden/env.json
   - the exact torch/cuda/cudnn/GPU fingerprint the golden pair was generated
     under, since determinism pinning only guarantees bit-stability on a
     fixed environment -- a future backend comparison should check this
     before trusting the golden reference as ground truth
4. golden/keypoint_diff.py-compatible reference (poses saved into the report)
   for the application-level (decoded-keypoint) comparison later stages need
   on top of the raw-tensor diff.

See README.md for the full roadmap.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
import ultralytics

from baseline import (
    DEFAULT_DETECTOR,
    REPO_ROOT,
    benchmark,
    detect_person_box,
    load_pose_model,
    resolve_checkpoint,
    run_pose,
    verify_output,
)
from ultralytics import YOLO

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
STAGE_SUM_TOLERANCE = 0.10  # 10% -- see run_stage_split_benchmark()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=REPO_ROOT / "samples" / "sample.jpg")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--detector", type=str, default=str(DEFAULT_DETECTOR))
    p.add_argument("--det-conf", type=float, default=0.35)
    p.add_argument("--dataset-index", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default=None)
    p.add_argument("--num-iters", type=int, default=100,
                    help="Timed iterations for the stage-split and noise-floor benchmarks.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    p.add_argument("--golden-runs", type=int, default=5,
                    help="Repeated forward passes used to prove the golden reference is "
                         "bit-stable under pinned cuDNN determinism before it's saved.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "raw")
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--skip-golden", action="store_true")
    p.add_argument("--strict-stage-sum", action="store_true",
                    help="Raise instead of warn if the sync-additivity check exceeds "
                         f"the {STAGE_SUM_TOLERANCE:.0%} tolerance.")
    return p.parse_args()


def env_fingerprint(device: str) -> dict:
    fp = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "ultralytics": ultralytics.__version__,
    }
    if device.startswith("cuda"):
        fp["gpu_name"] = torch.cuda.get_device_name(device)
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
            fp["nvidia_driver"] = driver
        except Exception as exc:  # pragma: no cover -- best-effort only
            fp["nvidia_driver"] = f"unavailable ({exc})"
    return fp


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


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Golden numerical reference
# --------------------------------------------------------------------------

def generate_golden_reference(processor, model, image_rgb: np.ndarray, box: np.ndarray,
                               device: str, dtype: torch.dtype, dataset_index: int | None,
                               n_runs: int, golden_dir: Path) -> dict:
    """Run the pose forward pass n_runs times under pinned cuDNN determinism and
    require bit-identical (pixel_values, raw heatmap) pairs before saving them as
    ground truth. A golden reference that isn't reproducible on its own generating
    machine is worthless as a fixture for diffing future backends against."""
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    pairs = []
    try:
        for _ in range(n_runs):
            _, pixel_values, raw_heatmaps = run_pose(processor, model, image_rgb, box,
                                                       device, dtype, dataset_index)
            pairs.append((pixel_values.detach().cpu(), raw_heatmaps.detach().cpu()))
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark

    ref_input, ref_output = pairs[0]
    for input_t, output_t in pairs[1:]:
        if not torch.equal(input_t, ref_input) or not torch.equal(output_t, ref_output):
            raise SystemExit(
                "[stage1] golden reference is NOT bit-stable across "
                f"{n_runs} runs even with cudnn.deterministic=True -- treating this as "
                "a build failure rather than saving an untrustworthy fixture. Some op in "
                "the model path has real nondeterminism (a non-deterministic CUDA kernel, "
                "uninitialized memory, etc.); it needs to be found before Stage 2 can "
                "trust any diff against this reference.")

    golden_dir.mkdir(parents=True, exist_ok=True)
    np.save(golden_dir / "person_crop.npy", ref_input.numpy())
    np.save(golden_dir / "pytorch_fp16_output.npy", ref_output.numpy())
    env = env_fingerprint(device)
    env["golden_runs_verified_bit_identical"] = n_runs
    env["input_shape"] = list(ref_input.shape)
    env["output_shape"] = list(ref_output.shape)
    env["output_dtype"] = str(ref_output.dtype)
    # The box is what turns a raw heatmap back into image-space keypoints --
    # persisted here so compare_golden.py can do the application-level
    # (decoded-keypoint) comparison the raw-tensor diff alone can't give you.
    env["box_xywh"] = box[0].tolist()
    (golden_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(f"[stage1] golden reference verified bit-identical across {n_runs} runs, "
          f"saved to {golden_dir}/")
    return env


# --------------------------------------------------------------------------
# Stage-split timing (YOLO / preprocess / ViTPose / postprocess)
# --------------------------------------------------------------------------

def run_stage_split_benchmark(processor, model, detector: YOLO, image_bgr: np.ndarray,
                               image_rgb: np.ndarray, det_conf: float, device: str,
                               dtype: torch.dtype, dataset_index: int | None,
                               num_iters: int, warmup: int, strict: bool) -> dict:
    kwargs_fn = lambda batch: (
        {"dataset_index": torch.full((batch,), dataset_index, dtype=torch.long,
                                      device=device)}
        if dataset_index is not None else {}
    )

    def one_pass(with_intermediate_sync: bool):
        """Run detect -> preprocess -> pose -> postprocess once. Returns either a
        dict of per-stage ms (with_intermediate_sync=True) or a single wall-clock
        ms figure (with_intermediate_sync=False, no syncs between stages)."""
        stage_ms = {}
        sync(device)
        t0 = time.perf_counter()
        box, _speed = detect_person_box(detector, image_bgr, det_conf, device)
        if with_intermediate_sync:
            sync(device)
            t1 = time.perf_counter()
            stage_ms["yolo_detect"] = (t1 - t0) * 1000
        else:
            t1 = t0

        inputs = processor(image_rgb, boxes=[box], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        if with_intermediate_sync:
            sync(device)
            t2 = time.perf_counter()
            stage_ms["preprocess"] = (t2 - t1) * 1000
        else:
            t2 = t1

        outputs = model(pixel_values=pixel_values, **kwargs_fn(pixel_values.shape[0]))
        if with_intermediate_sync:
            sync(device)
            t3 = time.perf_counter()
            stage_ms["vitpose_inference"] = (t3 - t2) * 1000
        else:
            t3 = t2

        outputs.heatmaps = outputs.heatmaps.float()
        processor.post_process_pose_estimation(outputs, boxes=[box])
        sync(device)
        t4 = time.perf_counter()
        if with_intermediate_sync:
            stage_ms["postprocess"] = (t4 - t3) * 1000
            return stage_ms
        return (t4 - t0) * 1000

    with torch.no_grad():
        for _ in range(warmup):
            one_pass(with_intermediate_sync=True)

        per_stage = {"yolo_detect": [], "preprocess": [], "vitpose_inference": [],
                     "postprocess": []}
        for _ in range(num_iters):
            stages = one_pass(with_intermediate_sync=True)
            for k, v in stages.items():
                per_stage[k].append(v)

        wall_clock_ms = [one_pass(with_intermediate_sync=False) for _ in range(num_iters)]

    stage_stats = {name: percentiles(values) for name, values in per_stage.items()}
    pipeline_sum_mean = sum(s["mean_ms"] for s in stage_stats.values())
    wall_clock_mean = statistics.mean(wall_clock_ms)
    rel_diff = abs(pipeline_sum_mean - wall_clock_mean) / wall_clock_mean

    check_msg = (f"[stage1] sync-additivity check: sum(stage means)={pipeline_sum_mean:.3f}ms "
                 f"vs independently-measured wall clock={wall_clock_mean:.3f}ms "
                 f"(rel diff {rel_diff:.1%}, tolerance {STAGE_SUM_TOLERANCE:.0%})")
    if rel_diff > STAGE_SUM_TOLERANCE:
        msg = check_msg + " -- FAILED: a stage boundary sync is likely misplaced."
        if strict:
            raise SystemExit(msg)
        print("[stage1] WARNING: " + msg)
    else:
        print(check_msg + " -- OK")

    return {
        "stages": stage_stats,
        "pipeline_sum_mean_ms": pipeline_sum_mean,
        "wall_clock_mean_ms": wall_clock_mean,
        "wall_clock_p50_ms": statistics.median(wall_clock_ms),
        "wall_clock_p95_ms": sorted(wall_clock_ms)[max(0, int(0.95 * num_iters) - 1)],
        "sync_additivity_rel_diff": rel_diff,
        "sync_additivity_ok": rel_diff <= STAGE_SUM_TOLERANCE,
    }


# --------------------------------------------------------------------------
# Noise floor
# --------------------------------------------------------------------------

def measure_noise_floor(device: str, num_iters: int) -> dict:
    """Times an empty sync-bracketed no-op to find the harness's own measurement
    resolution -- a stage whose mean sits below a few multiples of this floor is
    being measured near the instrument's own noise, not read as clean signal."""
    if device.startswith("cuda"):
        probe = torch.zeros(1, device=device)
    latencies_ms = []
    for _ in range(num_iters):
        sync(device)
        t0 = time.perf_counter()
        if device.startswith("cuda"):
            probe.add_(0.0)
        sync(device)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return percentiles(latencies_ms)


# --------------------------------------------------------------------------
# Batch-size sweep (ViTPose-only, reusing baseline.benchmark() unchanged)
# --------------------------------------------------------------------------

def run_batch_sweep(model, pixel_values: torch.Tensor, dataset_index: int | None,
                     device: str, batch_sizes: list[int], num_iters: int,
                     warmup: int) -> list[dict]:
    """pixel_values.repeat(B, 1, 1, 1) on the one real detected+preprocessed crop
    gives every batch size a model-faithful input without re-running YOLO or the
    processor -- this measures ViTPose forward-pass compute/VRAM scaling under a
    repeated input, not a varied-input production batch (see README caveat)."""
    results = []
    for b in batch_sizes:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # bound allocator-history contamination between
                                       # batch sizes sharing this process (see README)
        batched = pixel_values.repeat(b, 1, 1, 1)
        stats = benchmark(model, batched, dataset_index, device, num_iters, warmup)
        vram_str = f"{stats['peak_vram_mb']:.0f}MB" if stats["peak_vram_mb"] else "n/a"
        print(f"[stage1] batch={b:2d}  mean {stats['mean_ms']:7.2f}ms  "
              f"p99 {stats['p99_ms']:7.2f}ms  fps {stats['fps']:7.1f}  "
              f"peak_vram {vram_str}")
        results.append(stats)
    return results


def main() -> None:
    args = parse_args()
    device = args.device
    dtype_str = args.dtype or ("fp16" if device.startswith("cuda") else "fp32")
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    dataset_index = None if args.dataset_index < 0 else args.dataset_index

    if not args.image.is_file():
        raise SystemExit(f"--image not found: {args.image}")

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[stage1] loading ViTPose++-L from {checkpoint} ({device}, {dtype_str})")
    processor, model = load_pose_model(checkpoint, device, dtype)

    detector_weights = args.detector if Path(args.detector).is_file() else "yolov8s.pt"
    detector = YOLO(detector_weights)

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    box, _speed = detect_person_box(detector, image_bgr, args.det_conf, device)
    poses, pixel_values, _raw = run_pose(processor, model, image_rgb, box, device, dtype,
                                          dataset_index)
    verify_output(poses, image_bgr.shape[:2])

    report: dict = {
        "checkpoint": checkpoint,
        "device": device,
        "dtype": dtype_str,
        "image": str(args.image),
        "env": env_fingerprint(device),
    }

    if not args.skip_golden:
        print(f"[stage1] generating golden reference ({args.golden_runs} verification runs, "
              f"pinned cudnn determinism)...")
        report["golden_env"] = generate_golden_reference(
            processor, model, image_rgb, box, device, dtype, dataset_index,
            args.golden_runs, args.golden_dir)

    print(f"[stage1] noise-floor calibration ({args.num_iters} iters)...")
    noise_floor = measure_noise_floor(device, args.num_iters)
    report["noise_floor"] = noise_floor
    print(f"[stage1] noise floor: mean {noise_floor['mean_ms']:.4f}ms  "
          f"p95 {noise_floor['p95_ms']:.4f}ms")

    print(f"[stage1] stage-split benchmark ({args.num_iters} iters, warmup={args.warmup})...")
    stage_split = run_stage_split_benchmark(
        processor, model, detector, image_bgr, image_rgb, args.det_conf, device, dtype,
        dataset_index, args.num_iters, args.warmup, args.strict_stage_sum)
    for name, stats in stage_split["stages"].items():
        floor_ratio = stats["mean_ms"] / noise_floor["mean_ms"] if noise_floor["mean_ms"] else float("inf")
        flag = "  [near noise floor]" if floor_ratio < 3 else ""
        print(f"[stage1]   {name:18s} mean {stats['mean_ms']:7.3f}ms  "
              f"p95 {stats['p95_ms']:7.3f}ms{flag}")
    report["stage_split"] = stage_split

    print(f"[stage1] ViTPose-only batch-size sweep {args.batch_sizes}...")
    report["batch_sweep"] = run_batch_sweep(
        model, pixel_values, dataset_index, device, args.batch_sizes, args.num_iters,
        args.warmup)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "stage1_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[stage1] wrote {report_path}")


if __name__ == "__main__":
    main()
