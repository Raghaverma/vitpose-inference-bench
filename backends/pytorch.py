#!/usr/bin/env python3
"""PyTorch FP16 eager-mode backend, benchmarked with GENUINELY DISTINCT
crops per batch slot -- the fair denominator for Stage 4's ORT/TensorRT
batch-size comparisons.

    python -m backends.pytorch --batch-size B

Stage 1's original batch sweep (results/raw/stage1_report.json) used
pixel_values.repeat(B,1,1,1) -- B copies of one crop -- which is fine for
measuring PyTorch's own compute/VRAM scaling, but is the WRONG denominator
for a cross-backend speedup ratio at batch>1: dividing an ORT/TensorRT
distinct-crop latency by a PyTorch repeated-copy latency would compare two
different experiments to each other, even though every individual number
looks rigorous. This script re-measures PyTorch at each batch size using
the exact same golden/distinct_crops.npy construction ORT and TensorRT use,
so results/batch_matrix.py has one fair, same-construction number per
batch size for every backend. Stage 1's original repeat()-based numbers are
untouched -- this is a new, separate measurement, not a replacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint, time_calls


def verify_batch_correctness(model, pixel_values: torch.Tensor, dataset_index: torch.Tensor,
                              golden_slots: np.ndarray, max_abs_error_threshold: float = 0.01) -> None:
    """PyTorch eager IS the reference every other backend is checked against
    -- but Stage 1's repeat()-based sweep never actually proved PyTorch
    handles distinct-content batches correctly either (every slot's
    "correct" answer was identical, so nothing could have been caught).
    This is that check, finally, for the reference itself.

    Threshold found empirically, not guessed: an initial 0.001 threshold
    (roughly fp16 machine epsilon) failed at batch>=8 with errors around
    0.001-0.004 -- NOT cross-slot contamination, but normal floating-point
    non-associativity, since PyTorch/cuDNN select different batched-GEMM
    reduction kernels at different batch sizes than the single-sample
    kernel golden_slots was generated with. 0.01 has real margin over that
    (still 2-10x under it) while staying far tighter than the cross-backend
    thresholds (0.02 for ONNX/TensorRT) -- tight enough that a real
    cross-slot mix-up (which would substitute a different image's content
    entirely, not shift a few ULPs) would still fail loudly."""
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, dataset_index=dataset_index)
    batch_output = outputs.heatmaps.float().cpu().numpy()
    B = pixel_values.shape[0]
    results = []
    for i in range(B):
        max_abs_err = float(np.abs(batch_output[i] - golden_slots[i].astype(np.float32)).max())
        results.append(max_abs_err)
        if max_abs_err >= max_abs_error_threshold:
            print(f"[pytorch backend] SLOT {i} FAILED: max_abs_error {max_abs_err:.6f} "
                  f">= {max_abs_error_threshold}")
    failed = [i for i, e in enumerate(results) if e >= max_abs_error_threshold]
    if failed:
        raise SystemExit(f"[pytorch backend] REFUSING to benchmark batch={B}: "
                          f"{len(failed)}/{B} slot(s) diverged from their own standalone golden run "
                          f"by more than fp16-noise-level tolerance -- this would mean PyTorch eager "
                          f"itself doesn't handle a distinct-content batch consistently.")
    print(f"[pytorch backend] batch={B} cross-slot correctness: {B}/{B} slots OK "
          f"(max_abs_error range {min(results):.6f}-{max(results):.6f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output", type=Path, default=None,
                    help="Defaults to results/baseline/distinct_batch_b{B}.json.")
    args = p.parse_args()
    if args.output is None:
        args.output = REPO_ROOT / "results" / "baseline" / f"distinct_batch_b{args.batch_size}.json"
    return args


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    B = args.batch_size

    if B == 1:
        crop = np.load(args.golden_dir / "person_crop.npy").astype(np.float16)
        per_slot_dataset_index = [0]
        golden_slots = np.load(args.golden_dir / "pytorch_fp16_output.npy")
    else:
        distinct_crops = np.load(args.golden_dir / "distinct_crops.npy").astype(np.float16)
        distinct_env = json.loads((args.golden_dir / "distinct_batch_env.json").read_text())
        per_slot_dataset_index = [s["dataset_index"] for s in distinct_env["slots"][:B]]
        crop = distinct_crops[:B]
        golden_slots = np.load(args.golden_dir / "distinct_pytorch_fp16_outputs.npy")[:B]

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[pytorch backend] loading ViTPose++-L from {checkpoint} ({device}), batch={B}")
    _processor, model = load_pose_model(checkpoint, device, dtype)
    model.eval()

    pixel_values = torch.from_numpy(crop).to(device=device, dtype=dtype)
    dataset_index = torch.tensor(per_slot_dataset_index, dtype=torch.long, device=device)

    verify_batch_correctness(model, pixel_values, dataset_index, golden_slots)

    call_fn = lambda: model(pixel_values=pixel_values, dataset_index=dataset_index)
    if device.startswith("cuda"):
        for _ in range(args.warmup):
            call_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[pytorch backend] benchmarking batch={B} ({args.num_iters} iters, warmup={args.warmup})...")
    with torch.no_grad():
        stats = time_calls(call_fn, batch_size=B, device=device, num_iters=args.num_iters,
                            warmup=0 if device.startswith("cuda") else args.warmup)
    stats["peak_vram_mb"] = (torch.cuda.max_memory_allocated(device) / 2**20
                              if device.startswith("cuda") else None)
    vram_str = f"{stats['peak_vram_mb']:.0f}MB" if stats["peak_vram_mb"] else "n/a"
    print(f"[pytorch backend] mean {stats['mean_ms']:.2f}ms  p50 {stats['p50_ms']:.2f}ms  "
          f"p95 {stats['p95_ms']:.2f}ms  p99 {stats['p99_ms']:.2f}ms  fps {stats['fps']:.1f}  "
          f"peak_vram {vram_str}")

    report = {"backend": "PyTorch", "precision": "FP16", "batch": B,
               "distinct_crops_used": True, "benchmark": stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[pytorch backend] wrote {args.output}")


if __name__ == "__main__":
    main()
