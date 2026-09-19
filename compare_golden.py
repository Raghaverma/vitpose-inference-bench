#!/usr/bin/env python3
"""Diff a candidate backend's raw ViTPose++-L heatmap output against the
golden PyTorch fp16 reference from stage1_benchmark.py, at two levels:

  1. Raw tensor: max/mean absolute error, max relative error.
  2. Application-level: decode both heatmaps into COCO keypoints (using the
     same box the golden reference was generated with) and report the
     per-joint Euclidean pixel distance -- this is the number that actually
     answers "did the backend swap change the pose," not just "did the
     tensor change."

A candidate is any .npy file shaped like golden/pytorch_fp16_output.npy,
e.g. produced by running the same fixed crop through ONNX Runtime or
TensorRT once Stage 2/3 exist.

    python compare_golden.py --candidate results/raw/onnxrt_fp16_output.npy
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", type=Path, required=True,
                    help="Path to a candidate backend's raw heatmap .npy "
                         "(same shape/layout as golden/pytorch_fp16_output.npy).")
    p.add_argument("--candidate-name", type=str, default=None,
                    help="Label for the report, e.g. 'ONNX Runtime FP16'. Defaults to "
                         "the candidate filename.")
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None,
                    help="Optional path to write the comparison report as JSON.")
    return p.parse_args()


def decode_keypoints(processor, heatmaps: torch.Tensor, box_xywh: list[float]) -> dict:
    outputs = SimpleNamespace(heatmaps=heatmaps.float())
    poses = processor.post_process_pose_estimation(outputs, boxes=[[box_xywh]])[0]
    pose = poses[0]
    return {
        "keypoints": pose["keypoints"].cpu().tolist(),
        "scores": pose["scores"].cpu().reshape(-1).tolist(),
    }


def compare_outputs(golden: np.ndarray, candidate: np.ndarray, processor, box_xywh: list[float],
                     candidate_name: str, golden_env: dict | None = None) -> dict:
    """The actual Gate-C comparison, importable so tests/test_onnx_equivalence.py
    doesn't have to shell out to this script or duplicate the diff logic."""
    if golden.shape != candidate.shape:
        raise ValueError(f"shape mismatch: golden {golden.shape} vs candidate {candidate.shape}")

    golden_f32 = golden.astype(np.float32)
    candidate_f32 = candidate.astype(np.float32)
    abs_err = np.abs(candidate_f32 - golden_f32)
    # Heatmap values are mostly near zero away from the joint peak, where a
    # plain abs_err / |golden| blows up to a meaningless number even for a
    # negligible absolute difference -- relative error is only informative
    # where the golden value itself is non-trivial.
    rel_err_threshold = 1e-3
    significant = np.abs(golden_f32) > rel_err_threshold
    rel_err = abs_err[significant] / np.abs(golden_f32[significant])

    tensor_report = {
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "max_rel_error": float(rel_err.max()) if rel_err.size else 0.0,
        "rel_error_threshold": rel_err_threshold,
    }

    golden_pose = decode_keypoints(processor, torch.from_numpy(golden), box_xywh)
    candidate_pose = decode_keypoints(processor, torch.from_numpy(candidate), box_xywh)

    per_joint = []
    for name, (gx, gy), (cx, cy) in zip(
            COCO_KEYPOINT_NAMES, golden_pose["keypoints"], candidate_pose["keypoints"]):
        per_joint.append({
            "joint": name,
            "golden_xy": [gx, gy],
            "candidate_xy": [cx, cy],
            "euclidean_distance_px": math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2),
        })
    distances = [j["euclidean_distance_px"] for j in per_joint]

    keypoint_report = {
        "per_joint": per_joint,
        "mean_distance_px": sum(distances) / len(distances),
        # N=17 joints from one golden fixture -- np.percentile's default linear
        # interpolation is a small-N approximation of "P95," not a claim of
        # statistical robustness; reported anyway since it's what distinguishes
        # "one outlier joint" from "broadly worse," which mean/max alone can't.
        "p95_distance_px": float(np.percentile(distances, 95)),
        "max_distance_px": max(distances),
    }

    return {
        "candidate": candidate_name,
        "golden_env": golden_env,
        "tensor_diff": tensor_report,
        "keypoint_diff": keypoint_report,
    }


def main() -> None:
    args = parse_args()
    golden_env = json.loads((args.golden_dir / "env.json").read_text())
    box_xywh = golden_env["box_xywh"]

    golden = np.load(args.golden_dir / "pytorch_fp16_output.npy")
    candidate = np.load(args.candidate)
    candidate_name = args.candidate_name or args.candidate.name

    checkpoint = resolve_checkpoint(args.checkpoint)
    processor, _model = load_pose_model(checkpoint, args.device, torch.float16)

    report = compare_outputs(golden, candidate, processor, box_xywh, candidate_name, golden_env)
    tensor_report = report["tensor_diff"]
    keypoint_report = report["keypoint_diff"]
    per_joint = keypoint_report["per_joint"]

    print(f"[compare_golden] {candidate_name} vs PyTorch FP16 golden reference")
    print(f"  tensor:   max_abs {tensor_report['max_abs_error']:.6f}  "
          f"mean_abs {tensor_report['mean_abs_error']:.6f}  "
          f"max_rel {tensor_report['max_rel_error']:.6f}")
    print(f"  keypoint: mean_dist {keypoint_report['mean_distance_px']:.3f}px  "
          f"max_dist {keypoint_report['max_distance_px']:.3f}px")
    worst = max(per_joint, key=lambda j: j["euclidean_distance_px"])
    print(f"  worst joint: {worst['joint']} ({worst['euclidean_distance_px']:.3f}px)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"[compare_golden] wrote {args.output}")


if __name__ == "__main__":
    main()
