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


# transformers 5.17's post_dark_unbiased_data_processing builds its flat heatmap index from
# float32 keypoint coordinates and only then casts it to int. Past 2**24 elements -- N crops x
# 17 joints x 66 x 50 padded cells, i.e. N >= 300 boxes in one call -- the index rounds and DARK
# reads the wrong neighbours: on the real eval set every crop from index 299 on moved ~4 px
# when all 400 were decoded at once. Decoding in chunks far below that is exact.
HF_DECODE_CHUNK = 128

# Stage 6 INT8 gates on the held-out real crops (calibration/real_corpus.py). Keypoint errors
# are in model-input pixels -- source-px distance / crop height * 256 -- so large and small
# crops weigh alike, and only over joints the FP16 reference itself scores above
# INT8_VISIBLE_SCORE (a joint the reference can't see has no position to preserve). First
# measured, recipe as shipped (ONNX Runtime, batch 1): median 0.32 px, p95 1.31 px, 0.31% of
# joints over 5 px, heatmap RMSE 0.0048. The replaced all-tensors recipe measured 5.7 px /
# 48 px / 57% / 0.044, so every gate below separates the two by an order of magnitude while
# leaving ~2x headroom over normal engine-to-engine and batch-size variation.
INT8_VISIBLE_SCORE = 0.3
INT8_MEDIAN_ERR_THRESHOLD_PX = 0.6
INT8_P95_ERR_THRESHOLD_PX = 2.5
INT8_PCT_OVER_5PX_THRESHOLD = 1.0
INT8_HEATMAP_RMSE_THRESHOLD = 0.01


def decode_crop_batch(processor, heatmaps: np.ndarray, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 17, 64, 48) heatmaps + one box per crop -> keypoints (N, 17, 2) in source px and
    scores (N, 17), through HF's own post-processing, HF_DECODE_CHUNK crops per call."""
    xy, scores = [], []
    for k in range(0, len(heatmaps), HF_DECODE_CHUNK):
        hm = torch.from_numpy(np.ascontiguousarray(heatmaps[k:k + HF_DECODE_CHUNK])).float()
        boxes = [[b.tolist()] for b in boxes_xywh[k:k + HF_DECODE_CHUNK]]
        poses = processor.post_process_pose_estimation(SimpleNamespace(heatmaps=hm), boxes=boxes)
        xy.append(np.stack([p[0]["keypoints"].cpu().numpy() for p in poses]))
        scores.append(np.stack([p[0]["scores"].cpu().numpy().reshape(-1) for p in poses]))
    return np.concatenate(xy), np.concatenate(scores)


def compare_crop_batch(reference: np.ndarray, candidate: np.ndarray, processor, boxes_xywh: np.ndarray,
                       crop_height_px: np.ndarray, visible_score: float = INT8_VISIBLE_SCORE) -> dict:
    """Many-crop counterpart of compare_outputs(): tensor-level and keypoint-level drift of a
    candidate backend against per-crop reference heatmaps, both decoded the same way."""
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}")
    ref32, cand32 = reference.astype(np.float32), candidate.astype(np.float32)
    ref_xy, ref_scores = decode_crop_batch(processor, reference, boxes_xywh)
    cand_xy, cand_scores = decode_crop_batch(processor, candidate, boxes_xywh)
    err = np.linalg.norm(cand_xy - ref_xy, axis=-1) / crop_height_px.reshape(-1, 1) * 256.0
    visible = ref_scores > visible_score
    e = err[visible]
    per_joint = {name: float(np.median(err[:, j][visible[:, j]])) if visible[:, j].any() else None
                 for j, name in enumerate(COCO_KEYPOINT_NAMES)}
    return {
        "num_crops": int(len(reference)),
        "num_joints_scored": int(visible.sum()),
        "visible_score": visible_score,
        "tensor_diff": {
            "rmse": float(np.sqrt(np.mean((cand32 - ref32) ** 2))),
            "max_abs_error": float(np.abs(cand32 - ref32).max()),
        },
        "keypoint_err_crop_px": {
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)),
            "max": float(e.max()),
            "pct_over_2px": float((e > 2).mean() * 100),
            "pct_over_5px": float((e > 5).mean() * 100),
        },
        "per_joint_median_err_crop_px": per_joint,
        "score_abs_diff_mean": float(np.abs(cand_scores - ref_scores).mean()),
    }


def int8_gate_failures(report: dict) -> list[str]:
    """The four Stage 6 INT8 gates, as human-readable failures (empty = pass)."""
    kp, t = report["keypoint_err_crop_px"], report["tensor_diff"]
    checks = [
        (t["rmse"] < INT8_HEATMAP_RMSE_THRESHOLD, f"heatmap RMSE {t['rmse']:.5f} >= {INT8_HEATMAP_RMSE_THRESHOLD}"),
        (kp["median"] < INT8_MEDIAN_ERR_THRESHOLD_PX,
         f"median keypoint error {kp['median']:.3f} px >= {INT8_MEDIAN_ERR_THRESHOLD_PX} px"),
        (kp["p95"] < INT8_P95_ERR_THRESHOLD_PX, f"p95 keypoint error {kp['p95']:.3f} px >= {INT8_P95_ERR_THRESHOLD_PX} px"),
        (kp["pct_over_5px"] < INT8_PCT_OVER_5PX_THRESHOLD,
         f"{kp['pct_over_5px']:.2f}% of joints moved > 5 px (limit {INT8_PCT_OVER_5PX_THRESHOLD}%)"),
    ]
    return [msg for ok, msg in checks if not ok]


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
