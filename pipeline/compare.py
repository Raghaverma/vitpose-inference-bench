"""Compare two pipeline runs' per-frame outputs (Stages 7-9).

A run's outputs are, per video, NaN-padded arrays over frames: n_persons, boxes (xywh),
keypoints (frame px) and scores, up to MAX_PERSONS per frame (pipeline/sync_pipeline.py
writes them). Two runs are compared where they saw the same people: frames whose selected
boxes are bit-identical. There the keypoint error is measured the Stage 6 way -- per joint, in
model-input px (source-px distance / crop height x 256), over joints the reference itself
scores above 0.3. Frames whose boxes differ are counted, not compared: their crops differ, so
a pose difference there says nothing about the pose backend.
"""
from __future__ import annotations

import numpy as np

from compare_golden import (INT8_MEDIAN_ERR_THRESHOLD_PX, INT8_P95_ERR_THRESHOLD_PX,
                            INT8_PCT_OVER_5PX_THRESHOLD, INT8_VISIBLE_SCORE)
from pipeline.cpu_stages import crop_height_px

FIELDS = ("n_detected", "n_persons", "boxes", "keypoints", "scores")

# TensorRT FP16 vs PyTorch FP16: the Stage 6 INT8 gates tightened 10x, set before the first
# full Stage 7 run rather than fitted to it. Not "no joint moves": two FP16 kernels can round
# an almost-flat two-peak heatmap to different peaks, so the tail is gated as a rate, not a max.
FP16_MEDIAN_ERR_THRESHOLD_PX = INT8_MEDIAN_ERR_THRESHOLD_PX / 10
FP16_P95_ERR_THRESHOLD_PX = INT8_P95_ERR_THRESHOLD_PX / 10
FP16_PCT_OVER_5PX_THRESHOLD = INT8_PCT_OVER_5PX_THRESHOLD / 10
GATES = {
    "fp16": [("median", FP16_MEDIAN_ERR_THRESHOLD_PX), ("p95", FP16_P95_ERR_THRESHOLD_PX),
             ("pct_over_5px", FP16_PCT_OVER_5PX_THRESHOLD)],
    "int8": [("median", INT8_MEDIAN_ERR_THRESHOLD_PX), ("p95", INT8_P95_ERR_THRESHOLD_PX),
             ("pct_over_5px", INT8_PCT_OVER_5PX_THRESHOLD)],
}


CROP_HEIGHT_BINS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 100000)]


def compare_runs(ref: dict[str, dict[str, np.ndarray]], cand: dict[str, dict[str, np.ndarray]],
                 box_tol_px: float = 0.0) -> dict:
    """ref/cand: {job_id: {field: array over frames}}. Returns agreement + keypoint error.
    box_tol_px > 0 also compares frames whose boxes differ by at most that much (same people,
    e.g. a detector batched across frames, whose convolutions round differently); their
    keypoint error then includes the effect of the slightly different crop."""
    if sorted(ref) != sorted(cand):
        raise ValueError(f"runs cover different videos: {sorted(ref)} vs {sorted(cand)}")
    errs, heights, n_frames, n_same_boxes, n_crops, score_diff = [], [], 0, 0, 0, []
    n_identical, box_diffs = 0, []
    per_video = {}
    for job in sorted(ref):
        r, c = ref[job], cand[job]
        if len(r["n_persons"]) != len(c["n_persons"]):
            raise ValueError(f"{job}: {len(r['n_persons'])} vs {len(c['n_persons'])} frames")
        box_diff = np.nan_to_num(np.abs(r["boxes"] - c["boxes"]), nan=0.0).max(axis=(1, 2))
        same_people = (r["n_persons"] == c["n_persons"]) & np.all(np.isnan(r["boxes"]) == np.isnan(c["boxes"]), axis=(1, 2))
        same = same_people & (box_diff <= box_tol_px)
        identical = same_people & (box_diff == 0)
        n_identical += int(identical.sum())
        box_diffs.append(box_diff[same_people & (r["n_persons"] > 0)])
        vid_errs = []
        for f in np.flatnonzero(same & (r["n_persons"] > 0)):
            n = int(r["n_persons"][f])
            h = crop_height_px(r["boxes"][f, :n])
            e = np.linalg.norm(c["keypoints"][f, :n] - r["keypoints"][f, :n], axis=-1) / h[:, None] * 256.0
            vis = r["scores"][f, :n] > INT8_VISIBLE_SCORE
            vid_errs.append(e[vis])
            heights.append(np.broadcast_to(h[:, None], e.shape)[vis])
            score_diff.append(np.abs(c["scores"][f, :n] - r["scores"][f, :n]).ravel())
            n_crops += n
        vid_errs = np.concatenate(vid_errs) if vid_errs else np.zeros(0)
        errs.append(vid_errs)
        n_frames += len(same)
        n_same_boxes += int(same.sum())
        per_video[job] = {"frames": int(len(same)), "frames_same_boxes": int(same.sum()),
                          "median_err_px": float(np.median(vid_errs)) if vid_errs.size else None}
    e = np.concatenate(errs)
    hh = np.concatenate(heights) if heights else np.zeros(0)
    bd = np.concatenate(box_diffs) if box_diffs else np.zeros(0)
    return {
        "frames": n_frames,
        "box_tol_px": box_tol_px,
        "frames_same_boxes": n_same_boxes,
        "frames_boxes_identical": n_identical,
        "frames_boxes_differ": n_frames - n_same_boxes,
        "box_max_abs_diff_px": {"p50": float(np.median(bd)), "p99": float(np.percentile(bd, 99)),
                                "max": float(bd.max())} if bd.size else None,
        "crops_compared": n_crops,
        "joints_scored": int(e.size),
        "visible_score": INT8_VISIBLE_SCORE,
        "keypoint_err_crop_px": {
            "median": float(np.median(e)), "mean": float(e.mean()),
            "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)), "max": float(e.max()),
            "pct_over_2px": float((e > 2).mean() * 100), "pct_over_5px": float((e > 5).mean() * 100),
        } if e.size else None,
        "score_abs_diff_mean": float(np.concatenate(score_diff).mean()) if score_diff else None,
        # Error by the height of the region each crop was warped from: a person far from the
        # camera is a few dozen px tall and upsampled ~6x into the 256x192 input.
        "by_crop_height_px": {
            f"{lo}-{hi}": {"joints": int(m.sum()), "median": float(np.median(e[m])),
                           "p95": float(np.percentile(e[m], 95)), "pct_over_5px": float((e[m] > 5).mean() * 100)}
            for lo, hi in CROP_HEIGHT_BINS for m in [(hh >= lo) & (hh < hi)] if m.any()},
        "per_video": per_video,
    }


def gate_failures(report: dict, precision: str) -> list[str]:
    kp = report["keypoint_err_crop_px"]
    if kp is None:
        return ["no frames with identical boxes to compare"]
    return [f"{k} {kp[k]:.4f} >= {limit}" for k, limit in GATES[precision] if not kp[k] < limit]


def identical(a: dict[str, dict[str, np.ndarray]], b: dict[str, dict[str, np.ndarray]]) -> bool:
    """Bit-identical outputs (NaN == NaN) -- the determinism check between two passes."""
    if sorted(a) != sorted(b):
        return False
    return all(np.array_equal(a[j][k], b[j][k], equal_nan=True) for j in a for k in FIELDS)
