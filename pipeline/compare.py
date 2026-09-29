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


# ---- Stage 12: a different detector -------------------------------------------------------------
# A detector change (fp16, TensorRT) moves boxes by up to ~1-2 px and can change which people are
# selected, so frames with identical boxes are rare and compare_runs() would compare almost
# nothing. Here people are matched across the two runs by box IoU instead, per frame. Gates, set
# before the first full run: (1) at least DETECTOR_MIN_PERSON_AGREEMENT of the people either run
# poses have a partner at IoU >= DETECTOR_IOU in the other (who gets posed must not change),
# and (2) on matched people, the keypoint gates for an approximate change -- Stage 6's INT8
# gates, the tolerance this repo already accepts for a precision change.
DETECTOR_IOU = 0.9
DETECTOR_MIN_PERSON_AGREEMENT = 0.99


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)


def compare_runs_matched(ref: dict[str, dict[str, np.ndarray]], cand: dict[str, dict[str, np.ndarray]],
                         iou_min: float = 0.5) -> dict:
    """Like compare_runs(), but people are paired by box IoU (greedy, highest first, >= iou_min)
    within each frame instead of requiring identical boxes. Keypoint error is measured on pairs,
    normalized by the reference person's crop height."""
    if sorted(ref) != sorted(cand):
        raise ValueError(f"runs cover different videos: {sorted(ref)} vs {sorted(cand)}")
    errs, heights, ious, shifts = [], [], [], []
    n_ref = n_cand = matched = matched_hi = frames = frames_same = 0
    for job in sorted(ref):
        r, c = ref[job], cand[job]
        if len(r["n_persons"]) != len(c["n_persons"]):
            raise ValueError(f"{job}: {len(r['n_persons'])} vs {len(c['n_persons'])} frames")
        for f in range(len(r["n_persons"])):
            nr, nc = int(r["n_persons"][f]), int(c["n_persons"][f])
            n_ref, n_cand, frames = n_ref + nr, n_cand + nc, frames + 1
            if nr == 0 or nc == 0:
                frames_same += nr == nc
                continue
            rb, cb = r["boxes"][f, :nr], c["boxes"][f, :nc]
            iou = _iou_xywh(rb, cb)
            pairs, used_r, used_c = [], set(), set()
            for k in np.argsort(-iou, axis=None):
                i, j = divmod(int(k), nc)
                if iou[i, j] < iou_min:
                    break
                if i in used_r or j in used_c:
                    continue
                used_r.add(i)
                used_c.add(j)
                pairs.append((i, j))
            hi = sum(iou[i, j] >= DETECTOR_IOU for i, j in pairs)
            matched, matched_hi = matched + len(pairs), matched_hi + hi
            frames_same += nr == nc and hi == nr
            if not pairs:
                continue
            ii, jj = np.array(pairs).T
            h = crop_height_px(rb[ii])
            e = np.linalg.norm(c["keypoints"][f, jj] - r["keypoints"][f, ii], axis=-1) / h[:, None] * 256.0
            vis = r["scores"][f, ii] > INT8_VISIBLE_SCORE
            errs.append(e[vis])
            heights.append(np.broadcast_to(h[:, None], e.shape)[vis])
            ious.append(iou[ii, jj])
            rc = rb[ii, :2] + rb[ii, 2:] / 2
            cc = cb[jj, :2] + cb[jj, 2:] / 2
            shifts.append(np.linalg.norm(cc - rc, axis=1))
    e = np.concatenate(errs) if errs else np.zeros(0)
    hh = np.concatenate(heights) if heights else np.zeros(0)
    io = np.concatenate(ious) if ious else np.zeros(0)
    sh = np.concatenate(shifts) if shifts else np.zeros(0)
    return {
        "frames": frames, "frames_same_people": frames_same,
        "persons_ref": n_ref, "persons_cand": n_cand, "persons_matched": matched,
        f"persons_matched_iou{DETECTOR_IOU:g}": matched_hi,
        "person_agreement": min(matched_hi / max(n_ref, 1), matched_hi / max(n_cand, 1)),
        "matched_box_iou": {"p1": float(np.percentile(io, 1)), "p50": float(np.median(io))} if io.size else None,
        "matched_box_center_shift_px": {"p50": float(np.median(sh)), "p99": float(np.percentile(sh, 99)),
                                        "max": float(sh.max())} if sh.size else None,
        "joints_scored": int(e.size),
        "keypoint_err_crop_px": {
            "median": float(np.median(e)), "mean": float(e.mean()),
            "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)), "max": float(e.max()),
            "pct_over_2px": float((e > 2).mean() * 100), "pct_over_5px": float((e > 5).mean() * 100),
        } if e.size else None,
        "by_crop_height_px": {
            f"{lo}-{hi}": {"joints": int(m.sum()), "median": float(np.median(e[m])),
                           "p95": float(np.percentile(e[m], 95)), "pct_over_5px": float((e[m] > 5).mean() * 100)}
            for lo, hi in CROP_HEIGHT_BINS for m in [(hh >= lo) & (hh < hi)] if m.any()},
    }


def detector_gate_failures(report: dict) -> list[str]:
    fails = []
    if report["person_agreement"] < DETECTOR_MIN_PERSON_AGREEMENT:
        fails.append(f"person agreement {report['person_agreement']:.4f} < {DETECTOR_MIN_PERSON_AGREEMENT}")
    return fails + gate_failures(report, "int8")
