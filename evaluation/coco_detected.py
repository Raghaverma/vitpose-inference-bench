#!/usr/bin/env python3
"""Stage 12: the detector variants against human labels (COCO val2017, detected boxes).

    python -m evaluation.coco_detected          # ~10 min; needs datasets/coco (evaluation/coco_pose.py)

Stage 10 scores the pose model on ground-truth boxes. A detector change is judged here with the
other standard top-down protocol: the pipeline's own detector finds the people (person class,
conf 0.35, as in Stages 0-9, but no cap of 4 -- COCO images aren't nets footage), the pose model
(TensorRT FP16, HF crop warp and decode, as in Stage 10) poses every detection, and COCO's OKS
evaluation scores the result over all 5000 val2017 images. Pose score = box confidence x mean
joint score over joints > 0.2.

Detector variants (pipeline/stages.load_detector): pt32 (Stages 0-9), pt16 (PyTorch fp16, box
decoding fp32) and trt16 (the Stage 12 TensorRT engine of that graph). Each image goes through
detect_frames() alone, letterboxed like the pipeline's frames.

Besides AP, a paired statistic: for each annotated person, the best OKS any pose of that image
reaches against them ("best-OKS"), compared with pt32's with a bootstrap 95% interval.

Output: results/evaluation/coco_detected.json.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline import REPO_ROOT
from compare_golden import decode_crop_batch
from evaluation.coco_pose import (BATCH, COCO_DIR, RESULTS, SIGMAS, coco_ap, detections, forward_all, load_backend,
                                  paired)
from pipeline import gpu_guard, stages
from pipeline.cpu_stages import preprocess

VARIANTS = ("pt32", "pt16", "trt16")


def detect_all(det, images: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    boxes, confs = [], []
    for im in images:
        frame = cv2.imread(str(COCO_DIR / "val2017" / im["file_name"]), cv2.IMREAD_COLOR)
        (xyxy, conf), = stages.detect_frames_raw(det, [frame])
        boxes.append(np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
                     .astype(np.float32) if len(xyxy) else np.zeros((0, 4), np.float32))
        confs.append(conf.astype(np.float32))
    return boxes, confs


def best_oks(gt: dict, img_ids: np.ndarray, kp: np.ndarray, all_img_ids: list[int]) -> np.ndarray:
    """Per annotated person (num_keypoints > 0, not crowd), the best OKS of any pose predicted in
    its image."""
    by_img: dict[int, list[int]] = {}
    for n, i in enumerate(img_ids):
        by_img.setdefault(int(i), []).append(n)
    var = (SIGMAS * 2) ** 2
    out = []
    for a in gt["annotations"]:
        if a["num_keypoints"] == 0 or a["iscrowd"]:
            continue
        g = np.array(a["keypoints"], np.float32).reshape(17, 3)
        preds = by_img.get(a["image_id"], [])
        if not preds:
            out.append(0.0)
            continue
        d2 = ((kp[preds] - g[None, :, :2]) ** 2).sum(-1)
        e = d2 / var / (a["area"] + np.spacing(1)) / 2
        vis = g[:, 2] > 0
        out.append(float(((np.exp(-e) * vis).sum(1) / vis.sum()).max()))
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=COCO_DIR / "annotations" / "person_keypoints_val2017.json")
    p.add_argument("--limit", type=int, default=None, help="First N images only (smoke runs).")
    p.add_argument("--output", type=Path, default=RESULTS / "coco_detected.json")
    args = p.parse_args()
    gt = json.loads(args.annotations.read_text())
    images = sorted(gt["images"], key=lambda im: im["id"])[:args.limit]
    all_ids = [im["id"] for im in images]
    if args.limit:
        keep = set(all_ids)
        gt = {**gt, "images": images, "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
    processor = stages.load_processor()
    t0 = time.perf_counter()

    dets = {}
    for v in VARIANTS:
        gpu_guard.wait_for_no_prod()
        det = stages.load_detector(v)
        stages.detect(det, np.zeros((480, 640, 3), np.uint8))
        dets[v] = detect_all(det, images)
        del det
        torch.cuda.empty_cache()
        print(f"[coco-det] {v}: {sum(len(b) for b in dets[v][0])} person boxes ({time.perf_counter() - t0:.0f}s)", flush=True)

    gpu_guard.wait_for_no_prod()
    be = load_backend("trt-fp16")
    report = {"stage": 12, "variants": {}, "paired_best_oks_vs_pt32": {}}
    ref_best = None
    rng = np.random.default_rng(0)
    for v in VARIANTS:
        boxes, confs = dets[v]
        img_ids = np.concatenate([np.full(len(b), i) for b, i in zip(boxes, all_ids)]).astype(np.int64)
        flat_boxes = np.concatenate(boxes)
        flat_conf = np.concatenate(confs)
        pv = np.empty((len(flat_boxes), 3, 256, 192), np.float16)
        row = 0
        for im, b in zip(images, boxes):
            if len(b):
                frame = cv2.imread(str(COCO_DIR / "val2017" / im["file_name"]), cv2.IMREAD_COLOR)
                pv[row:row + len(b)] = preprocess(processor, frame, b)
                row += len(b)
        kp, sc = decode_crop_batch(processor, forward_all(be, pv), flat_boxes)
        ap = coco_ap_all(gt, detections(img_ids, kp, sc, flat_conf), all_ids)
        bo = best_oks(gt, img_ids, kp, all_ids)
        report["variants"][v] = {"person_boxes": int(len(flat_boxes)), "mean_best_oks": float(bo.mean()), **ap}
        if v == "pt32":
            ref_best = bo
        else:
            report["paired_best_oks_vs_pt32"][v] = paired(ref_best, bo, rng)
        print(f"[coco-det] {v}: AP {ap['AP']:.4f} AP50 {ap['AP50']:.4f} AR {ap['AR']:.4f}, mean best-OKS {bo.mean():.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    report.update({
        "protocol": {"detector": "pipeline.stages.detect_frames_raw, person class, conf 0.35, no person cap",
                     "pose": "TensorRT FP16 ViTPose++-L (b16 engine), HF crop warp + HF DARK decode, dataset_index 0",
                     "score": "box confidence x mean joint score over joints > 0.2",
                     "images": len(images), "batch": BATCH},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[coco-det] wrote {args.output}")


def coco_ap_all(gt: dict, dets: list[dict], img_ids: list[int]) -> dict:
    """coco_ap() over every image, including those where nothing was detected."""
    if not dets:
        return {"AP": 0.0}
    return coco_ap(gt, dets, img_ids)


if __name__ == "__main__":
    main()
