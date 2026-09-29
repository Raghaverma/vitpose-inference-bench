#!/usr/bin/env python3
"""Stage 10: pose quality against human labels (COCO val2017 keypoints).

    python -m evaluation.coco_pose              # ~15 min; needs datasets/coco (see below)

Every accuracy number in Stages 2-9 is agreement with another model: PyTorch FP16 is the
reference and the other backends are gated on how far they move from it. This measures each
backend against people's own keypoint annotations instead, with the standard top-down protocol
(ground-truth person boxes -> crops -> pose), scored with COCO's own OKS evaluation
(pycocotools).

The pipeline's own pre- and post-processing is used throughout: HF's crop warp (1.25 box
padding) and HF's DARK decode, dataset_index 0 (the COCO expert), no flip test. So these are
this repo's numbers, not a reproduction of the paper's.

COCO's keypoint annotations barely cover the regime where Stage 7 found INT8 failing: only 0.4%
of annotated people give a crop under 64 px tall, against a third of the crops in nets footage.
So besides the original images, each annotated person is also evaluated SHRUNK: its image is
downscaled (cv2.INTER_AREA) so that its crop is a given height (32 ... 160 px), and its
annotation is scaled with it. OKS normalizes by the person's area, so a model that is equally
good at every scale scores the same at every height. A person already smaller than the target is
left out of that set rather than upscaled.

Variants: PyTorch FP16, TensorRT FP16 and INT8 (the Stage 4/6 batch-16 engines), and "hybrid"
policies that decode crops under a height threshold from FP16 and the rest from INT8 (Stage 13
builds that into the pipeline). Also the Stage 11 codecs on the original set: the GPU crop warp
(checked bit-identical to HF's here too), the cv2 crop warp, and the GPU pose decode.

Metrics: COCO keypoint AP (pycocotools, all area ranges) per variant and set, and per-person OKS
against that person's own annotation, compared pairwise with PyTorch FP16 with a bootstrap 95%
interval, so small differences are testable. Person score for AP ranking: the mean of the joint
scores above 0.2 (mmpose's top-down rescoring; boxes are ground truth).

Data: COCO val2017 images + person_keypoints_val2017.json under datasets/coco/ (gitignored):
    curl -O http://images.cocodataset.org/zips/val2017.zip
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Outputs: results/evaluation/coco_pose.json (committed; COCO is public); per-person keypoints in
results/raw/evaluation/coco_pose_keypoints.npz (gitignored).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline import REPO_ROOT
from compare_golden import decode_crop_batch
from pipeline import fast_codec, gpu_guard, stages
from pipeline.cpu_stages import preprocess

COCO_DIR = REPO_ROOT / "datasets" / "coco"
RESULTS = REPO_ROOT / "results" / "evaluation"
RAW = REPO_ROOT / "results" / "raw" / "evaluation"
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
SHRINK_TARGETS = (32, 40, 56, 80, 112, 160)  # crop heights, px (none equal to a hybrid threshold)
HYBRID_THRESHOLDS = (48, 64, 96, 128)        # crops under this height -> FP16, the rest -> INT8
KPT_SCORE_THR = 0.2
BATCH = 16
N_BOOT = 2000


# ---- data ---------------------------------------------------------------------------------------

def load_instances(ann_path: Path) -> tuple[dict, list[dict]]:
    gt = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in gt["images"]}
    inst = [a for a in gt["annotations"] if a["num_keypoints"] > 0 and not a["iscrowd"]]
    inst.sort(key=lambda a: (a["image_id"], a["id"]))
    for a in inst:
        a["file_name"] = images[a["image_id"]]["file_name"]
    return gt, inst


def build_set(inst: list[dict], target: int | None, processor, cropper: fast_codec.GpuCropper | None = None):
    """Crops for one evaluation set. target None = original images; else each person's image is
    downscaled so its crop is `target` px tall. Returns dict with pixel values (HF warp), boxes,
    scaled GT keypoints/areas, image sizes, the member indices into `inst`, and (original set)
    the cv2 warp's crops plus, with a cropper, a count of GPU-warp values differing from HF's."""
    boxes_all = np.array([a["bbox"] for a in inst], np.float32)
    heights = fast_codec.crop_height_px(boxes_all)
    members = np.arange(len(inst)) if target is None else np.flatnonzero(heights > target)
    out = {"members": members, "pv": np.empty((len(members), 3, 256, 192), np.float16),
           "boxes": np.empty((len(members), 4), np.float32), "kps": np.empty((len(members), 17, 3), np.float32),
           "area": np.empty(len(members), np.float64), "wh": np.empty((len(members), 2), np.int64)}
    gpu_mismatch, cv2_pv = 0, None
    if target is None:
        cv2_pv = np.empty_like(out["pv"])
    cache: dict = {}
    for row, i in enumerate(members):
        a = inst[i]
        img = cache.get(a["file_name"])
        if img is None:
            cache.clear()
            img = cache[a["file_name"]] = cv2.imread(str(COCO_DIR / "val2017" / a["file_name"]), cv2.IMREAD_COLOR)
        box = boxes_all[i]
        kps = np.array(a["keypoints"], np.float32).reshape(17, 3)
        area = float(a["area"])
        if target is not None:
            f = target / float(heights[i])
            h, w = img.shape[:2]
            nw, nh = max(1, round(w * f)), max(1, round(h * f))
            fx, fy = nw / w, nh / h
            src = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            box = np.array([box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy], np.float32)
            kps = kps.copy()
            kps[:, 0] *= fx
            kps[:, 1] *= fy
            area *= fx * fy
        else:
            src = img
        pv = preprocess(processor, src, box[None])
        out["pv"][row] = pv[0]
        out["boxes"][row] = box
        out["kps"][row] = kps
        out["area"][row] = area
        out["wh"][row] = (src.shape[1], src.shape[0])
        if target is None:
            cv2_pv[row] = fast_codec.warp_cv2(src, box[None])[0]
            if cropper is not None:
                maps = torch.from_numpy(fast_codec.sample_maps(*fast_codec.center_scale(box[None]))).cuda()
                g = cropper.warp(torch.from_numpy(src).cuda()[None], torch.zeros(1, dtype=torch.long, device="cuda"), maps)
                gpu_mismatch += int((g.cpu().numpy()[0] != pv[0]).sum())
    out["crop_height"] = fast_codec.crop_height_px(out["boxes"])
    out["gpu_warp_values_differing_from_hf"] = gpu_mismatch if target is None and cropper is not None else None
    out["cv2_pv"] = cv2_pv
    return out


# ---- inference ----------------------------------------------------------------------------------

def forward_all(backend, pv: np.ndarray) -> np.ndarray:
    hm = np.empty((len(pv), *stages.HEATMAP_SHAPE), np.float16)
    for k in range(0, len(pv), BATCH):
        chunk = np.ascontiguousarray(pv[k:k + BATCH])
        hm[k:k + len(chunk)] = backend.download(backend.forward(backend.upload(chunk)))
    return hm


def load_backend(name: str):
    if name == "pytorch":
        return stages.TorchPose(max_batch=BATCH)
    return stages.TrtPose(name.split("-")[1], (1, 2, 4, 8, 16))


def gpu_decode(hm: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp, sc = [], []
    for k in range(0, len(hm), 256):
        c, s = fast_codec.center_scale(boxes[k:k + 256])
        a, b = fast_codec.decode(torch.from_numpy(hm[k:k + 256]).cuda(), c, s)
        kp.append(a.cpu().numpy())
        sc.append(b.cpu().numpy())
    return np.concatenate(kp), np.concatenate(sc)


# ---- scoring ------------------------------------------------------------------------------------

def person_scores(scores: np.ndarray) -> np.ndarray:
    m = scores > KPT_SCORE_THR
    s = np.where(m, scores, 0).sum(1) / np.maximum(m.sum(1), 1)
    return np.where(m.any(1), s, 0.0)


def oks(pred: np.ndarray, gt_kps: np.ndarray, area: np.ndarray) -> np.ndarray:
    """COCO OKS of each prediction against its own annotation (pycocotools' formula)."""
    var = (SIGMAS * 2) ** 2
    d2 = ((pred - gt_kps[..., :2]) ** 2).sum(-1)
    e = d2 / var / (area[:, None] + np.spacing(1)) / 2
    vis = gt_kps[..., 2] > 0
    return (np.exp(-e) * vis).sum(1) / vis.sum(1)


def coco_ap(gt: dict, dets: list[dict], img_ids: list[int] | None = None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(dets)
        ev = COCOeval(cg, cd, "keypoints")
        ev.params.imgIds = sorted(img_ids if img_ids is not None else {d["image_id"] for d in dets})
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = ("AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L")
    return {k: float(v) for k, v in zip(names, ev.stats)}


def detections(ids: np.ndarray, kp: np.ndarray, sc: np.ndarray, box_scores: np.ndarray | None = None) -> list[dict]:
    ps = person_scores(sc) * (1.0 if box_scores is None else box_scores)
    return [{"image_id": int(i), "category_id": 1, "score": float(p),
             "keypoints": np.concatenate([k, s[:, None]], 1).ravel().round(3).tolist()}
            for i, k, s, p in zip(ids, kp, sc, ps)]


def synthetic_gt(inst: list[dict], s: dict) -> tuple[dict, np.ndarray]:
    """A COCO ground truth with one image per person of a shrunk set (image id = annotation id)."""
    ids = np.array([inst[i]["id"] for i in s["members"]])
    images = [{"id": int(i), "width": int(w), "height": int(h)} for i, (w, h) in zip(ids, s["wh"])]
    anns = [{"id": int(i), "image_id": int(i), "category_id": 1, "iscrowd": 0, "area": float(ar),
             "bbox": b.tolist(), "keypoints": k.ravel().tolist(), "num_keypoints": int((k[:, 2] > 0).sum())}
            for i, ar, b, k in zip(ids, s["area"], s["boxes"], s["kps"])]
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}, ids


def paired(o_ref: np.ndarray, o: np.ndarray, rng: np.random.Generator) -> dict:
    d = o - o_ref
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    boots = d[idx].mean(1)
    return {"mean_oks_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "persons_worse_by_0.1_oks": int((d < -0.1).sum()), "persons_better_by_0.1_oks": int((d > 0.1).sum())}


def drift(ref: tuple, cand: tuple, crop_height: np.ndarray) -> dict:
    """Keypoint distance to the reference's, the Stage 6-7 way: model-input px, over joints the
    reference scores above 0.3."""
    e = np.linalg.norm(cand[0] - ref[0], axis=-1) / crop_height[:, None] * 256
    e = e[ref[1] > 0.3]
    return {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "pct_over_5px": float((e > 5).mean() * 100)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=COCO_DIR / "annotations" / "person_keypoints_val2017.json")
    p.add_argument("--limit", type=int, default=None, help="First N annotated people only (smoke runs).")
    p.add_argument("--output", type=Path, default=RESULTS / "coco_pose.json")
    args = p.parse_args()

    gt, inst = load_instances(args.annotations)
    if args.limit:
        inst = inst[:args.limit]
    processor = stages.load_processor()
    gpu_guard.wait_for_no_prod()
    cropper = fast_codec.GpuCropper()
    t0 = time.perf_counter()
    sets = {"original": build_set(inst, None, processor, cropper)}
    for t in SHRINK_TARGETS:
        sets[f"crop_{t}px"] = build_set(inst, t, processor)
    print(f"[coco] {len(inst)} annotated people; sets {({k: len(v['members']) for k, v in sets.items()})}; "
          f"GPU warp differs from HF's in {sets['original']['gpu_warp_values_differing_from_hf']} values "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # Heatmaps -> keypoints, one backend on the GPU at a time.
    kp: dict[str, dict[str, tuple]] = {}
    extra: dict[str, tuple] = {}
    for name in ("pytorch", "trt-fp16", "trt-int8"):
        gpu_guard.wait_for_no_prod()
        be = load_backend(name)
        kp[name] = {}
        for sname, s in sets.items():
            hm = forward_all(be, s["pv"])
            kp[name][sname] = decode_crop_batch(processor, hm, s["boxes"])
            if sname == "original" and name == "trt-fp16":
                extra["trt-fp16 + gpu decode"] = gpu_decode(hm, s["boxes"])
                extra["trt-fp16 + cv2 warp"] = decode_crop_batch(processor, forward_all(be, s["cv2_pv"]), s["boxes"])
        del be
        torch.cuda.empty_cache()
        print(f"[coco] {name} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    for t in HYBRID_THRESHOLDS:
        kp[f"hybrid<{t}px"] = {}
        for sname, s in sets.items():
            small = (s["crop_height"] < t)[:, None, None]
            (k16, s16), (k8, s8) = kp["trt-fp16"][sname], kp["trt-int8"][sname]
            kp[f"hybrid<{t}px"][sname] = (np.where(small, k16, k8), np.where(small[..., 0], s16, s8))

    rng = np.random.default_rng(0)
    report = {"stage": 10, "sets": {}, "variants": {}, "paired_vs_pytorch": {}, "codecs_original_set": {}}
    oks_ref = {}
    for sname, s in sets.items():
        h = s["crop_height"]
        report["sets"][sname] = {"persons": int(len(s["members"])),
                                 "crop_height_px_quantiles": dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                                                      np.percentile(h, [5, 25, 50, 75, 95]).round(1).tolist())),
                                 "crops_under_64px": int((h < 64).sum())}
        if sname == "original":
            report["sets"][sname]["gpu_warp_values_differing_from_hf"] = s["gpu_warp_values_differing_from_hf"]
            gt_s, ids = gt, np.array([inst[i]["image_id"] for i in s["members"]])
        else:
            gt_s, ids = synthetic_gt(inst, s)
        for vname in kp:
            k, sc = kp[vname][sname]
            o = oks(k, s["kps"], s["area"])
            if vname == "pytorch":
                oks_ref[sname] = o
            ap = coco_ap(gt_s, detections(ids, k, sc))
            report["variants"].setdefault(vname, {})[sname] = {"mean_oks": float(o.mean()), **ap}
            if vname != "pytorch":
                report["paired_vs_pytorch"].setdefault(vname, {})[sname] = {
                    **paired(oks_ref[sname], o, rng), "keypoint_drift_vs_pytorch_crop_px": drift(kp["pytorch"][sname], (k, sc), h)}
        if sname == "original":
            for cname, (k, sc) in extra.items():
                o = oks(k, s["kps"], s["area"])
                ref_k = kp["trt-fp16"]["original"][0]
                shift = np.linalg.norm(k - ref_k, axis=-1) / h[:, None] * 256
                report["codecs_original_set"][cname] = {
                    "mean_oks": float(o.mean()), **coco_ap(gt_s, detections(ids, k, sc)),
                    "paired_vs_trt-fp16_hf": paired(oks(ref_k, s["kps"], s["area"]), o, rng),
                    "keypoint_shift_vs_hf_codec_crop_px": {"median": float(np.median(shift)),
                                                           "p99": float(np.percentile(shift, 99)),
                                                           "max": float(shift.max())}}
        line = "  ".join(f"{v} {report['variants'][v][sname]['AP']:.3f}" for v in kp)
        print(f"[coco] {sname:10s} AP: {line}", flush=True)

    report.update({
        "protocol": {"boxes": "ground truth", "crop": "HF VitPoseImageProcessor (padding 1.25)",
                     "decode": "HF post_process_pose_estimation (DARK), in chunks of 128",
                     "dataset_index": stages.DATASET_INDEX, "flip_test": False,
                     "person_score": f"mean joint score over joints > {KPT_SCORE_THR}",
                     "shrink": "image downscaled with cv2.INTER_AREA so the person's crop is the target height; "
                               "annotation scaled alike; people already smaller are left out",
                     "hybrid": "crop height < threshold -> TensorRT FP16, else TensorRT INT8",
                     "bootstrap": f"{N_BOOT} resamples of persons, paired"},
        "annotations": str(args.annotations.relative_to(REPO_ROOT)) if args.annotations.is_relative_to(REPO_ROOT) else str(args.annotations),
        "persons": len(inst),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / "coco_pose_keypoints.npz",
                        **{f"{v}__{s}__{f}": a for v, d in kp.items() for s, (k_, sc_) in d.items()
                           for f, a in (("kp", k_), ("scores", sc_))})
    print(f"[coco] wrote {args.output} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
