#!/usr/bin/env python3
"""Stage 0 baseline: load ViTPose++-L from a local checkpoint, run it on one
image, sanity-check the output, then benchmark latency / FPS / VRAM.

    python baseline.py --image samples/sample.jpg

ViTPose is top-down, so a YOLO detector supplies the person box that gets
cropped and fed to ViTPose. Only the ViTPose forward pass is timed in the
benchmark -- the detector is just how Stage 0 gets a realistic crop; later
stages (ONNX, TensorRT) replace the ViTPose forward pass alone.

See README.md for the checkpoint download instructions and the full
PyTorch -> ONNX -> TensorRT roadmap.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "vitpose-plus-large"
DEFAULT_HUB_ID = "usyd-community/vitpose-plus-large"
DEFAULT_DETECTOR = REPO_ROOT / "checkpoints" / "yolov8s.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=REPO_ROOT / "samples" / "sample.jpg",
                    help="Path to a single test image containing at least one person.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help=f"Local dir or HF hub id for ViTPose++-L. Defaults to "
                         f"{DEFAULT_CHECKPOINT_DIR} if present, else the hub id "
                         f"'{DEFAULT_HUB_ID}' (downloaded/cached by transformers).")
    p.add_argument("--detector", type=str, default=str(DEFAULT_DETECTOR),
                    help="YOLO weights (.pt) used to find the person box. Falls back "
                         "to ultralytics' own 'yolov8s.pt' auto-download if not found.")
    p.add_argument("--det-conf", type=float, default=0.35)
    p.add_argument("--dataset-index", type=int, default=0,
                    help="ViTPose++ MoE expert index (0 = COCO). Pass -1 to omit.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="Defaults to fp16 on cuda, fp32 on cpu.")
    p.add_argument("--num-iters", type=int, default=100,
                    help="ViTPose forward passes to time for the benchmark.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "raw")
    p.add_argument("--skip-benchmark", action="store_true",
                    help="Only run the single-image correctness check.")
    return p.parse_args()


def resolve_checkpoint(checkpoint: str | None) -> str:
    if checkpoint is not None:
        return checkpoint
    if DEFAULT_CHECKPOINT_DIR.is_dir():
        return str(DEFAULT_CHECKPOINT_DIR)
    print(f"[baseline] {DEFAULT_CHECKPOINT_DIR} not found locally; falling back to "
          f"hub id '{DEFAULT_HUB_ID}' (transformers will download and cache it).")
    return DEFAULT_HUB_ID


def load_pose_model(checkpoint: str, device: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = (VitPoseForPoseEstimation
             .from_pretrained(checkpoint, torch_dtype=dtype)
             .to(device).eval())
    return processor, model


def detect_person_box(detector: YOLO, image_bgr: np.ndarray, conf: float,
                       device: str) -> np.ndarray:
    """Largest-area person box in the image, as a single-row (1, 4) xywh array."""
    result = detector(image_bgr, verbose=False, conf=conf, classes=[0], device=device)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No person detected in the image -- try a lower --det-conf "
                            "or a different --image.")
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[areas.argmax()]
    return np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32)


@torch.no_grad()
def run_pose(processor, model, image_rgb: np.ndarray, boxes: np.ndarray, device: str,
             dtype: torch.dtype, dataset_index: int | None):
    inputs = processor(image_rgb, boxes=[boxes], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    kwargs = {}
    if dataset_index is not None:
        kwargs["dataset_index"] = torch.full((pixel_values.shape[0],), dataset_index,
                                              dtype=torch.long, device=device)
    outputs = model(pixel_values=pixel_values, **kwargs)
    # DARK decode (post_process_pose_estimation) runs a scipy gaussian_filter
    # that rejects fp16 arrays.
    outputs.heatmaps = outputs.heatmaps.float()
    poses = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
    return poses, pixel_values


def verify_output(poses: list[dict], image_shape: tuple[int, int]) -> None:
    """Hard invariants on the decoded keypoints. These exist to catch a broken
    model port (bad weight names, wrong dtype casts, a transposed heatmap-to-
    image coordinate mapping) rather than to judge pose quality."""
    h, w = image_shape
    assert len(poses) > 0, "processor returned zero poses for a detected box"
    for pose in poses:
        kpts, scores = pose["keypoints"], pose["scores"].reshape(-1)
        assert kpts.shape == (17, 2), f"expected 17 COCO keypoints, got {tuple(kpts.shape)}"
        assert torch.isfinite(kpts).all(), "non-finite keypoint coordinates"
        assert torch.isfinite(scores).all(), "non-finite keypoint scores"
        # DARK/UDP subpixel refinement interpolates the heatmap peak and can
        # overshoot slightly past 1.0 -- that's expected, not a broken model.
        assert (scores >= 0).all() and (scores <= 1.1).all(), \
            f"scores out of [0, 1.1]: min={scores.min():.3f} max={scores.max():.3f}"
        # A little slack outside the crop is normal (the model can extrapolate
        # past the box edge); wildly out-of-frame values mean the box<->heatmap
        # coordinate mapping is broken.
        slack = 0.25
        x_ok = (kpts[:, 0] >= -slack * w) & (kpts[:, 0] <= w * (1 + slack))
        y_ok = (kpts[:, 1] >= -slack * h) & (kpts[:, 1] <= h * (1 + slack))
        assert bool((x_ok & y_ok).all()), "keypoints fall far outside the image bounds"
    mean_conf = float(torch.cat([p["scores"].reshape(-1) for p in poses]).mean())
    print(f"[baseline] verify_output OK -- {len(poses)} pose(s), "
          f"mean keypoint confidence {mean_conf:.3f}")


def draw_pose(image_bgr: np.ndarray, poses: list[dict], out_path: Path,
              conf_thresh: float = 0.3) -> None:
    canvas = image_bgr.copy()
    for pose in poses:
        kpts = pose["keypoints"].cpu().numpy()
        scores = pose["scores"].cpu().numpy().reshape(-1)
        for i, j in COCO_SKELETON:
            if scores[i] > conf_thresh and scores[j] > conf_thresh:
                cv2.line(canvas, tuple(kpts[i].astype(int)), tuple(kpts[j].astype(int)),
                          (0, 255, 0), 2)
        for (x, y), s in zip(kpts, scores):
            if s > conf_thresh:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[baseline] wrote annotated image to {out_path}")


@torch.no_grad()
def benchmark(model, pixel_values: torch.Tensor, dataset_index: int | None, device: str,
              num_iters: int, warmup: int) -> dict:
    kwargs = {}
    if dataset_index is not None:
        kwargs["dataset_index"] = torch.full((pixel_values.shape[0],), dataset_index,
                                              dtype=torch.long, device=device)
    for _ in range(warmup):
        model(pixel_values=pixel_values, **kwargs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    for _ in range(num_iters):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(pixel_values=pixel_values, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean = statistics.mean(latencies_ms)
    batch_size = int(pixel_values.shape[0])
    return {
        "num_iters": num_iters,
        "batch_size": batch_size,
        "mean_ms": mean,
        "std_ms": statistics.pstdev(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[max(0, int(0.95 * num_iters) - 1)],
        "fps": 1000.0 / mean * batch_size,
        "peak_vram_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                          if device.startswith("cuda") else None),
    }


def main() -> None:
    args = parse_args()
    device = args.device
    dtype_str = args.dtype or ("fp16" if device.startswith("cuda") else "fp32")
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    dataset_index = None if args.dataset_index < 0 else args.dataset_index

    if not args.image.is_file():
        raise SystemExit(
            f"--image not found: {args.image}\n"
            f"Supply any photo containing a person, e.g. --image path/to/photo.jpg")

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[baseline] loading ViTPose++-L from {checkpoint} ({device}, {dtype_str})")
    processor, model = load_pose_model(checkpoint, device, dtype)

    detector_weights = args.detector if Path(args.detector).is_file() else "yolov8s.pt"
    print(f"[baseline] loading person detector from {detector_weights}")
    detector = YOLO(detector_weights)

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    box = detect_person_box(detector, image_bgr, args.det_conf, device)
    print(f"[baseline] detected person box (xywh): {box[0].round(1).tolist()}")

    poses, pixel_values = run_pose(processor, model, image_rgb, box, device, dtype,
                                    dataset_index)
    verify_output(poses, (h, w))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_pose(image_bgr, poses, args.output_dir / "baseline_sample.jpg")

    report = {
        "checkpoint": checkpoint,
        "device": device,
        "dtype": dtype_str,
        "image": str(args.image),
        "box_xywh": box[0].tolist(),
        "keypoints": poses[0]["keypoints"].cpu().tolist(),
        "scores": poses[0]["scores"].cpu().reshape(-1).tolist(),
    }

    if not args.skip_benchmark:
        print(f"[baseline] benchmarking {args.num_iters} ViTPose forward passes "
              f"(warmup={args.warmup})...")
        stats = benchmark(model, pixel_values, dataset_index, device, args.num_iters,
                           args.warmup)
        vram_str = f"{stats['peak_vram_mb']:.0f} MB" if stats["peak_vram_mb"] else "n/a"
        print(f"[baseline] mean {stats['mean_ms']:.2f} ms  p50 {stats['p50_ms']:.2f} ms  "
              f"p95 {stats['p95_ms']:.2f} ms  fps {stats['fps']:.1f}  "
              f"peak_vram {vram_str}")
        report["benchmark"] = stats

    report_path = args.output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[baseline] wrote {report_path}")


if __name__ == "__main__":
    main()
