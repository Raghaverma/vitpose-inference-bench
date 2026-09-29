"""The CPU-side pipeline stages -- person selection, HF crop + preprocess, HF pose decode --
kept free of TensorRT, ultralytics and the pose model, so Stage 8's worker processes can
import them without loading any of those. pipeline/stages.py re-exports everything here.

What is held fixed from Stages 0-1: the HF VitPoseImageProcessor for crops (box padding
1.25, scipy affine warp to 256x192, ImageNet normalization) and HF post_process_pose_estimation
(DARK) for decoding, fed fp32 heatmaps.

MAX_PERSONS = 4 and the largest-area-first rule are AutoClipping's production values
(vitpose.max_persons in its config/base.yaml): nets footage has a bowler, a batsman and
people in the neighbouring nets, and the cap bounds the per-frame pose cost.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale

DET_CONF = 0.35
MAX_PERSONS = 4
DATASET_INDEX = 0          # COCO expert: the one production runs, and INT8's only validated one
POSE_INPUT = (3, 256, 192)
HEATMAP_SHAPE = (17, 64, 48)


def load_processor(checkpoint: str):
    """The HF VitPoseImageProcessor alone -- no model weights."""
    return AutoProcessor.from_pretrained(checkpoint)


def select_persons(xyxy: np.ndarray, max_persons: int = MAX_PERSONS) -> np.ndarray:
    """Detector xyxy boxes -> up to max_persons xywh boxes, largest area first (float32)."""
    if len(xyxy) == 0:
        return np.zeros((0, 4), np.float32)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    # Stable sort: equal areas keep the detector's own (confidence) order.
    keep = xyxy[np.argsort(-area, kind="stable")[:max_persons]]
    return np.stack([keep[:, 0], keep[:, 1], keep[:, 2] - keep[:, 0], keep[:, 3] - keep[:, 1]],
                    axis=1).astype(np.float32)


def preprocess(processor, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """HF crop warp + normalization -> contiguous fp16 (n, 3, 256, 192), ready for an H2D copy.
    The fp32 -> fp16 cast happens here rather than on the GPU (Stage 1 cast after upload); both
    round to nearest even, so the pose input is bit-identical and the copy is half the bytes."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pv = processor(rgb, boxes=[boxes], return_tensors="pt")["pixel_values"]
    return np.ascontiguousarray(pv.to(torch.float16).numpy())


def postprocess(processor, heatmaps: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fp16 heatmaps (n, 17, 64, 48) -> keypoints (n, 17, 2) in frame px and scores (n, 17).
    n <= MAX_PERSONS, far below the >= 300-box float32 index bug documented in Stage 6."""
    hm = torch.from_numpy(heatmaps).float()
    poses = processor.post_process_pose_estimation(SimpleNamespace(heatmaps=hm), boxes=[boxes])[0]
    return (np.stack([p["keypoints"].numpy() for p in poses]).astype(np.float32),
            np.stack([p["scores"].numpy().reshape(-1) for p in poses]).astype(np.float32))


def crop_height_px(boxes: np.ndarray) -> np.ndarray:
    """Height in source px of the padded, aspect-matched region each crop was warped from --
    the denominator that turns a keypoint error into model-input px (x 256), as in Stage 6."""
    w, h = POSE_INPUT[2], POSE_INPUT[1]
    return np.array([box_to_center_and_scale(b, image_width=w, image_height=h)[1][1] * 200.0
                     for b in boxes], np.float32)
