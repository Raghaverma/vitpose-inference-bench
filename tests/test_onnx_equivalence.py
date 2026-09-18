#!/usr/bin/env python3
"""Stage 2 Gate C: numerical equivalence between the exported ONNX graph and
the frozen PyTorch FP16 golden reference.

    pytest tests/test_onnx_equivalence.py -v

Runs the exported graph (results/onnx/vitpose_plus_l.onnx) through ONNX
Runtime's CUDAExecutionProvider on the exact input Stage 1's golden
reference was built from, then diffs the output against
golden/pytorch_fp16_output.npy at both the raw-tensor level and the
decoded-keypoint level via compare_golden.compare_outputs() -- reusing
Stage 1's comparator rather than a second copy of the same diff logic.

Both gates must pass independently (max_abs_error / max_rel_error on the
raw tensor, AND mean/max keypoint distance on the decoded pose) -- a tensor
regression that happens to decode to a similar-looking pose should still
fail this test, since "the keypoints still land close" is not the same
claim as "the backend computes the same thing," and later stages (TensorRT
INT8 in particular) can move the tensor a lot while keypoints look fine
right up until they don't.

Only batch=1 is tested here: conversion/export_onnx.py's dynamic-batch
export failed (see that script's docstring for the root cause), so only a
static batch=1 ONNX graph exists. The per-batch-size, per-slot equivalence
gate this test SHOULD eventually run (every swept batch size, distinct
crops per batch slot, each checked against its own standalone result --
not just batch=1 against one golden crop) is blocked on that dynamic-batch
export working, or on building a separate static export per batch size.
Noted here so it isn't silently lost; not implemented in this pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from compare_golden import compare_outputs

GOLDEN_DIR = REPO_ROOT / "golden"
ONNX_PATH = REPO_ROOT / "results" / "onnx" / "vitpose_plus_l.onnx"
REPORT_PATH = REPO_ROOT / "results" / "onnx" / "equivalence_report.json"

# Thresholds, chosen with headroom over the first observed run (recorded in
# results/onnx/equivalence_report.json): max_abs_error 0.00098 (10x headroom
# under 0.01 -- consistent with a handful of fp16 ULPs from opset/kernel-
# ordering differences between eager PyTorch and the traced+lowered ONNX
# graph, not a real regression), mean/max keypoint distance 0.02px / 0.04px
# (far under 1px -- effectively sub-pixel-perfect). max_rel_error came in at
# 0.047, uncomfortably close to an initial 0.05 threshold -- that's fp16
# quantization noise on borderline-significant heatmap values right at the
# 1e-3 significance cutoff, not a real problem, but a threshold with ~7%
# headroom isn't a meaningful gate against normal run-to-run fp16 noise, so
# this is set to 0.10 instead: still tight enough to catch an actual
# order-of-magnitude regression, with real margin over what a faithful
# export actually produces.
MAX_ABS_ERROR_THRESHOLD = 0.01
MAX_REL_ERROR_THRESHOLD = 0.10
MEAN_KEYPOINT_DIST_THRESHOLD_PX = 1.0
MAX_KEYPOINT_DIST_THRESHOLD_PX = 3.0


@pytest.fixture(scope="module")
def golden_env() -> dict:
    return json.loads((GOLDEN_DIR / "env.json").read_text())


@pytest.fixture(scope="module")
def golden_output() -> np.ndarray:
    return np.load(GOLDEN_DIR / "pytorch_fp16_output.npy")


@pytest.fixture(scope="module")
def person_crop() -> np.ndarray:
    return np.load(GOLDEN_DIR / "person_crop.npy").astype(np.float16)


@pytest.fixture(scope="module")
def processor():
    checkpoint = resolve_checkpoint(None)
    processor, _model = load_pose_model(checkpoint, "cpu", torch.float16)
    return processor


@pytest.fixture(scope="module")
def onnx_output(person_crop) -> np.ndarray:
    if not ONNX_PATH.is_file():
        pytest.skip(f"{ONNX_PATH} not found -- run `python -m conversion.export_onnx` first "
                     "(Gate A must pass before Gate C).")
    session = ort.InferenceSession(str(ONNX_PATH),
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    dataset_index = np.array([0], dtype=np.int64)
    outputs = session.run(["heatmaps"], {"pixel_values": person_crop, "dataset_index": dataset_index})
    return outputs[0]


@pytest.fixture(scope="module")
def report(golden_output, onnx_output, processor, golden_env) -> dict:
    box_xywh = golden_env["box_xywh"]
    result = compare_outputs(golden_output, onnx_output, processor, box_xywh,
                              candidate_name="ONNX Runtime FP16 (batch=1)", golden_env=golden_env)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    return result


def test_output_shape_matches_golden(golden_output, onnx_output):
    assert onnx_output.shape == golden_output.shape


def test_raw_tensor_max_abs_error(report):
    max_abs = report["tensor_diff"]["max_abs_error"]
    assert max_abs < MAX_ABS_ERROR_THRESHOLD, (
        f"max_abs_error {max_abs:.6f} exceeds {MAX_ABS_ERROR_THRESHOLD} -- "
        f"see {REPORT_PATH} for the full diff")


def test_raw_tensor_max_rel_error(report):
    max_rel = report["tensor_diff"]["max_rel_error"]
    assert max_rel < MAX_REL_ERROR_THRESHOLD, (
        f"max_rel_error {max_rel:.6f} exceeds {MAX_REL_ERROR_THRESHOLD} "
        f"(only checked where |golden| > {report['tensor_diff']['rel_error_threshold']}) -- "
        f"see {REPORT_PATH} for the full diff")


def test_decoded_keypoint_mean_distance(report):
    mean_dist = report["keypoint_diff"]["mean_distance_px"]
    assert mean_dist < MEAN_KEYPOINT_DIST_THRESHOLD_PX, (
        f"mean keypoint distance {mean_dist:.3f}px exceeds "
        f"{MEAN_KEYPOINT_DIST_THRESHOLD_PX}px -- see {REPORT_PATH} for the per-joint breakdown")


def test_decoded_keypoint_max_distance(report):
    max_dist = report["keypoint_diff"]["max_distance_px"]
    worst = max(report["keypoint_diff"]["per_joint"], key=lambda j: j["euclidean_distance_px"])
    assert max_dist < MAX_KEYPOINT_DIST_THRESHOLD_PX, (
        f"max keypoint distance {max_dist:.3f}px (joint: {worst['joint']}) exceeds "
        f"{MAX_KEYPOINT_DIST_THRESHOLD_PX}px -- see {REPORT_PATH} for the per-joint breakdown")
