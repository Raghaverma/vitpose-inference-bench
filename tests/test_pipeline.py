#!/usr/bin/env python3
"""Stages 7-8: the video pipelines, checked on a short real segment.

    pytest tests/test_pipeline.py -v

- person selection (CPU, no footage needed);
- stages.detect_frames() -- Stage 8's sync-free detector call -- gives bit-identical boxes to
  stages.detect(), the Stage 7 call it replaces;
- the synchronous pipeline with TensorRT FP16 matches the same pipeline with PyTorch FP16
  within pipeline/compare.py's FP16 gates;
- the asynchronous pipeline returns every frame, in order, with keypoints within the same gates,
  at a micro-batch that splits frames across batches: with YOLO per frame its boxes are
  bit-identical to the synchronous pipeline's, with YOLO batched over 4 frames they agree within
  BOX_TOL_BATCHED_DET_PX on all but a few frames.

The GPU tests need the Stage 6 held-out videos (an AutoClipping jobs dir, see
pipeline/stages.workload_videos) and skip without them.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline import compare, stages
from pipeline.cpu_stages import select_persons

N_FRAMES = 90


def test_select_persons_largest_first_capped():
    xyxy = np.array([[0, 0, 10, 10], [0, 0, 30, 30], [5, 5, 25, 15], [0, 0, 20, 20], [1, 1, 2, 2]], np.float32)
    boxes = select_persons(xyxy, max_persons=4)
    assert boxes.shape == (4, 4)
    np.testing.assert_array_equal(boxes[:, 2] * boxes[:, 3], [900, 400, 200, 100])
    np.testing.assert_array_equal(boxes[0], [0, 0, 30, 30])
    assert select_persons(np.zeros((0, 4), np.float32)).shape == (0, 4)


@pytest.fixture(scope="module")
def video():
    jobs = stages.default_jobs_dir()
    if not jobs.is_dir():
        pytest.skip(f"{jobs} not found -- the pipeline tests need the Stage 6 held-out footage")
    return stages.workload_videos(jobs)[0]


@pytest.fixture(scope="module")
def detector(video):
    det = stages.load_detector()
    cap = stages.open_video(video["path"])
    stages.detect(det, cap.read()[1])
    cap.release()
    return det


@pytest.fixture(scope="module")
def processor():
    return stages.load_processor()


def _frames(video, n):
    cap = stages.open_video(video["path"])
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    return frames


def test_detect_frames_matches_detect(video, detector):
    for frame in _frames(video, 30):
        boxes, n_det, _ = stages.detect(detector, frame)
        (boxes2, n_det2), = stages.detect_frames(detector, [frame])
        assert n_det == n_det2
        np.testing.assert_array_equal(boxes, boxes2)


@pytest.fixture(scope="module")
def sync_runs(video, detector, processor):
    from pipeline.sync_pipeline import run_pass
    out = {}
    for name in ("pytorch", "trt-fp16"):
        backend = stages.load_backend(name, (1, 2, 4, 8))
        log, _ = run_pass(video, detector, processor, backend, False, N_FRAMES)
        out[name] = {video["job_id"]: log.arrays()}
    return out


def test_sync_trt_fp16_matches_pytorch(sync_runs):
    rep = compare.compare_runs(sync_runs["pytorch"], sync_runs["trt-fp16"])
    assert rep["frames_boxes_differ"] == 0, "same frames, same detector: boxes must not depend on the pose backend"
    assert rep["crops_compared"] > 0
    assert not compare.gate_failures(rep, "fp16"), rep["keypoint_err_crop_px"]


@pytest.mark.parametrize("det_batch", [1, 4])
def test_async_matches_sync(video, detector, sync_runs, det_batch):
    from pipeline.async_pipeline import BOX_TOL_BATCHED_DET_PX, AsyncConfig, AsyncPose, CpuPools, Run
    # micro-batch 8 with up to 4 crops per frame: batches straddle frames.
    cfg = AsyncConfig(micro_batch=8, batch_timeout_ms=20, preprocess_workers=2, postprocess_workers=1, ring_slots=16,
                      det_batch=det_batch)
    frame_shape = _frames(video, 1)[0].shape
    pools = CpuPools(cfg, frame_shape)
    try:
        pose = AsyncPose(stages.load_backend("trt-fp16", (1, 2, 4, 8)), cfg.micro_batch)
        log, stats = Run(video, detector, pools, pose, cfg, N_FRAMES).run()
    finally:
        pools.close()
    assert stats["frames"] == N_FRAMES
    tol = 0.0 if det_batch == 1 else BOX_TOL_BATCHED_DET_PX
    rep = compare.compare_runs(sync_runs["trt-fp16"], {video["job_id"]: log.arrays()}, tol)
    if det_batch == 1:
        assert rep["frames_boxes_identical"] == N_FRAMES, "per-frame detection must reproduce Stage 7's boxes exactly"
    # A 0.1 px box change can reorder two near-equal people (largest first) or swap the 4th and
    # 5th: Stage 8 measured it on 0.7% of frames.
    assert rep["frames_boxes_differ"] <= max(2, N_FRAMES // 20), rep["box_max_abs_diff_px"]
    assert not compare.gate_failures(rep, "fp16"), rep["keypoint_err_crop_px"]
    assert max(int(k) for k in stats["batch_crops_hist"]) > stages.MAX_PERSONS, "no batch spanned frames"
