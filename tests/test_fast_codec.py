#!/usr/bin/env python3
"""Stages 11-13: the GPU codec, the detector variants and the FP16/INT8 hybrid.

    pytest tests/test_fast_codec.py -v

- pipeline/fast_codec.py against HF's VitPoseImageProcessor, which it replaces:
  box -> (center, scale) and the crop geometry bit-identical; the GPU crop warp (Triton kernel
  and its torch reference) bit-identical on synthetic frames with boxes off every edge; the GPU
  pose decode within 1e-3 model-input px; the cv2 warp within one grey level.
- On real footage (skipped without the Stage 6 held-out videos): the GPU warp is bit-identical
  to HF's; the TensorRT FP16 detector's boxes stay within a couple of pixels of fp32's; and the
  asynchronous pipeline with the GPU codec, and with the hybrid, returns every frame, routes
  each crop by its height, and stays within its gates.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale

from pipeline import compare, fast_codec, stages
from pipeline.cpu_stages import crop_height_px, postprocess, preprocess

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _boxes(rng, n, w=1280, h=720):
    return np.stack([rng.uniform(-80, w, n), rng.uniform(-80, h, n), rng.uniform(3, 600, n),
                     rng.uniform(3, 700, n)], 1).astype(np.float32)


def test_center_scale_and_crop_height_match_hf():
    b = _boxes(np.random.default_rng(0), 2000)
    c, s = fast_codec.center_scale(b)
    for i in range(len(b)):
        hc, hs = box_to_center_and_scale(b[i], image_width=192, image_height=256)
        assert np.array_equal(hc, c[i]) and np.array_equal(hs, s[i])
    assert np.array_equal(fast_codec.crop_height_px(b), crop_height_px(b))


@pytest.fixture(scope="module")
def processor():
    return stages.load_processor()


@cuda
def test_gpu_warp_bit_identical_to_hf_synthetic(processor):
    rng = np.random.default_rng(1)
    cropper = fast_codec.GpuCropper()
    for shape in ((720, 1280, 3), (1920, 1080, 3)):
        frame = rng.integers(0, 256, shape, dtype=np.uint8)
        boxes = _boxes(rng, 24, shape[1], shape[0])
        ref = preprocess(processor, frame, boxes)
        maps = torch.from_numpy(fast_codec.sample_maps(*fast_codec.center_scale(boxes))).cuda()
        dev = torch.from_numpy(frame).cuda()[None]
        slots = torch.zeros(len(boxes), dtype=torch.long, device="cuda")
        np.testing.assert_array_equal(cropper.warp(dev, slots, maps).cpu().numpy(), ref)
        np.testing.assert_array_equal(cropper.warp_reference(dev, slots, maps).cpu().numpy(), ref)


def test_cv2_warp_within_one_grey_level(processor):
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    boxes = _boxes(rng, 16)
    diff = np.abs(fast_codec.warp_cv2(frame, boxes).astype(np.float32) - preprocess(processor, frame, boxes).astype(np.float32))
    assert diff.max() <= 1.05 / 255 / 0.224        # one grey level, in the normalized units of the narrowest std


@cuda
def test_gpu_decode_matches_hf(processor):
    rng = np.random.default_rng(3)
    boxes = _boxes(rng, 48)
    boxes[:, 2:] = np.abs(boxes[:, 2:]) + 10
    # Peaked, noisy heatmaps like the model's: one Gaussian per joint plus low noise.
    yy, xx = np.mgrid[0:64, 0:48]
    hm = rng.uniform(0, 0.02, (48, 17, 64, 48)).astype(np.float32)
    cy, cx = rng.uniform(3, 60, (48, 17)), rng.uniform(3, 44, (48, 17))
    hm += 0.8 * np.exp(-((yy - cy[..., None, None]) ** 2 + (xx - cx[..., None, None]) ** 2) / 4.0)
    hm = hm.astype(np.float16)
    kp, sc = postprocess(processor, hm, boxes)
    gk, gs = fast_codec.decode(torch.from_numpy(hm).cuda(), *fast_codec.center_scale(boxes))
    np.testing.assert_array_equal(gs.cpu().numpy(), sc)
    err = np.linalg.norm(gk.cpu().numpy() - kp, axis=-1) / crop_height_px(boxes)[:, None] * 256
    assert err.max() < 1e-3


# ---- real footage -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def video():
    jobs = stages.default_jobs_dir()
    if not jobs.is_dir():
        pytest.skip(f"{jobs} not found -- these tests need the Stage 6 held-out footage")
    return stages.workload_videos(jobs)[2]        # the bowler clip: a third of its crops are under 64 px


def _frames(video, n, step=1):
    cap = stages.open_video(video["path"])
    out = []
    for k in range(n * step):
        ok, f = cap.read()
        if k % step == 0:
            out.append(f)
    cap.release()
    return out


@cuda
def test_gpu_warp_bit_identical_to_hf_real(video, processor):
    det = stages.load_detector()
    cropper = fast_codec.GpuCropper()
    for frame in _frames(video, 12, 40):
        boxes, _, _ = stages.detect(det, frame)
        if not len(boxes):
            continue
        maps = torch.from_numpy(fast_codec.sample_maps(*fast_codec.center_scale(boxes))).cuda()
        got = cropper.warp(torch.from_numpy(frame).cuda()[None], torch.zeros(len(boxes), dtype=torch.long, device="cuda"), maps)
        np.testing.assert_array_equal(got.cpu().numpy(), preprocess(processor, frame, boxes))


@cuda
def test_detector_variants_close_to_fp32(video):
    frames = _frames(video, 32)
    ref = [b for i in range(0, 32, 4) for b in stages.detect_frames(_det("pt32"), frames[i:i + 4])]
    for v in ("pt16", "trt16"):
        det = _det(v)
        got = [b for i in range(0, 32, 4) for b in stages.detect_frames(det, frames[i:i + 4])]
        same = [(a, b) for (a, _), (b, _) in zip(ref, got) if a.shape == b.shape]
        assert len(same) >= 30
        assert max(np.abs(a - b).max() for a, b in same if len(a)) < 3.0


_DETS: dict = {}


def _det(variant):
    if variant not in _DETS:
        _DETS[variant] = stages.load_detector(variant)
        if variant == "pt32":
            stages.detect(_DETS[variant], np.zeros((720, 1280, 3), np.uint8))
    return _DETS[variant]


@cuda
def test_async_gpu_codec_and_hybrid(video):
    from dataclasses import replace

    from pipeline.async_pipeline import AsyncConfig, AsyncGpuState
    from pipeline.sync_pipeline import run_pass

    n = 120
    frame0 = _frames(video, 1)[0]
    gpu = AsyncGpuState(["trt-fp16", "trt-hybrid"], [4], frame0, detectors=("pt32",), gpu_codec=True)
    gpu.ensure_loaded()
    processor = stages.load_processor()
    sync_log, _ = run_pass(video, gpu.detector, processor, gpu.backends["trt-fp16"], False, n)
    sync = {video["job_id"]: sync_log.arrays()}
    cfg = AsyncConfig(micro_batch=4, det_batch=1, codec="gpu", preprocess_workers=0, postprocess_workers=0)
    fp16_log, fp16_stats = gpu.run(("trt-fp16", 4), video, None, cfg, n)
    fp16 = {video["job_id"]: fp16_log.arrays()}
    assert len(fp16_log.rows["n_persons"]) == n
    # YOLO per frame: the same boxes as the synchronous pipeline; same pose backend, so the
    # only difference is the codec (and the batch composition) -- within the FP16 gates.
    rep = compare.compare_runs(sync, fp16)
    assert rep["frames_same_boxes"] == n
    assert not compare.gate_failures(rep, "fp16"), rep["keypoint_err_crop_px"]

    hyb_log, hyb_stats = gpu.run(("trt-hybrid", 4), video, None, replace(cfg, hybrid_threshold_px=64.0), n)
    hyb = {video["job_id"]: hyb_log.arrays()}
    small = int(sum((fast_codec.crop_height_px(b[:k]) < 64).sum()
                    for b, k in zip(fp16[video["job_id"]]["boxes"], fp16[video["job_id"]]["n_persons"])))
    assert hyb_stats["crops_per_route"][0] == small > 0
    rep = compare.compare_runs(fp16, hyb)
    # Crops under 64 px ran on the same FP16 engines: their keypoints match (FP16 gates on that bin).
    assert rep["by_crop_height_px"]["0-64"]["median"] < compare.FP16_MEDIAN_ERR_THRESHOLD_PX
    assert not compare.gate_failures(rep, "int8"), rep["keypoint_err_crop_px"]
    gpu.ensure_released()


@cuda
def test_trt_detector_pads_wide_images_to_its_profile():
    """A letterbox shorter than the engine profile's minimum (a > 2:1 image) must be padded, not
    run with a stale input shape -- the bug evaluation/coco_detected.py found on COCO panoramas."""
    pt, trt = _det("pt32"), _det("trt16")
    rng = np.random.default_rng(5)
    wide = np.full((200, 800, 3), 114, np.uint8)
    wide[40:190, 300:360] = rng.integers(0, 256, (150, 60, 3), dtype=np.uint8)
    for frame in (wide, np.ascontiguousarray(wide.transpose(1, 0, 2))):
        (xyxy_t, conf_t), = stages.detect_frames_raw(trt, [frame])      # must not raise or fault
        (xyxy_p, conf_p), = stages.detect_frames_raw(pt, [frame])
        assert len(xyxy_t) == len(xyxy_p)
        if len(xyxy_t):
            assert np.abs(xyxy_t - xyxy_p).max() < 4.0
