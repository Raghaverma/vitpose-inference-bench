"""Stage 8's CPU worker processes: HF crop + preprocess, and HF pose decode.

Processes, not threads, measured before building on it (pipeline/gil_check.py,
results/pipeline/gil_check.json; 4 workers vs 1): the HF crop warp scales across threads
(3.7x -- scipy's affine_transform releases the GIL) but slightly better across processes
(3.9x), and the HF pose decode does not scale across threads at all -- 4 threads run it at
0.66x of one, its per-crop Python and small numpy/scipy calls fighting over the GIL -- while 4
processes give 3.4x.

Frames, crops and heatmaps never go through a pipe: they live in a shared-memory ring
(SharedRing) indexed by slot, one slot per frame in flight, and the queues carry only
(frame index, slot, boxes) and the small keypoint arrays. This module imports only numpy, cv2,
torch (CPU) and transformers -- never TensorRT, ultralytics or the pose model -- so a spawned
worker starts in a couple of seconds and a few hundred MB.

Stage 11 adds a second codec for both workers, "cv2": cv2.warpAffine for the crop warp and the
batched torch DARK decode of pipeline/fast_codec.py, in place of HF's per-crop scipy code.
"""
from __future__ import annotations

import os
import time
import traceback
from multiprocessing import shared_memory

import cv2
import numpy as np
import torch

from pipeline import fast_codec
from pipeline.cpu_stages import HEATMAP_SHAPE, MAX_PERSONS, POSE_INPUT, load_processor, postprocess, preprocess


class SharedRing:
    """`slots` frames in flight: the decoded frame, its crops and its heatmaps, in three shared
    memory blocks. The creating process owns (and unlinks) them; workers attach by name."""

    def __init__(self, slots: int, frame_shape: tuple[int, int, int], names: dict | None = None):
        self.slots, self.frame_shape = slots, tuple(frame_shape)
        specs = {
            "frames": ((slots, *frame_shape), np.uint8),
            "crops": ((slots, MAX_PERSONS, *POSE_INPUT), np.float16),
            "heatmaps": ((slots, MAX_PERSONS, *HEATMAP_SHAPE), np.float16),
        }
        self.owner = names is None
        self._shm, self.arrays = {}, {}
        for key, (shape, dtype) in specs.items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = (shared_memory.SharedMemory(create=True, size=nbytes) if self.owner
                   else shared_memory.SharedMemory(name=names[key]))
            self._shm[key] = shm
            self.arrays[key] = np.ndarray(shape, dtype, buffer=shm.buf)
        self.frames, self.crops, self.heatmaps = (self.arrays[k] for k in ("frames", "crops", "heatmaps"))

    def spec(self) -> tuple:
        return self.slots, self.frame_shape, {k: s.name for k, s in self._shm.items()}

    def close(self) -> None:
        self.frames = self.crops = self.heatmaps = None
        self.arrays.clear()
        for shm in self._shm.values():
            shm.close()
            if self.owner:
                shm.unlink()


def _setup(checkpoint: str):
    # One core per worker: the pool size is the parallelism knob, not intra-op threads.
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    return load_processor(checkpoint)


def preprocess_worker(checkpoint: str, ring_spec: tuple, task_q, done_q, codec: str = "hf") -> None:
    """task: (frame_idx, slot, boxes) -> crops written to ring.crops[slot, :n];
    reply ("crops", frame_idx, slot, n, busy_s). codec "hf": HF's crop warp (Stages 7-9);
    "cv2": Stage 11's cv2.warpAffine (pipeline/fast_codec.warp_cv2)."""
    processor = _setup(checkpoint)
    ring = SharedRing(*ring_spec[:2], names=ring_spec[2])
    warp = (lambda frame, boxes: preprocess(processor, frame, boxes)) if codec == "hf" else fast_codec.warp_cv2
    done_q.put(("ready", "preprocess", os.getpid()))
    try:
        while (task := task_q.get()) is not None:
            idx, slot, boxes = task
            t0 = time.perf_counter()
            try:
                ring.crops[slot, :len(boxes)] = warp(ring.frames[slot], boxes)
            except Exception:
                # Report instead of dying silently: the main process fails the run on it.
                done_q.put(("error", idx, traceback.format_exc()))
                continue
            done_q.put(("crops", idx, slot, len(boxes), time.perf_counter() - t0))
    finally:
        ring.close()


def _decode_vectorized(heatmaps: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp, sc = fast_codec.decode(torch.from_numpy(heatmaps), *fast_codec.center_scale(boxes))
    return kp.numpy(), sc.numpy()


def postprocess_worker(checkpoint: str, ring_spec: tuple, task_q, done_q, codec: str = "hf") -> None:
    """task: (frame_idx, slot, boxes) with heatmaps in ring.heatmaps[slot, :n];
    reply ("pose", frame_idx, slot, keypoints, scores, busy_s). codec "hf": HF's DARK decode;
    "cv2": Stage 11's batched torch decode (pipeline/fast_codec.decode), on this CPU core."""
    processor = _setup(checkpoint)
    ring = SharedRing(*ring_spec[:2], names=ring_spec[2])
    decode = (lambda hm, boxes: postprocess(processor, hm, boxes)) if codec == "hf" else _decode_vectorized
    done_q.put(("ready", "postprocess", os.getpid()))
    try:
        while (task := task_q.get()) is not None:
            idx, slot, boxes = task
            t0 = time.perf_counter()
            try:
                kp, sc = decode(np.array(ring.heatmaps[slot, :len(boxes)]), boxes)
            except Exception:
                done_q.put(("error", idx, traceback.format_exc()))
                continue
            done_q.put(("pose", idx, slot, kp, sc, time.perf_counter() - t0))
    finally:
        ring.close()
