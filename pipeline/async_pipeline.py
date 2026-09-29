#!/usr/bin/env python3
"""Stage 8: an asynchronous, multi-worker video pipeline, benchmarked against Stage 7's
synchronous one on the same footage, in the same process.

    python -m pipeline.async_pipeline                                # trt-fp16 + pytorch, defaults
    python -m pipeline.async_pipeline --backends trt-fp16 --max-frames 300    # smoke run

Stage 7 ran every stage of a frame back to back, so at any moment either the CPU or the GPU
was idle, and only one CPU core did the crop warp and the pose decode. Here every stage runs
concurrently, decoupled by queues:

  decode thread ──> detect thread ──> preprocess processes ──> pose batcher ──> pose completer ──> postprocess processes ──> collector
  (cv2 -> ring)     (YOLO, CUDA        (HF crop warp,          (cross-frame      (event wait,      (HF DARK decode)
                     stream 1)          N workers)              micro-batches,    heatmaps -> ring)
                                                                 CUDA streams 2+3)

- CPU stages in worker processes (pipeline/cpu_workers.py): the HF pose decode is GIL-bound
  and runs slower on 4 threads than on 1, while 4 processes run it 3.4x faster
  (pipeline/gil_check.py).

- Shared-memory ring (pipeline/cpu_workers.SharedRing): one slot per frame in flight holds the
  frame, its crops and its heatmaps; queues carry indices and boxes only. The ring size is the
  backpressure: the decoder blocks when every slot is in use.
- Micro-batching: the pose batcher packs crops ACROSS frames into batches of up to
  `micro_batch` (a static engine size), flushing a partial batch after `batch_timeout_ms` or at
  end of stream, on the smallest engine that fits. Stage 7 ran one frame's 1-4 crops per call.
- Pinned memory + two CUDA streams for the pose path (copy stream for H2D, pose stream for the
  forward and D2H, ordered by events), double-buffered so batch k+1's H2D overlaps batch k's
  forward; and the detector on a third stream of its own, so YOLO and ViTPose kernels overlap.
- Detection goes through stages.detect_frames(): the predictor's own methods without
  ultralytics' per-stage device-wide torch.cuda.synchronize(), which would make every detection
  wait for the pose stream. `det_batch` frames per YOLO call: 1 gives boxes bit-identical to
  Stage 7's; more is faster but not bit-exact (a batched convolution rounds differently: boxes
  move by <= 0.13 px on 128 frames checked), so both are measured as separate variants.

What stays identical to Stage 7: every stage's code (pipeline/stages.py, cpu_stages.py). What
differs: the pose batch composition (engine batch sizes) and, with det_batch > 1, the boxes'
last bits, so the check against Stage 7 is the keypoint gates of pipeline/compare.py (with a
BOX_TOL_BATCHED_DET_PX box tolerance for det_batch > 1), not bit-identity.

The throughput comparison is paired: each chunk of each video (pipeline/runner.py) runs Stage
7's plain synchronous pass and then this pipeline, per backend, in one process, so the
power-capped L4's clock drift lands on both; every unit waits for a quiet box and is re-run if
anything overlapped it, and the GPU state is released while a prod job runs.
Outputs: results/pipeline/async_<backend>.json, results/pipeline/async_summary.json, and
per-frame outputs in results/raw/pipeline/async_<backend>.npz (gitignored).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import queue
import resource
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from baseline import REPO_ROOT, resolve_checkpoint
from pipeline import compare, cpu_workers, runner, stages
from pipeline.sync_pipeline import FrameLog, env_fingerprint, merge_logs, repo_relative, run_pass

ENGINE_SIZES = (1, 2, 4, 8, 16)
N_BUFFERS = 2
STALL_S = 300.0     # no frame finishing for this long means a stage is stuck: fail, don't hang


@dataclass
class AsyncConfig:
    micro_batch: int = 16             # max crops per pose forward: one of ENGINE_SIZES
    batch_timeout_ms: float = 50.0    # flush a partial batch this long after its first crop
    preprocess_workers: int = 6
    postprocess_workers: int = 2
    ring_slots: int = 64              # frames in flight
    max_persons: int = stages.MAX_PERSONS   # Stage 9 lowers it to vary crops per frame
    det_batch: int = 1                # frames per YOLO call; > 1 moves boxes by ~0.1 px (not bit-exact)


class CpuPools:
    """The preprocess + postprocess worker processes and the shared ring they read, started once
    and reused across videos and backends (they're backend-independent)."""

    def __init__(self, cfg: AsyncConfig, frame_shape: tuple[int, int, int]):
        self.ctx = mp.get_context("spawn")
        self.ring = cpu_workers.SharedRing(cfg.ring_slots, frame_shape)
        spec, ckpt = self.ring.spec(), resolve_checkpoint(None)
        self.pre_q, self.pre_done_q = self.ctx.Queue(), self.ctx.Queue()
        self.post_q, self.post_done_q = self.ctx.Queue(), self.ctx.Queue()
        self.procs = (
            [self.ctx.Process(target=cpu_workers.preprocess_worker, args=(ckpt, spec, self.pre_q, self.pre_done_q),
                              daemon=True) for _ in range(cfg.preprocess_workers)] +
            [self.ctx.Process(target=cpu_workers.postprocess_worker, args=(ckpt, spec, self.post_q, self.post_done_q),
                              daemon=True) for _ in range(cfg.postprocess_workers)])
        for p in self.procs:
            p.start()
        ready = collections.Counter()
        while ready["preprocess"] < cfg.preprocess_workers:
            ready[self.pre_done_q.get(timeout=300)[1]] += 1
        while ready["postprocess"] < cfg.postprocess_workers:
            ready[self.post_done_q.get(timeout=300)[1]] += 1
        self.cfg = cfg

    def close(self) -> None:
        for _ in range(self.cfg.preprocess_workers):
            self.pre_q.put(None)
        for _ in range(self.cfg.postprocess_workers):
            self.post_q.put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self.ring.close()


class AsyncPose:
    """The GPU side of the pose stage for one backend: NBUF pinned host + device buffer sets of
    micro_batch crops, a copy stream and a pose stream."""

    def __init__(self, backend, micro_batch: int, streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None):
        self.be = backend
        self.is_trt = isinstance(backend, stages.TrtPose)
        if self.is_trt and micro_batch not in backend.engines:
            raise SystemExit(f"[stage8] micro_batch {micro_batch} has no {backend.name} engine "
                             f"(loaded: {backend.batch_sizes})")
        self.mb = micro_batch
        # (copy, pose) streams. Pass them in when several AsyncPose objects exist: PyTorch hands
        # out streams round-robin from a pool of 32, so creating a pair per object would, past
        # 16 objects, give two of them the same "new" streams (see AsyncGpuState).
        self.copy_stream, self.pose_stream = streams or (torch.cuda.Stream(), torch.cuda.Stream())
        mk = lambda shape, **kw: [torch.zeros((micro_batch, *shape), dtype=torch.float16, **kw) for _ in range(N_BUFFERS)]
        self.host_in = [t.pin_memory() for t in mk(stages.POSE_INPUT)]
        self.host_out = [t.pin_memory() for t in mk(stages.HEATMAP_SHAPE)]
        self.dev_in = mk(stages.POSE_INPUT, device="cuda")
        self.dev_out = mk(stages.HEATMAP_SHAPE, device="cuda")
        self.dataset_index = torch.full((micro_batch,), stages.DATASET_INDEX, dtype=torch.int64, device="cuda")
        # One TensorRT execution context per (engine, buffer): two batches are in flight at once.
        self.contexts = backend.buffer_contexts(N_BUFFERS) if self.is_trt else {}

    def engine_batch(self, n: int) -> int:
        if not self.is_trt:
            return n
        return min(b for b in self.be.batch_sizes if b >= n)   # n <= micro_batch, which has an engine

    def launch(self, i: int, n: int) -> tuple[int, torch.cuda.Event, torch.cuda.Event]:
        """Crops are already in host_in[i][:n]. Enqueue H2D -> forward -> D2H; return the engine
        batch and (start, done) events on the pose stream. Never blocks the host."""
        b = self.engine_batch(n)
        h2d, start, done = (torch.cuda.Event(enable_timing=t) for t in (False, True, True))
        with torch.cuda.stream(self.copy_stream):
            self.dev_in[i][:n].copy_(self.host_in[i][:n], non_blocking=True)
            if b > n:
                self.dev_in[i][n:b].zero_()
            h2d.record(self.copy_stream)
        self.pose_stream.wait_event(h2d)
        start.record(self.pose_stream)
        if self.is_trt:
            self.be.enqueue(b, self.dev_in[i][:b], self.dev_out[i][:b], self.pose_stream, self.contexts[b][i])
        else:
            with torch.cuda.stream(self.pose_stream), torch.inference_mode():
                out = self.be.model(pixel_values=self.dev_in[i][:n], dataset_index=self.dataset_index[:n]).heatmaps
                self.dev_out[i][:n].copy_(out)
        with torch.cuda.stream(self.pose_stream):
            self.host_out[i][:n].copy_(self.dev_out[i][:n], non_blocking=True)
        done.record(self.pose_stream)
        return b, start, done


class Run:
    """One pass of the asynchronous pipeline over one video."""

    def __init__(self, video: dict, detector, pools: CpuPools, pose: AsyncPose, cfg: AsyncConfig,
                 max_frames: int | None, start: int = 0, det_stream: torch.cuda.Stream | None = None, cap=None):
        self.video, self.detector, self.pools, self.pose, self.cfg = video, detector, pools, pose, cfg
        self.max_frames = max_frames
        # A capture already positioned at `start` (runner.open_at, outside the measurement
        # window), or skip to it now, before run() starts the clock.
        self.cap = cap if cap is not None else runner.open_at(video, start)
        self.ring = pools.ring
        self.free_slots: queue.Queue = queue.Queue()
        for s in range(cfg.ring_slots):
            self.free_slots.put(s)
        self.det_q: queue.Queue = queue.Queue()
        self.inflight: queue.Queue = queue.Queue()
        self.buffers = threading.Semaphore(N_BUFFERS)
        self.det_stream = det_stream or torch.cuda.Stream()
        handles = {self.det_stream.cuda_stream, pose.copy_stream.cuda_stream, pose.pose_stream.cuda_stream}
        if len(handles) != 3:
            raise RuntimeError("[stage8] detect, copy and pose streams alias one another (PyTorch's pooled "
                               "streams wrapped around) -- detection would serialize with the pose path")
        self.lock = threading.Lock()
        self.results: dict[int, tuple] = {}
        self.boxes: dict[int, np.ndarray] = {}
        self.remaining: dict[int, int] = {}
        self.t_decoded: dict[int, float] = {}
        self.t_done: dict[int, float] = {}
        self.total_frames = None
        self.expected_crops = None
        self.done_frames = 0
        self.finished = threading.Event()
        self.aborted = False
        self.errors: list[BaseException] = []
        self.busy = collections.defaultdict(float)
        self.batches: list[dict] = []
        self.qdepth: list[dict] = []

    # -- helpers --------------------------------------------------------------------------------
    def _guard(self, fn):
        def wrapped():
            try:
                fn()
            except BaseException as e:      # surface worker-thread failures in the main thread
                self.errors.append(e)
                self.finished.set()
        return wrapped

    def _finish_frame(self, idx: int, slot: int, result: tuple) -> None:
        self.results[idx] = result
        self.t_done[idx] = time.perf_counter()
        self.free_slots.put(slot)
        with self.lock:
            self.done_frames += 1
            if self.total_frames is not None and self.done_frames == self.total_frames:
                self.finished.set()

    # -- threads --------------------------------------------------------------------------------
    def decode_loop(self) -> None:
        cap = self.cap
        idx = 0
        while self.max_frames is None or idx < self.max_frames:
            if runner.ABORT.is_set():
                # Stop feeding the pipeline and let it drain like at end of stream, so no stale
                # messages are left in the worker queues for the next run.
                self.aborted = True
                break
            slot = self.free_slots.get()
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                self.free_slots.put(slot)
                break
            self.ring.frames[slot] = frame
            t1 = time.perf_counter()
            self.busy["decode"] += t1 - t0
            self.t_decoded[idx] = t1
            self.det_q.put((idx, slot))
            idx += 1
        cap.release()
        self.det_q.put(None)
        with self.lock:
            self.total_frames = idx
            if self.done_frames == idx:
                self.finished.set()

    def detect_loop(self) -> None:
        """YOLO on up to det_batch frames per call: the first frame blocks, the rest are taken
        only if already decoded, so a batch never waits for the decoder."""
        expected, end = 0, False
        while not end and (item := self.det_q.get()) is not None:
            items = [item]
            while len(items) < self.cfg.det_batch:
                try:
                    nxt = self.det_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    end = True
                    break
                items.append(nxt)
            t0 = time.perf_counter()
            with torch.cuda.stream(self.det_stream):
                dets = stages.detect_frames(self.detector, [self.ring.frames[slot] for _, slot in items],
                                            self.cfg.max_persons)
            self.busy["detect"] += time.perf_counter() - t0
            for (idx, slot), (boxes, n_det) in zip(items, dets):
                self.boxes[idx] = boxes
                if len(boxes) == 0:
                    self._finish_frame(idx, slot, (n_det, boxes, None, None))
                    continue
                self.results[idx] = (n_det,)
                self.remaining[idx] = len(boxes)
                expected += len(boxes)
                self.pools.pre_q.put((idx, slot, boxes))
        self.pools.pre_done_q.put(("eof", expected))

    def batch_loop(self) -> None:
        """Pack crops across frames into micro-batches and launch them."""
        pending: collections.deque = collections.deque()    # (idx, slot, person)
        state = {"received": 0, "expected": None, "deadline": None}
        timeout = self.cfg.batch_timeout_ms / 1000
        mb = self.pose.mb
        buf = 0

        def take(msg) -> None:
            if msg[0] == "error":
                raise RuntimeError(f"preprocess worker failed on frame {msg[1]}: {msg[2]}")
            if msg[0] == "eof":
                state["expected"] = msg[1]
                return
            _, idx, slot, n, busy = msg
            self.busy["preprocess"] += busy
            if not pending:
                state["deadline"] = time.perf_counter() + timeout
            pending.extend((idx, slot, p) for p in range(n))
            state["received"] += n

        while True:
            exhausted = state["expected"] is not None and state["received"] == state["expected"]
            if not pending and exhausted:
                break
            due = pending and (len(pending) >= mb or exhausted or time.perf_counter() >= state["deadline"])
            if not due:
                wait = 1.0 if not pending else max(0.0, state["deadline"] - time.perf_counter())
                try:
                    take(self.pools.pre_done_q.get(timeout=wait))
                except queue.Empty:
                    pass
                continue
            # A batch is due. Wait for a free buffer set BEFORE deciding what goes in it: crops
            # that arrive while the GPU is still busy with the previous two batches then fill
            # this one instead of it launching half-empty and padded.
            self.buffers.acquire()
            while len(pending) < mb:
                try:
                    take(self.pools.pre_done_q.get_nowait())
                except queue.Empty:
                    break
            items = [pending.popleft() for _ in range(min(mb, len(pending)))]
            if pending:
                state["deadline"] = time.perf_counter() + timeout
            t0 = time.perf_counter()
            host = self.pose.host_in[buf].numpy()
            for j, (idx, slot, p) in enumerate(items):
                host[j] = self.ring.crops[slot, p]
            b, start, done = self.pose.launch(buf, len(items))
            self.busy["batcher"] += time.perf_counter() - t0
            self.inflight.put((items, buf, b, start, done))
            buf = (buf + 1) % N_BUFFERS
        self.inflight.put(None)

    def complete_loop(self) -> None:
        """Wait for each launched batch in order, scatter its heatmaps back to the frames' slots,
        and hand every frame whose crops are all back to the postprocess workers."""
        while (item := self.inflight.get()) is not None:
            items, buf, b, start, done = item
            done.synchronize()
            t0 = time.perf_counter()
            out = self.pose.host_out[buf].numpy()
            ready = []
            for j, (idx, slot, p) in enumerate(items):
                self.ring.heatmaps[slot, p] = out[j]
                self.remaining[idx] -= 1
                if self.remaining[idx] == 0:
                    ready.append((idx, slot))
            self.buffers.release()
            for idx, slot in ready:
                self.pools.post_q.put((idx, slot, self.boxes[idx]))
            self.busy["completer"] += time.perf_counter() - t0
            self.batches.append({"crops": len(items), "engine_batch": b, "gpu_ms": start.elapsed_time(done)})

    def collect_loop(self) -> None:
        while not self.finished.is_set():
            try:
                msg = self.pools.post_done_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg[0] == "error":
                raise RuntimeError(f"postprocess worker failed on frame {msg[1]}: {msg[2]}")
            _, idx, slot, kp, sc, busy = msg
            self.busy["postprocess"] += busy
            self._finish_frame(idx, slot, (self.results[idx][0], self.boxes[idx], kp, sc))

    def sample_loop(self) -> None:
        q = self.pools
        while not self.finished.wait(0.1):
            self.qdepth.append({"det_q": self.det_q.qsize(), "pre_q": q.pre_q.qsize(),
                                "pre_done_q": q.pre_done_q.qsize(), "post_q": q.post_q.qsize(),
                                "free_slots": self.free_slots.qsize()})

    # -- run ------------------------------------------------------------------------------------
    def run(self) -> tuple[FrameLog, dict]:
        loops = [self.decode_loop, self.detect_loop, self.batch_loop, self.complete_loop,
                 self.collect_loop, self.sample_loop]
        threads = [threading.Thread(target=self._guard(f), name=f.__name__, daemon=True) for f in loops]
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        last_done, last_progress = -1, time.perf_counter()
        while not self.finished.wait(1.0):
            # A worker process that died (OOM kill, crash) would leave this run waiting forever.
            if dead := [p.pid for p in self.pools.procs if not p.is_alive()]:
                self.errors.append(RuntimeError(f"worker process(es) {dead} died"))
                break
            if self.done_frames != last_done:
                last_done, last_progress = self.done_frames, time.perf_counter()
            elif time.perf_counter() - last_progress > STALL_S:
                self.errors.append(RuntimeError(f"no frame finished for {STALL_S:.0f} s"))
                break
        wall_s = time.perf_counter() - t_start
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        if self.errors:
            raise RuntimeError(f"[stage8] pipeline thread failed: {self.errors[0]!r}") from self.errors[0]
        for t in threads:
            t.join(timeout=10)
        # A thread still running could consume the next run's queue messages: never go on.
        if alive := [t.name for t in threads if t.is_alive()]:
            raise RuntimeError(f"[stage8] pipeline threads {alive} did not finish after the last frame")
        if self.aborted:
            raise runner.Aborted()

        log = FrameLog()
        for idx in range(self.total_frames):
            n_det, boxes, kp, sc = self.results[idx]
            log.add(n_det, boxes, kp, sc)
        lat = np.array([(self.t_done[i] - self.t_decoded[i]) * 1000 for i in range(self.total_frames)])
        crops = sum(len(b) for b in self.boxes.values())
        engine_slots = sum(bt["engine_batch"] for bt in self.batches)
        gpu_pose_ms = sum(bt["gpu_ms"] for bt in self.batches)
        workers = {"preprocess": self.cfg.preprocess_workers, "postprocess": self.cfg.postprocess_workers}
        stats = {
            "frames": self.total_frames,
            "crops": crops,
            "wall_s": wall_s,
            "fps": self.total_frames / wall_s,
            "crops_per_s": crops / wall_s,
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                           "p99": float(np.percentile(lat, 99)), "max": float(lat.max()), "mean": float(lat.mean())},
            "latencies_ms": lat.astype(np.float32),        # per frame; aggregate() pools these
            "batches": len(self.batches),
            "batch_crops_hist": {str(k): v for k, v in sorted(collections.Counter(bt["crops"] for bt in self.batches).items())},
            "engine_slots": engine_slots,
            "pad_fraction": 1 - crops / engine_slots if engine_slots else 0.0,
            "pose_gpu_ms": gpu_pose_ms,
            "pose_gpu_ms_per_frame": gpu_pose_ms / self.total_frames,
            "pose_gpu_busy_share": gpu_pose_ms / 1000 / wall_s,
            # Busy share per stage: busy time / (wall x workers). The stage nearest 1.0 is the
            # one the others wait for.
            "stage_busy_share": {k: v / wall_s / workers.get(k, 1) for k, v in self.busy.items()},
            "stage_busy_s": dict(self.busy),
            "queue_depth_mean": {k: float(np.mean([d[k] for d in self.qdepth])) for k in self.qdepth[0]}
                                if self.qdepth else {},
            "main_process_cpu_cores_busy": ((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) / wall_s,
        }
        return log, stats


def aggregate(chunks: list[dict], cfg: AsyncConfig) -> dict:
    """Pool the per-chunk stats of async runs: throughput over summed wall time, latency
    percentiles over every frame, busy shares weighted by wall time."""
    frames = sum(c["frames"] for c in chunks)
    crops = sum(c["crops"] for c in chunks)
    wall = sum(c["wall_s"] for c in chunks)
    lat = np.concatenate([c["latencies_ms"] for c in chunks])
    slots = sum(c["engine_slots"] for c in chunks)
    workers = {"preprocess": cfg.preprocess_workers, "postprocess": cfg.postprocess_workers}
    busy = collections.Counter()
    for c in chunks:
        busy.update(c["stage_busy_s"])
    hist = collections.Counter()
    for c in chunks:
        hist.update({int(k): v for k, v in c["batch_crops_hist"].items()})
    wmean = lambda f: sum(f(c) * c["wall_s"] for c in chunks) / wall
    return {
        "frames": frames, "crops": crops, "wall_s": wall,
        "fps": frames / wall, "crops_per_s": crops / wall,
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                       "p99": float(np.percentile(lat, 99)), "max": float(lat.max()), "mean": float(lat.mean())},
        "batches": sum(c["batches"] for c in chunks),
        "batch_crops_hist": {str(k): hist[k] for k in sorted(hist)},
        "pad_fraction": 1 - crops / slots if slots else 0.0,
        "pose_gpu_ms_per_frame": sum(c["pose_gpu_ms"] for c in chunks) / frames,
        "pose_gpu_busy_share": sum(c["pose_gpu_ms"] for c in chunks) / 1000 / wall,
        "stage_busy_share": {k: v / wall / workers.get(k, 1) for k, v in busy.items()},
        "stage_busy_s": dict(busy),
        "pose_gpu_s": sum(c["pose_gpu_ms"] for c in chunks) / 1000,
        "gpu_util_mean_pct": wmean(lambda c: c["gpu"]["gpu_util_mean_pct"]),
        "sm_clock_mean_mhz": wmean(lambda c: c["gpu"]["sm_clock_mean_mhz"]),
        "main_process_cpu_cores_busy": wmean(lambda c: c["main_process_cpu_cores_busy"]),
        "queue_depth_mean": {k: wmean(lambda c: c["queue_depth_mean"].get(k, 0.0)) for k in chunks[0]["queue_depth_mean"]},
        "contaminated_attempts_discarded": sum(len(c["contaminated_attempts"]) for c in chunks),
    }


def aggregate_sync(chunks: list[dict]) -> dict:
    frames, wall = sum(c["frames"] for c in chunks), sum(c["wall_s"] for c in chunks)
    return {"frames": frames, "wall_s": wall, "fps": frames / wall,
            "cpu_cores_busy": sum(c["cpu_s"] for c in chunks) / wall,
            "contaminated_attempts_discarded": sum(len(c["contaminated_attempts"]) for c in chunks)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = AsyncConfig()
    p.add_argument("--backends", nargs="+", default=["trt-fp16", "pytorch"], choices=stages.BACKENDS)
    p.add_argument("--micro-batch", type=int, default=d.micro_batch, choices=ENGINE_SIZES)
    p.add_argument("--batch-timeout-ms", type=float, default=d.batch_timeout_ms)
    p.add_argument("--preprocess-workers", type=int, default=d.preprocess_workers)
    p.add_argument("--postprocess-workers", type=int, default=d.postprocess_workers)
    p.add_argument("--ring-slots", type=int, default=d.ring_slots)
    p.add_argument("--det-batches", nargs="+", type=int, default=[1, 4],
                   help="YOLO frames per call, one async variant each (1 = boxes bit-identical to Stage 7).")
    p.add_argument("--jobs-dir", type=Path, default=stages.default_jobs_dir())
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--chunk-frames", type=int, default=1000, help="Frames per measurement unit (pipeline/runner.py).")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint tag of an interrupted run to continue.")
    p.add_argument("--warmup-frames", type=int, default=150)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "pipeline")
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "results" / "raw" / "pipeline")
    return p.parse_args()


class AsyncGpuState(runner.GpuResources):
    """Everything Stages 8-9 hold on the GPU: the detector, each backend's engines/model and an
    AsyncPose per (backend, micro-batch). Released while a prod job runs (pipeline/runner.py)."""

    def __init__(self, names: list[str], micro_batches: list[int], first_frame: np.ndarray):
        self.names, self.micro_batches, self.first_frame = names, micro_batches, first_frame

    def load(self) -> None:
        # One detect, one copy and one pose stream for every configuration: only one runs at a
        # time, and PyTorch's streams come from a round-robin pool of 32, so a fresh stream per
        # Run or per AsyncPose eventually aliases one already in use (detection then silently
        # serializes with the pose path). Run() asserts the three are distinct.
        self.det_stream, copy_stream, pose_stream = (torch.cuda.Stream() for _ in range(3))
        self.detector = stages.load_detector()
        for _ in range(5):
            stages.detect(self.detector, self.first_frame)          # builds the predictor detect_frames() reuses
            stages.detect_frames(self.detector, [self.first_frame] * 4)
        sizes = tuple(b for b in ENGINE_SIZES if b <= max(max(self.micro_batches), stages.MAX_PERSONS))
        self.backends = {name: stages.load_backend(name, sizes) for name in self.names}
        self.poses = {(name, b): AsyncPose(be, b, (copy_stream, pose_stream))
                      for name, be in self.backends.items() for b in self.micro_batches}
        for be in self.backends.values():                           # the sync path's batch sizes
            for n in range(1, stages.MAX_PERSONS + 1):
                be.download(be.forward(be.upload(np.zeros((n, *stages.POSE_INPUT), np.float16))))
        for pose in self.poses.values():                            # every engine and buffer a batch can land on
            for i in range(N_BUFFERS):
                for n in sorted({1, *(b for b in ENGINE_SIZES if b <= pose.mb)}):
                    pose.launch(i, n)[2].synchronize()

    def run(self, key: tuple[str, int], video: dict, pools: "CpuPools", cfg: AsyncConfig, frames: int | None,
            start: int = 0, cap=None) -> tuple:
        return Run(video, self.detector, pools, self.poses[key], cfg, frames, start, self.det_stream, cap).run()

    def release(self) -> None:
        del self.detector, self.backends, self.poses, self.det_stream


BOX_TOL_BATCHED_DET_PX = 0.5   # a batched YOLO call rounds differently: <= 0.13 px measured on 128 frames


def main() -> None:
    args = parse_args()
    cfg = AsyncConfig(args.micro_batch, args.batch_timeout_ms, args.preprocess_workers,
                      args.postprocess_workers, args.ring_slots)
    variants = {f"det_batch={d}": replace(cfg, det_batch=d) for d in args.det_batches}
    videos = stages.workload_videos(args.jobs_dir)
    processor = stages.load_processor()
    frames0 = []
    for v in videos:
        c = stages.open_video(v["path"])
        frames0.append(c.read()[1])
        c.release()
    if len({f.shape for f in frames0}) != 1:
        raise SystemExit(f"[stage8] videos differ in frame shape {[f.shape for f in frames0]}: one ring per shape")
    pools = CpuPools(cfg, frames0[0].shape)
    gpu = AsyncGpuState(args.backends, [cfg.micro_batch], frames0[0])

    def warm() -> None:
        for name in args.backends:
            run_pass(videos[0], gpu.detector, processor, gpu.backends[name], False, args.warmup_frames)
            for vcfg in variants.values():
                gpu.run((name, cfg.micro_batch), videos[0], pools, vcfg, args.warmup_frames)

    gpu.warm = warm
    gpu.ensure_loaded()
    env = env_fingerprint()
    own = {os.getpid()} | {p.pid for p in pools.procs}
    tag = args.resume or runner.new_tag("stage8")
    store = runner.Store(tag)
    units = runner.chunk_units(videos, args.chunk_frames, args.max_frames)
    print(f"[stage8] config {asdict(cfg)}, async variants {list(variants)}; {len(units)} chunks x "
          f"{len(args.backends)} backends; checkpoints {store.dir} (resume with --resume {tag})", flush=True)
    done = {}
    try:
        for unit in units:
            for name in args.backends:
                v, n, start = unit["video"], unit["frames"], unit["start"]
                modes = [("sync", lambda cap: run_pass(v, gpu.detector, processor, gpu.backends[name], False, n,
                                                       start, cap))]
                modes += [(vname, lambda cap, c=vcfg: gpu.run((name, cfg.micro_batch), v, pools, c, n, start, cap))
                          for vname, vcfg in variants.items()]
                for mode, fn in modes:
                    key = f"{name}/{mode}/{unit['key']}"
                    if (hit := store.get(key)) is None:
                        hit = runner.guarded(fn, own, label=key, resources=gpu,
                                             prepare=lambda: runner.open_at(v, start))
                        store.put(key, hit)
                    done[key] = hit
                s_ = done[f"{name}/sync/{unit['key']}"][1]
                sf = s_["frames"] / s_["wall_s"]
                line = []
                for vname in variants:
                    a_ = done[f"{name}/{vname}/{unit['key']}"][1]
                    af = a_["frames"] / a_["wall_s"]
                    line.append(f"{vname} {af:5.1f} fps ({af / sf:.2f}x, detect busy {a_['stage_busy_share']['detect']:.2f})")
                print(f"[stage8] {unit['key']} {name:8s} sync {sf:5.1f} fps | " + " | ".join(line), flush=True)
    finally:
        pools.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    summary = {"stage": 8, "config": asdict(cfg), "variants": {k: asdict(c) for k, c in variants.items()},
               "backends": {}, "created_utc": datetime.now(timezone.utc).isoformat()}
    for name in args.backends:
        per = lambda mode, v: [done[f"{name}/{mode}/{u['key']}"] for u in units if u["video"] is v]
        sync_out = {v["job_id"]: merge_logs([h[0] for h in per("sync", v)]).arrays() for v in videos}
        sync_chunks = {v["job_id"]: [h[1] for h in per("sync", v)] for v in videos}
        total_sync = aggregate_sync([c for cs in sync_chunks.values() for c in cs])
        precision = "int8" if name == "trt-int8" else "fp16"
        report = {"stage": 8, "backend": name, "config": asdict(cfg),
                  "workload": [{"job_id": v["job_id"], "fingerprint": v["fingerprint"]} for v in videos],
                  "sync": {"total": total_sync, "per_video": {j: aggregate_sync(cs) for j, cs in sync_chunks.items()}},
                  "async": {}, "checkpoint_tag": tag, "env": env}
        summary["backends"][name] = {"sync_fps": total_sync["fps"], "async": {}}
        for vname, vcfg in variants.items():
            async_out = {v["job_id"]: merge_logs([h[0] for h in per(vname, v)]).arrays() for v in videos}
            chunks = {v["job_id"]: [h[1] for h in per(vname, v)] for v in videos}
            total = aggregate([c for cs in chunks.values() for c in cs], vcfg)
            rep = compare.compare_runs(sync_out, async_out, 0.0 if vcfg.det_batch == 1 else BOX_TOL_BATCHED_DET_PX)
            rep["gate_failures"] = compare.gate_failures(rep, precision)
            raw = {f"{j}__{k}": a for j, arrs in async_out.items() for k, a in arrs.items()}
            raw_path = args.raw_dir / f"async_{name}_{vname.replace('=', '')}.npz"
            np.savez_compressed(raw_path, **raw)
            report["async"][vname] = {
                "config": asdict(vcfg), "total": total, "speedup_vs_sync": total["fps"] / total_sync["fps"],
                "per_video": {j: {**aggregate(cs, vcfg), "speedup_vs_sync": aggregate(cs, vcfg)["fps"]
                                  / aggregate_sync(sync_chunks[j])["fps"]} for j, cs in chunks.items()},
                "equivalence_vs_sync": rep, "outputs_sha256": stages.outputs_digest(raw),
                "raw_outputs": repo_relative(raw_path)}
            summary["backends"][name]["async"][vname] = {
                "fps": total["fps"], "speedup_vs_sync": total["fps"] / total_sync["fps"],
                "latency_ms": total["latency_ms"], "stage_busy_share": total["stage_busy_share"],
                "pose_gpu_busy_share": total["pose_gpu_busy_share"], "gpu_util_mean_pct": total["gpu_util_mean_pct"],
                "pad_fraction": total["pad_fraction"],
                "frames": rep["frames"], "frames_boxes_identical": rep["frames_boxes_identical"],
                "frames_boxes_differ": rep["frames_boxes_differ"], "box_max_abs_diff_px": rep["box_max_abs_diff_px"],
                "keypoint_err_crop_px": rep["keypoint_err_crop_px"], "gate_failures": rep["gate_failures"]}
            kp = rep["keypoint_err_crop_px"]
            print(f"[stage8] {name} {vname}: sync {total_sync['fps']:.1f} fps -> async {total['fps']:.1f} fps "
                  f"({total['fps'] / total_sync['fps']:.2f}x); boxes identical {rep['frames_boxes_identical']}/"
                  f"{rep['frames']}, compared {rep['frames_same_boxes']}; keypoints vs sync median {kp['median']:.4f} px "
                  f"p95 {kp['p95']:.4f} -- "
                  f"{'gates OK' if not rep['gate_failures'] else 'GATES FAILED: ' + '; '.join(rep['gate_failures'])}",
                  flush=True)
        (args.output_dir / f"async_{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "async_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
