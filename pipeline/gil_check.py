#!/usr/bin/env python3
"""Stage 8 design check: do HF's crop preprocess and pose decode run in parallel across threads,
or only across processes?

    python -m pipeline.gil_check

Times each stage over the same 48 real frames (4 people each, the per-frame cap) with 1 thread,
N threads and N processes. Every worker decodes its own copy of the frames up front and a task
carries only a frame index, so the process numbers don't include pickling frames (Stage 8
passes frames through shared memory, not pipes). One intra-op thread everywhere
(torch/cv2), so the pool size is the only parallelism.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline import REPO_ROOT, resolve_checkpoint
from pipeline.cpu_stages import load_processor, postprocess, preprocess

N_FRAMES = 48
BOXES = np.array([[400, 150, 120, 330], [700, 160, 110, 300], [100, 200, 60, 160], [900, 220, 50, 140]], np.float32)
_state: dict = {}


def _init(video: str) -> None:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    _state["frames"] = [cap.read()[1] for _ in range(N_FRAMES)]
    _state["processor"] = load_processor(resolve_checkpoint(None))
    _state["heatmaps"] = np.random.default_rng(0).random((4, 17, 64, 48)).astype(np.float16)


def _pre(i: int) -> int:
    return preprocess(_state["processor"], _state["frames"][i], BOXES).shape[0]


def _post(i: int) -> int:
    return postprocess(_state["processor"], _state["heatmaps"], BOXES)[0].shape[0]


def _bench(pool, fn) -> float:
    list(pool.map(fn, range(8)))                                  # warm
    t = time.perf_counter()
    list(pool.map(fn, range(N_FRAMES)))
    return N_FRAMES / (time.perf_counter() - t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Any 720p source video, e.g. a Stage 6 held-out one.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "pipeline" / "gil_check.json")
    args = p.parse_args()
    _init(args.video)
    out = {}
    for name, fn in (("preprocess", _pre), ("postprocess", _post)):
        with cf.ThreadPoolExecutor(1) as ex:
            one = _bench(ex, fn)
        with cf.ThreadPoolExecutor(args.workers) as ex:
            threads = _bench(ex, fn)
        with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                    initargs=(args.video,)) as ex:
            procs = _bench(ex, fn)
        out[name] = {"frames_per_s": {"1_thread": one, f"{args.workers}_threads": threads,
                                      f"{args.workers}_processes": procs},
                     "thread_scaling": threads / one, "process_scaling": procs / one}
        print(f"[gil_check] {name:11s} 1 thread {one:6.1f} fps | {args.workers} threads {threads:6.1f} "
              f"({threads / one:.2f}x) | {args.workers} processes {procs:6.1f} ({procs / one:.2f}x)")
    out["method"] = f"{N_FRAMES} real 1280x720 frames x 4 fixed boxes, torch/cv2 at 1 intra-op thread"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
