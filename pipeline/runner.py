"""Run pipeline measurements as short, checkpointed, contention-guarded units (Stages 7-9).

The box is shared (pipeline/gpu_guard.py), and a production job can start at any moment. A
whole-video pass over the 5374-frame bowler clip takes 3-5 minutes, long enough that on a busy
evening most attempts overlap something. So the workload is cut into chunks of consecutive
frames (default 1000), each measured as its own unit:

  - a unit starts only once the box is quiet (gpu_guard.wait_until_quiet);
  - it runs under gpu_guard.GpuMonitor, and is simply re-run if anything overlapped it, with no
    attempt limit -- waiting is cheap, a contaminated number is not;
  - each clean unit is pickled to a checkpoint directory at once, so an interrupted run resumes
    where it stopped (--resume <tag>) instead of losing the units already measured.

A chunk starts by grab()-ing up to its first frame outside the timed region: the decoder then
yields exactly the frames a whole-video read would, in the same order, so chunk outputs
concatenate into the whole-video outputs.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path

from baseline import REPO_ROOT
from pipeline import gpu_guard, stages

CHECKPOINT_ROOT = REPO_ROOT / "results" / "raw" / "pipeline" / "checkpoints"
_last_clean_end = float("-inf")


def count_frames(path: Path) -> int:
    """Frames cv2 actually decodes (container frame counts can disagree on phone footage)."""
    cap = stages.open_video(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def chunk_units(videos: list[dict], chunk_frames: int, max_frames: int | None = None) -> list[dict]:
    """[{video, start, frames, key}] covering each video's first max_frames (or all) frames."""
    units = []
    for v in videos:
        total = v.get("frames") or count_frames(v["path"])
        v["frames"] = total
        total = min(total, max_frames) if max_frames else total
        for start in range(0, total, chunk_frames):
            n = min(chunk_frames, total - start)
            units.append({"video": v, "start": start, "frames": n, "key": f"{v['job_id']}@{start}+{n}"})
    return units


def skip_frames(cap, n: int) -> None:
    for _ in range(n):
        if not cap.grab():
            raise RuntimeError(f"video ended while skipping to frame {n}")


def open_at(video: dict, start: int):
    """A capture positioned at frame `start` of `video`. Callers do this outside both the timed
    region and the contention monitor's window (guarded(prepare=...)): skipping to frame 4000
    decodes 4000 frames with the GPU idle, which would dilute the window's averages."""
    cap = stages.open_video(video["path"])
    skip_frames(cap, start)
    return cap


class Store:
    """Pickled unit results under results/raw/pipeline/checkpoints/<tag>/ (gitignored)."""

    def __init__(self, tag: str):
        self.dir = CHECKPOINT_ROOT / tag
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / (key.replace("/", "_") + ".pkl")

    def get(self, key: str):
        p = self._path(key)
        return pickle.loads(p.read_bytes()) if p.is_file() else None

    def put(self, key: str, value) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(value))
        tmp.replace(self._path(key))


def new_tag(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}"


# Set by the running unit's GpuMonitor the moment a prod job or a new GPU process appears; the
# pipelines poll it per frame and stop early (sync: at once; async: stop decoding, drain).
ABORT = threading.Event()


class Aborted(Exception):
    """A measurement stopped early because the box stopped being quiet."""


class GpuResources:
    """What a benchmark holds on the GPU, so it can be given back while a prod job runs. A prod
    bowler job takes ~7 GB and prod runs up to 3 at once (AutoClipping config/l4.yaml), which is
    the whole L4 together with its web server: memory held by an idle benchmark could make a
    prod job fail to allocate. Subclasses implement load() (build + warm up) and release()."""

    loaded = False
    warm = None     # optional callable, run after every (re)load: a unit after a reload isn't measured cold

    def load(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def ensure_loaded(self, log=print) -> bool:
        """Load (and warm) if released. Returns True if it loaded now."""
        if self.loaded:
            return False
        t = time.perf_counter()
        self.load()
        self.loaded = True
        if self.warm is not None:
            self.warm()
        log(f"[runner] GPU state loaded + warmed up ({time.perf_counter() - t:.0f}s)")
        return True

    def ensure_released(self, log=print) -> None:
        if self.loaded:
            import gc

            import torch
            self.release()
            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = False
            log(f"[runner] released GPU memory while the box is busy "
                f"({torch.cuda.memory_reserved() / 2**20:.0f} MB still reserved by PyTorch)")


def guarded(fn, own_pids: set[int] | None = None, label: str = "", log=print,
            resources: GpuResources | None = None, prepare=None):
    """Run fn() (-> (result, stats dict)) until one run overlaps nothing on the box. Returns
    (result, stats) with stats["gpu"] = the clean window's monitor report and
    stats["contaminated_attempts"] = the discarded ones. While the box is busy, `resources`
    (if given) are released, and reloaded once it is quiet. With `prepare`, each attempt calls
    prepare() before the monitor starts and then fn(prepared)."""
    global _last_clean_end
    own = own_pids or {os.getpid()}
    discarded = []
    while True:
        # A unit that follows a clean one within seconds skips the quiet window: the monitor
        # of that unit just watched the box, and this one's will catch whatever starts now.
        if discarded or time.monotonic() - _last_clean_end > 10 or gpu_guard.prod_lock_holders():
            gpu_guard.wait_until_quiet(own, log=log, on_prod=None if resources is None
                                       else lambda: resources.ensure_released(log))
        ABORT.clear()
        if resources is not None and resources.ensure_loaded(log):
            # Something may have started while we reloaded; it must not count as resident.
            gpu_guard.wait_until_quiet(own, quiet_s=10.0, log=log)
        prepared = prepare() if prepare is not None else None
        aborted = False
        with gpu_guard.GpuMonitor(own, abort=ABORT) as mon:
            try:
                result, stats = fn(prepared) if prepare is not None else fn()
            except Aborted:
                aborted = True
        rep = mon.report()
        if aborted and rep["clean"]:
            # Aborted on a transient signal that the full report doesn't count against the
            # window (it can't happen by construction, but never return a partial result).
            rep["clean"] = False
            rep["aborted_without_cause"] = True
        if rep["clean"]:
            _last_clean_end = time.monotonic()
            stats["gpu"] = rep
            stats["contaminated_attempts"] = discarded
            return result, stats
        discarded.append(rep)
        if resources is not None and (rep["prod_lock_holders"] or rep["new_foreign_gpu_pids"]):
            resources.ensure_released(log)
        why = ["stopped early"] if aborted else []
        if rep["prod_lock_holders"]:
            why.append(f"prod job {rep['prod_lock_holders']}")
        if rep["new_foreign_gpu_pids"] or rep["foreign_pids_that_grew_vram"]:
            why.append(f"GPU pids {rep['new_foreign_gpu_pids'] + rep['foreign_pids_that_grew_vram']}")
        if rep["foreign_cpu_cores_mean"] > gpu_guard.FOREIGN_CPU_MAX_CORES:
            why.append(f"foreign CPU {rep['foreign_cpu_cores_mean']:.1f} cores")
        log(f"[runner] {label}: overlapped {', '.join(why)} -- discarded, re-running (attempt {len(discarded) + 1})")
