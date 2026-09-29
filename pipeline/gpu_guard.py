"""Keep the shared box's other tenants out of Stage 7-9 measurements, without ever blocking them.

This L4 box also runs AutoClipping's production service (the pipeline this repo's work feeds)
and other people's sessions. Prod analysis jobs serialize on `fcntl.flock` slots --
`data/.gpu.lock` plus `.gpu.lock.<i>` siblings, one per `worker.concurrency` slot
(autoclip/core/gpu_lock.py there) -- but other GPU users don't take that lock. A pipeline
benchmark that overlaps any of them measures two workloads at once, and because the video
pipeline is largely CPU-bound, a tenant that only uses CPU (a prod job decoding video) skews it
too.

Taking the prod lock ourselves would keep prod off the GPU for the length of a benchmark, which
is not ours to do. Instead this module only OBSERVES:

  - `prod_lock_holders()` reads /proc/locks for any flock on those lockfiles' inodes, so it
    never opens or locks them;
  - `wait_until_quiet()` blocks before a measurement until no prod job holds a slot, the GPU
    has been idle and the box's foreign CPU load low for a sustained window;
  - `GpuMonitor` samples every 0.5 s while a measurement runs: prod lock holders, compute
    processes that aren't ours, GPU utilization and SM clock, and CPU time used by processes
    other than ours.

A measurement whose monitor saw any of that is marked contaminated, and pipeline/runner.py
re-runs it.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pynvml

from baseline import REPO_ROOT

PROD_LOCK_DIR = Path(os.environ.get("AUTOCLIP_DATA_DIR", REPO_ROOT.parent / "AutoClipping" / "data"))
SAMPLE_SEC = 0.5
# Foreign CPU (cores busy outside our own processes) above which a window counts as contaminated.
# The box idles at ~0.2 cores (IDE servers, other agent sessions); a prod job decoding video uses
# several.
FOREIGN_CPU_MAX_CORES = 1.5
HZ = os.sysconf("SC_CLK_TCK")


def _lock_ids() -> set[tuple[int, int, int]]:
    """(major, minor, inode) of every AutoClipping GPU lockfile that exists."""
    ids = set()
    for p in PROD_LOCK_DIR.glob(".gpu.lock*"):
        st = p.stat()
        ids.add((os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))
    return ids


def prod_lock_holders() -> list[int]:
    """PIDs holding a flock on an AutoClipping GPU lockfile right now (empty = no prod job)."""
    ids = _lock_ids()
    if not ids:
        return []
    holders = []
    with open("/proc/locks") as f:
        for line in f:
            parts = line.split()
            # "1: FLOCK  ADVISORY  WRITE 798 103:02:1841898 0 EOF" -- a "->" marks a waiter.
            if "->" in parts or len(parts) < 6 or parts[1] != "FLOCK":
                continue
            major, minor, inode = parts[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) in ids:
                holders.append(int(parts[4]))
    return holders


def _system_busy_ticks() -> int:
    fields = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(fields) - fields[3] - fields[4]          # minus idle and iowait


def _process_ticks(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            # comm (field 2) may contain spaces; utime/stime are fields 14/15, counted after ')'.
            rest = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            total += int(rest[11]) + int(rest[12])
        except (FileNotFoundError, ProcessLookupError, IndexError):
            pass
    return total


class _CpuMeter:
    """Cores busy outside `own_pids` since the last call. Own time is read from each process's
    /proc/<pid>/stat (all its threads); a process that exits between samples drops out of the
    subtraction and reads as foreign for that one interval, so callers pass long-lived pids."""

    def __init__(self, own_pids: set[int]):
        self.own = own_pids
        self.t, self.sys, self.proc = time.monotonic(), _system_busy_ticks(), _process_ticks(own_pids)

    def foreign_cores(self) -> float:
        t, sys_, proc = time.monotonic(), _system_busy_ticks(), _process_ticks(self.own)
        cores = ((sys_ - self.sys) - (proc - self.proc)) / HZ / max(t - self.t, 1e-3)
        self.t, self.sys, self.proc = t, sys_, proc
        return max(cores, 0.0)


def _foreign_gpu_procs(handle, own_pids: set[int]) -> dict[int, int]:
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return {p.pid: (p.usedGpuMemory or 0) >> 20 for p in procs if p.pid not in own_pids}


def wait_until_quiet(own_pids: set[int] | None = None, quiet_s: float = 15.0, gpu_util_max: int = 5,
                     log=print, on_prod=None) -> float:
    """Block until no prod job holds a GPU slot, GPU utilization has stayed <= gpu_util_max and
    foreign CPU <= FOREIGN_CPU_MAX_CORES for quiet_s seconds in a row. Returns seconds waited.
    Our own processes are idle while this runs, so any GPU activity it sees is someone else's.
    on_prod() is called whenever a prod job is seen holding the lock (to give back VRAM)."""
    own = own_pids or {os.getpid()}
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meter = _CpuMeter(own)
    t0 = quiet_since = time.monotonic()
    last_log = 0.0
    while True:
        time.sleep(1.0)
        now = time.monotonic()
        holders = prod_lock_holders()
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        cpu = meter.foreign_cores()
        if holders or util > gpu_util_max or cpu > FOREIGN_CPU_MAX_CORES:
            quiet_since = now
            if holders and on_prod is not None:
                on_prod()
            if now - last_log > 300:
                log(f"[gpu_guard] box busy (prod lock holders {holders}, GPU util {util}%, foreign CPU "
                    f"{cpu:.1f} cores) -- waiting, not competing ({now - t0:.0f}s so far)")
                last_log = now
        elif now - quiet_since >= quiet_s:
            waited = now - t0
            if waited > quiet_s + 5:
                log(f"[gpu_guard] quiet after {waited:.0f}s")
            return waited


def wait_for_no_prod(log=print, poll_s: float = 10.0) -> None:
    """For offline GPU work that isn't timed (accuracy evaluations, engine builds): start only
    while no prod job holds a GPU slot, so the memory it takes can't be what a prod job fails to
    allocate. Weaker than wait_until_quiet(): other tenants' load doesn't matter here."""
    waited = 0.0
    while prod_lock_holders():
        if waited % 300 < poll_s:
            log(f"[gpu_guard] prod job running -- waiting before GPU work ({waited:.0f}s so far)")
        time.sleep(poll_s)
        waited += poll_s


class GpuMonitor:
    """Background sampler for one measurement window: `with GpuMonitor(own_pids) as mon: ...`,
    then `mon.report()`. Foreign GPU processes already resident at the start are tolerated (the
    start follows wait_until_quiet(), so they were idle -- the prod web server keeps its models
    loaded) unless their VRAM grows; a new one, a prod lock holder, or foreign CPU above
    FOREIGN_CPU_MAX_CORES on average contaminates the window."""

    def __init__(self, own_pids: set[int] | None = None, abort: threading.Event | None = None):
        self.own_pids = own_pids or {os.getpid()}
        self.abort = abort
        self._stop = threading.Event()
        self.samples: list[dict] = []

    def _run(self) -> None:
        while not self._stop.wait(SAMPLE_SEC):
            holders = prod_lock_holders()
            foreign = _foreign_gpu_procs(self._handle, self.own_pids)
            # A prod job or a new GPU process means this window is lost anyway: tell the
            # measurement to stop now, so it gives the GPU (and its VRAM) back within a second
            # instead of competing with prod until the chunk ends.
            if self.abort is not None and (holders or set(foreign) - set(self.resident_at_start)):
                self.abort.set()
            self.samples.append({
                "lock_holders": holders,
                "foreign": foreign,
                "util": pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu,
                "sm_mhz": pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
                "foreign_cpu": self._cpu.foreign_cores(),
            })

    def __enter__(self) -> "GpuMonitor":
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.resident_at_start = _foreign_gpu_procs(self._handle, self.own_pids)
        self.lock_holders_at_start = prod_lock_holders()
        self._cpu = _CpuMeter(self.own_pids)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.lock_holders_at_end = prod_lock_holders()

    def report(self) -> dict:
        holders = set(self.lock_holders_at_start) | set(self.lock_holders_at_end)
        new_pids, grew = set(), set()
        for s in self.samples:
            holders.update(s["lock_holders"])
            for pid, mb in s["foreign"].items():
                if pid not in self.resident_at_start:
                    new_pids.add(pid)
                elif mb > self.resident_at_start[pid] + 256:
                    grew.add(pid)
        n = max(1, len(self.samples))
        foreign_cpu = sum(s["foreign_cpu"] for s in self.samples) / n
        return {
            "clean": not holders and not new_pids and not grew and foreign_cpu <= FOREIGN_CPU_MAX_CORES,
            "prod_lock_holders": sorted(holders),
            "new_foreign_gpu_pids": sorted(new_pids),
            "foreign_pids_that_grew_vram": sorted(grew),
            "resident_foreign_pids": sorted(self.resident_at_start),
            "foreign_cpu_cores_mean": foreign_cpu,
            "foreign_cpu_cores_max": max((s["foreign_cpu"] for s in self.samples), default=0.0),
            "samples": len(self.samples),
            "gpu_util_mean_pct": sum(s["util"] for s in self.samples) / n,
            "sm_clock_mean_mhz": sum(s["sm_mhz"] for s in self.samples) / n,
            "power_mean_w": sum(s["power_w"] for s in self.samples) / n,
        }
