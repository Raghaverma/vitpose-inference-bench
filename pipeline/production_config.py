#!/usr/bin/env python3
"""Stage 9, step 2: the production configuration -- which pose backend and micro-batch give the
most useful throughput on the real workload.

    python -m pipeline.workload_profile          # step 1: people per frame on every job video
    python -m pipeline.production_config         # step 2: this sweep + the recommendation

"Useful" throughput counts real crops and frames, never padded engine slots.

Phase 1 -- micro-batch sweep. Stage 8's asynchronous pipeline at every static engine batch
size, 1-16, for each backend, on the same chunked footage as Stages 7-8, interleaved per chunk
so clock drift lands on every configuration alike. Every configuration's poses are checked
against Stage 7's PyTorch FP16 synchronous run (results/raw/pipeline/sync_pytorch.npz) with its
precision's gates (pipeline/compare.py); a configuration that fails them is reported but not
eligible.

All of it runs Stage 8's faster variant, YOLO on 4 frames per call (--det-batch), so poses are
compared with pipeline/async_pipeline.BOX_TOL_BATCHED_DET_PX of box tolerance.

Phase 1b -- the flush timeout (how long a partial micro-batch waits for more crops), at the
fastest eligible configuration, the default included, interleaved per chunk.

Phase 2 -- crops-per-frame sweep. The benchmark footage averages one crop count and the fleet
of job videos another (results/pipeline/workload_profile.json), and the chunks of the
benchmark footage all sit close to it, so they can't say how throughput moves with it. So the
fastest eligible configuration of each backend is re-run with the person cap at 1, 2, 3 and 4,
interleaved per chunk: same frames, fewer crops per frame. Throughput at the workload's crops per
frame is then read off that measured curve by linear interpolation of time per frame, and at
most MAX_EXTRAPOLATION crops/frame past its top (flagged as extrapolated).

Phase 3 -- the recommendation: the eligible configuration with the highest projected
throughput on the workload, preferring less engine VRAM and lower latency among those within
PROJECTION_TIE of the best.

Outputs: results/pipeline/production_config.json (committed), per-config per-frame outputs in
results/raw/pipeline/stage9_<backend>_b<B>.npz (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baseline import REPO_ROOT
from pipeline import compare, runner, stages
from pipeline.async_pipeline import (BOX_TOL_BATCHED_DET_PX, ENGINE_SIZES, AsyncConfig, AsyncGpuState, CpuPools,
                                     aggregate)
from pipeline.sync_pipeline import env_fingerprint, merge_logs

RESULTS = REPO_ROOT / "results" / "pipeline"
RAW = REPO_ROOT / "results" / "raw" / "pipeline"
CAPS = (1, 2, 3)
PROJECTION_TIE = 0.03


def load_run_outputs(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k in z.files:
        job, field = k.split("__", 1)
        if field in compare.FIELDS:
            out.setdefault(job, {})[field] = z[k]
    return out


def engine_vram_mb(backend: str, sizes: tuple[int, ...]) -> float | None:
    """Engine file sizes (~ the weights' device footprint) of the engines a config loads."""
    if backend == "pytorch":
        return None
    prefix = "int8_engine" if backend == "trt-int8" else "engine"
    total = 0.0
    for b in sizes:
        sfx = "" if b == 1 else f"_b{b}"
        m = json.loads((REPO_ROOT / "results" / "tensorrt" / f"{prefix}_metadata{sfx}.json").read_text())
        total += Path(m["engine_path"]).stat().st_size / 2**20
    return total


MAX_EXTRAPOLATION = 0.5   # crops/frame past the top of the measured curve


def interpolate_fps(curve: list[dict], crops_per_frame: float, max_extrapolation: float = MAX_EXTRAPOLATION) -> dict:
    """curve: [{crops_per_frame, fps}] measured at several person caps. Linear in time per frame
    between the two nearest measured points. Up to max_extrapolation above the measured range,
    a least-squares line through the top three points, flagged; anything else is refused."""
    pts = sorted((p["crops_per_frame"], 1.0 / p["fps"]) for p in curve)
    xs, ts = zip(*pts)
    if xs[0] <= crops_per_frame <= xs[-1]:
        return {"fps": float(1.0 / np.interp(crops_per_frame, xs, ts)), "extrapolated": False}
    if xs[-1] < crops_per_frame <= xs[-1] + max_extrapolation and len(xs) >= 3:
        slope, icpt = np.polyfit(xs[-3:], ts[-3:], 1)
        return {"fps": float(1.0 / (icpt + slope * crops_per_frame)), "extrapolated": True}
    return {"fps": None, "note": f"{crops_per_frame:.2f} crops/frame is outside the measured "
                                 f"{xs[0]:.2f}-{xs[-1]:.2f} (+{max_extrapolation:.2f})"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = AsyncConfig()
    p.add_argument("--backends", nargs="+", default=["trt-fp16", "trt-int8", "pytorch"], choices=stages.BACKENDS)
    p.add_argument("--micro-batches", nargs="+", type=int, default=list(ENGINE_SIZES))
    p.add_argument("--batch-timeout-ms", type=float, default=d.batch_timeout_ms)
    p.add_argument("--extra-timeouts-ms", nargs="*", type=float, default=[10.0, 200.0],
                   help="Also measure these flush timeouts at the fastest eligible configuration.")
    p.add_argument("--det-batch", type=int, default=4,
                   help="YOLO frames per call (Stage 8: 4 lifts the detector bottleneck; boxes move <= 0.13 px).")
    p.add_argument("--preprocess-workers", type=int, default=d.preprocess_workers)
    p.add_argument("--postprocess-workers", type=int, default=d.postprocess_workers)
    p.add_argument("--ring-slots", type=int, default=d.ring_slots)
    p.add_argument("--jobs-dir", type=Path, default=stages.default_jobs_dir())
    p.add_argument("--chunk-frames", type=int, default=1000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--warmup-frames", type=int, default=150)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--profile", type=Path, default=RESULTS / "workload_profile.json")
    p.add_argument("--output", type=Path, default=RESULTS / "production_config.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text())
    base = AsyncConfig(max(args.micro_batches), args.batch_timeout_ms, args.preprocess_workers,
                       args.postprocess_workers, args.ring_slots, det_batch=args.det_batch)
    box_tol = 0.0 if args.det_batch == 1 else BOX_TOL_BATCHED_DET_PX
    videos = stages.workload_videos(args.jobs_dir)
    reference = load_run_outputs(RAW / "sync_pytorch.npz")
    cap = stages.open_video(videos[0]["path"])
    frame0 = cap.read()[1]
    cap.release()
    pools = CpuPools(base, frame0.shape)
    gpu = AsyncGpuState(args.backends, args.micro_batches, frame0)
    gpu.warm = lambda: [gpu.run(key, videos[0], pools, replace(base, micro_batch=key[1]), args.warmup_frames)
                        for key in gpu.poses]
    gpu.ensure_loaded()
    own = {os.getpid()} | {p.pid for p in pools.procs}
    tag = args.resume or runner.new_tag("stage9")
    store = runner.Store(tag)
    units = runner.chunk_units(videos, args.chunk_frames, args.max_frames)
    print(f"[stage9] {len(units)} chunks x {len(args.backends)} backends x micro-batches {args.micro_batches}; "
          f"checkpoints {store.dir} (resume with --resume {tag})", flush=True)

    def measure(phase: str, name: str, b: int, persons: int, unit: dict, timeout_ms: float | None = None):
        timeout_ms = base.batch_timeout_ms if timeout_ms is None else timeout_ms
        key = f"{phase}/{name}/b{b}/cap{persons}/t{timeout_ms:g}/{unit['key']}"
        if (hit := store.get(key)) is None:
            cfg = replace(base, micro_batch=b, max_persons=persons, batch_timeout_ms=timeout_ms)
            v, n, start = unit["video"], unit["frames"], unit["start"]
            hit = runner.guarded(lambda cap: gpu.run((name, b), v, pools, cfg, n, start, cap), own, label=key,
                                 resources=gpu, prepare=lambda: runner.open_at(v, start))
            store.put(key, hit)
            print(f"[stage9] {key}: {hit[1]['fps']:.1f} fps, latency p50 {hit[1]['latency_ms']['p50']:.0f} ms",
                  flush=True)
        return hit

    def summarize_variant(hits: list, cfg: AsyncConfig) -> dict:
        agg = aggregate([h[1] for h in hits], cfg)
        return {"crops_per_frame": agg["crops"] / agg["frames"], "fps": agg["fps"], "latency_ms": agg["latency_ms"],
                "pad_fraction": agg["pad_fraction"], "stage_busy_share": agg["stage_busy_share"]}

    rows, curves, timeout_rows = [], {}, []
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        # Phase 1: every (backend, micro-batch), person cap 4, interleaved per chunk.
        done = {(name, b, u["key"]): measure("p1", name, b, stages.MAX_PERSONS, u)
                for u in units for name in args.backends for b in args.micro_batches}
        for name in args.backends:
            precision = "int8" if name == "trt-int8" else "fp16"
            for b in args.micro_batches:
                cfg = replace(base, micro_batch=b)
                outputs, per_video, all_chunks = {}, {}, []
                for v in videos:
                    hits = [done[(name, b, u["key"])] for u in units if u["video"] is v]
                    outputs[v["job_id"]] = merge_logs([h[0] for h in hits]).arrays()
                    per_video[v["job_id"]] = aggregate([h[1] for h in hits], cfg)
                    all_chunks += [h[1] for h in hits]
                total = aggregate(all_chunks, cfg)
                ref = {j: {k: a[:len(outputs[j]["n_persons"])] for k, a in reference[j].items()} for j in outputs}
                eq = compare.compare_runs(ref, outputs, box_tol)
                eq["gate_failures"] = compare.gate_failures(eq, precision)
                np.savez_compressed(RAW / f"stage9_{name}_b{b}.npz",
                                    **{f"{j}__{k}": a for j, arrs in outputs.items() for k, a in arrs.items()})
                sizes = tuple(s for s in ENGINE_SIZES if s <= b) if name != "pytorch" else ()
                rows.append({
                    "backend": name, "micro_batch": b,
                    **{k: total[k] for k in ("frames", "crops", "wall_s", "fps", "crops_per_s", "latency_ms",
                                             "pad_fraction", "pose_gpu_ms_per_frame", "pose_gpu_busy_share",
                                             "stage_busy_share", "stage_busy_s", "gpu_util_mean_pct",
                                             "sm_clock_mean_mhz", "batch_crops_hist",
                                             "contaminated_attempts_discarded")},
                    "crops_per_frame": total["crops"] / total["frames"],
                    "engines_loaded": list(sizes), "engine_vram_mb": engine_vram_mb(name, sizes),
                    "per_video_fps": {j: pv["fps"] for j, pv in per_video.items()},
                    "equivalence_vs_sync_pytorch": eq,
                    "eligible": not eq["gate_failures"],
                })
                kp = eq["keypoint_err_crop_px"]
                print(f"[stage9] {name:8s} b{b:<2d}: {total['fps']:5.1f} fps, {total['crops_per_s']:6.1f} crops/s, "
                      f"latency p50/p95 {total['latency_ms']['p50']:.0f}/{total['latency_ms']['p95']:.0f} ms, "
                      f"pad {total['pad_fraction']:.1%}; vs PyTorch median {kp['median']:.3f} px -- "
                      f"{'eligible' if not eq['gate_failures'] else 'NOT eligible: ' + '; '.join(eq['gate_failures'])}",
                      flush=True)

        # Phase 1b: the flush timeout at the fastest eligible configuration. The default timeout
        # is measured again here, interleaved with the others per chunk: phase 1's number for
        # it is the best of five noisy configurations, and would favour the default.
        eligible_rows = [r for r in rows if r["eligible"]]
        if eligible_rows and args.extra_timeouts_ms:
            best = max(eligible_rows, key=lambda r: r["fps"])
            ts = sorted({base.batch_timeout_ms, *args.extra_timeouts_ms})
            hits = {t: [] for t in ts}
            for u in units:
                for t in ts:
                    hits[t].append(measure("p1b", best["backend"], best["micro_batch"], stages.MAX_PERSONS, u, t))
            for t in ts:
                timeout_rows.append({"backend": best["backend"], "micro_batch": best["micro_batch"],
                                     "batch_timeout_ms": t, **summarize_variant(
                                         hits[t], replace(base, micro_batch=best["micro_batch"], batch_timeout_ms=t))})

        # Phase 2: the fastest eligible micro-batch of each backend at person caps 1-4,
        # interleaved per chunk (cap 4 measured again, for the same reason).
        for name in args.backends:
            eligible = [r for r in rows if r["backend"] == name and r["eligible"]]
            if not eligible:
                continue
            b = max(eligible, key=lambda r: r["fps"])["micro_batch"]
            caps = (*CAPS, stages.MAX_PERSONS)
            hits = {k: [] for k in caps}
            for u in units:
                for k in caps:
                    hits[k].append(measure("p2", name, b, k, u))
            curve = [{"max_persons": k, "latency_p95_ms": None,
                      **summarize_variant(hits[k], replace(base, micro_batch=b, max_persons=k))} for k in caps]
            for p in curve:
                p["latency_p95_ms"] = p["latency_ms"]["p95"]
            curves[name] = {"micro_batch": b, "points": sorted(curve, key=lambda p: p["crops_per_frame"])}
    finally:
        pools.close()

    # Phase 3: project onto the workload and recommend.
    targets = {"all": profile["all"]["mean_crops"],
               **{f"mode={m}": s["mean_crops"] for m, s in profile["by_mode"].items() if m not in ("None", "null")},
               **{f"per_video_{q}": v for q, v in profile["per_video_mean_crops_quantiles"].items()
                  if q in ("p25", "p50", "p75")}}
    projections = {name: {t: interpolate_fps(c["points"], x) for t, x in targets.items()}
                   for name, c in curves.items()}
    candidates = []
    for name, c in curves.items():
        row = next(r for r in rows if r["backend"] == name and r["micro_batch"] == c["micro_batch"])
        candidates.append({"backend": name, "micro_batch": c["micro_batch"],
                           "projected_fps_workload": projections[name]["all"]["fps"],
                           "benchmark_fps": row["fps"], "latency_p95_ms": row["latency_ms"]["p95"],
                           "engine_vram_mb": row["engine_vram_mb"]})
    scored = [c for c in candidates if c["projected_fps_workload"] is not None]
    recommended = None
    if scored:
        top = max(c["projected_fps_workload"] for c in scored)
        near = [c for c in scored if c["projected_fps_workload"] >= top * (1 - PROJECTION_TIE)]
        recommended = min(near, key=lambda c: (c["engine_vram_mb"] or float("inf"), c["latency_p95_ms"]))

    report = {
        "stage": 9,
        "config_base": asdict(base),
        "sweep": rows,
        "timeout_sweep": timeout_rows,
        "crops_per_frame_curves": curves,
        "workload_targets_crops_per_frame": targets,
        "projections": projections,
        "candidates": candidates,
        "recommended": recommended,
        "projection_tie": PROJECTION_TIE,
        "workload": [{"job_id": v["job_id"], "fingerprint": v["fingerprint"]} for v in videos],
        "checkpoint_tag": tag,
        "env": env_fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[stage9] recommended: {recommended}")
    print(f"[stage9] wrote {args.output}")


if __name__ == "__main__":
    main()
