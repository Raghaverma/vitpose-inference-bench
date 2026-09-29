#!/usr/bin/env python3
"""Stages 11-14: measure the pipeline upgrades, one change at a time, on Stage 9's pipeline.

    python -m pipeline.upgrades codec          # Stage 11: HF vs cv2 vs GPU crop warp + pose decode
    python -m pipeline.upgrades detector       # Stage 12: YOLOv8s fp32 vs fp16 vs TensorRT FP16
    python -m pipeline.upgrades pose           # Stage 13: TensorRT FP16 vs FP16/INT8 hybrid (vs INT8)
    python -m pipeline.upgrades resolutions    # Stage 14: the 1080p / portrait / 4K job videos
    python -m pipeline.upgrades projection     # Stage 14: people-per-frame curve -> the whole fleet

Every phase runs Stage 8's asynchronous pipeline (YOLO on 4 frames per call) with the same
chunked, contention-guarded method as Stages 7-9 (pipeline/runner.py): 1000-frame units, each
started only on a quiet box and re-run if anything overlapped it, all variants of a phase
interleaved per unit so the L4's clock drift lands on each alike, every clean unit checkpointed
(--resume <tag>).

The phases build on each other. Each one changes one thing against the previous phase's choice,
and keeps that choice as its baseline variant in the same interleaved run:

- codec: Stage 9's configuration (TensorRT FP16, micro-batch 4, YOLOv8s fp32) with the three
  codecs of pipeline/fast_codec.py. Gate: Stage 9's own (FP16 gates against Stage 7's synchronous
  PyTorch run, boxes within 0.5 px).
- detector: --codec's winner with the three detectors. Gate: pipeline/compare.py's detector gates
  against the fp32 detector's run (people matched by IoU), plus evaluation/coco_detected.py on
  human labels.
- pose: --codec + --detector with TensorRT FP16 and the hybrid (crops under 64 px -> FP16, the
  rest -> INT8, the threshold set by Stage 7 before this was built) at micro-batches 4-16, and
  all-INT8 as the ceiling. Gate: INT8 gates against the FP16 variant (people matched by IoU), and
  evaluation/coco_pose.py on human labels.
- resolutions: the 27 job videos that aren't 1280x720, Stage 9's configuration against the new
  one(s), plus a PyTorch FP16 reference pass (not timed) to gate their poses on this new footage.
- projection: Stage 9's configuration and the new one(s) at person caps 1-4 on the benchmark
  footage, read off at every 720p job video's own people per frame, plus the resolutions phase's
  measured throughput for the rest: frames/s over the whole fleet.

Outputs: results/pipeline/stage1{1,2,3,4}_*.json; per-frame outputs in results/raw/pipeline/.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baseline import REPO_ROOT
from calibration.real_corpus import content_fingerprint
from pipeline import compare, gpu_guard, runner, stages
from pipeline.async_pipeline import (BOX_TOL_BATCHED_DET_PX, AsyncConfig, AsyncGpuState, CpuPools, aggregate,
                                     frame_shape)
from pipeline.production_config import interpolate_fps, load_run_outputs
from pipeline.sync_pipeline import FrameLog, env_fingerprint, merge_logs

RESULTS = REPO_ROOT / "results" / "pipeline"
RAW = REPO_ROOT / "results" / "raw" / "pipeline"
STAGE9 = dict(backend="trt-fp16", mb=4, codec="hf", detector="pt32")     # Stage 9's recommendation
CPU_WORKERS = {"hf": (6, 2), "cv2": (6, 2), "gpu": (0, 0)}              # (preprocess, postprocess) processes


@dataclass
class Variant:
    label: str
    backend: str
    mb: int
    cfg: AsyncConfig

    def describe(self) -> dict:
        return {"label": self.label, "backend": self.backend, "micro_batch": self.mb,
                "codec": self.cfg.codec, "detector": self.cfg.detector, "max_persons": self.cfg.max_persons,
                **({"hybrid_threshold_px": self.cfg.hybrid_threshold_px} if self.backend == "trt-hybrid" else {})}


def variant(label: str, backend: str, mb: int, codec: str, detector: str, **kw) -> Variant:
    pre, post = CPU_WORKERS[codec]
    cfg = AsyncConfig(micro_batch=mb, det_batch=4, codec=codec, detector=detector,
                      preprocess_workers=pre, postprocess_workers=post, **kw)
    return Variant(label, backend, mb, cfg)


# ---- footage ------------------------------------------------------------------------------------

def other_resolution_videos(jobs_dir: Path) -> list[dict]:
    """Every distinct job video that isn't 1280x720 (Stage 9's workload profile lists them),
    checked against its content fingerprint."""
    rows = json.loads((RAW / "workload_profile_videos.json").read_text())
    out = []
    for r in rows:
        if r["resolution"] in (None, "1280x720"):
            continue
        path = jobs_dir / r["job_id"] / "source.mp4"
        if content_fingerprint(path) != r["fingerprint"]:
            raise SystemExit(f"[upgrades] REFUSING: {path} changed since the workload profile")
        out.append({"job_id": r["job_id"], "path": path, "fingerprint": r["fingerprint"],
                    "resolution": r["resolution"], "mode": r["mode"]})
    return out


# ---- measurement --------------------------------------------------------------------------------

def measure(phase: str, variants: list[Variant], videos: list[dict], args, tag: str,
            store: runner.Store) -> dict[str, dict]:
    """Run every variant over every chunk of `videos`, interleaved per chunk, grouped by frame
    shape (a CPU pool's shared ring and the GPU frame ring are per shape). Returns
    {label: {"outputs": {job: arrays}, "chunks": {job: [stats]}}}."""
    names = sorted({v.backend for v in variants})
    mbs = sorted({v.mb for v in variants})
    detectors = tuple(sorted({v.cfg.detector for v in variants}))
    by_shape = collections.defaultdict(list)
    for v in videos:
        by_shape[frame_shape(v)].append(v)
    first = videos[0]
    cap = stages.open_video(first["path"])
    frame0 = cap.read()[1]
    cap.release()
    gpu = AsyncGpuState(names, mbs, frame0, detectors, gpu_codec=any(v.cfg.codec == "gpu" for v in variants))
    done: dict[tuple, tuple] = {}
    for shape, group in by_shape.items():
        units = runner.chunk_units(group, args.chunk_frames, args.max_frames)
        codecs = sorted({v.cfg.codec for v in variants} - {"gpu"})
        pools = {c: CpuPools(next(v.cfg for v in variants if v.cfg.codec == c), shape) for c in codecs}
        try:
            own = {os.getpid()} | {p.pid for pl in pools.values() for p in pl.procs}
            gpu.warm = lambda: [gpu.run((v.backend, v.mb), group[0], pools.get(v.cfg.codec), v.cfg, args.warmup_frames)
                                for v in variants]
            if gpu.loaded:
                gpu.warm()
            else:
                gpu.ensure_loaded()
            print(f"[{phase}] {shape}: {len(group)} video(s), {len(units)} chunk(s) x {len(variants)} variants; "
                  f"checkpoints {store.dir} (resume with --resume {tag})", flush=True)
            for u in units:
                for var in variants:
                    key = f"{phase}/{var.label}/{u['key']}"
                    if (hit := store.get(key)) is None:
                        vid, n, start = u["video"], u["frames"], u["start"]
                        hit = runner.guarded(
                            lambda c, var=var, vid=vid, n=n, start=start: gpu.run(
                                (var.backend, var.mb), vid, pools.get(var.cfg.codec), var.cfg, n, start, c),
                            own, label=key, resources=gpu, prepare=lambda vid=u["video"], s=u["start"]: runner.open_at(vid, s))
                        store.put(key, hit)
                        st = hit[1]
                        print(f"[{phase}] {key}: {st['fps']:.1f} fps, {st['cpu_cores_busy_total']:.1f} cores, "
                              f"detect busy {st['stage_busy_share'].get('detect', 0):.2f}", flush=True)
                    done[(var.label, u["video"]["job_id"], u["start"])] = hit
        finally:
            for pl in pools.values():
                pl.close()
    gpu.ensure_released()
    out = {}
    for var in variants:
        outputs, chunks = {}, {}
        for v in videos:
            hits = [done[(var.label, v["job_id"], s)] for (lbl, j, s) in sorted(done) if lbl == var.label and j == v["job_id"]]
            outputs[v["job_id"]] = merge_logs([h[0] for h in hits]).arrays()
            chunks[v["job_id"]] = [h[1] for h in hits]
        out[var.label] = {"outputs": outputs, "chunks": chunks}
    return out


def summarize(var: Variant, res: dict) -> dict:
    all_chunks = [c for cs in res["chunks"].values() for c in cs]
    total = aggregate(all_chunks, var.cfg)
    keep = ("frames", "crops", "wall_s", "fps", "crops_per_s", "latency_ms", "pad_fraction", "pose_gpu_ms_per_frame",
            "pose_gpu_busy_share", "stage_busy_share", "gpu_util_mean_pct", "sm_clock_mean_mhz", "batch_crops_hist",
            "main_process_cpu_cores_busy", "cpu_cores_busy_total", "crops_per_route", "contaminated_attempts_discarded")
    return {**var.describe(), **{k: total[k] for k in keep}, "crops_per_frame": total["crops"] / total["frames"],
            "per_video_fps": {j: aggregate(cs, var.cfg)["fps"] for j, cs in res["chunks"].items()}}


def save_raw(name: str, outputs: dict) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.npz"
    np.savez_compressed(path, **{f"{j}__{k}": a for j, arrs in outputs.items() for k, a in arrs.items()})
    return str(path.relative_to(REPO_ROOT))


def trim(ref: dict, outputs: dict) -> dict:
    return {j: {k: a[:len(outputs[j]["n_persons"])] for k, a in ref[j].items()} for j in outputs}


def kp_line(rep: dict) -> str:
    kp = rep["keypoint_err_crop_px"] or {}
    return f"median {kp.get('median', float('nan')):.3f} p95 {kp.get('p95', float('nan')):.3f} >5px {kp.get('pct_over_5px', float('nan')):.2f}%"


# ---- phases ---------------------------------------------------------------------------------------

def phase_codec(args):
    s9 = dict(STAGE9)
    return [variant(f"codec={c}", s9["backend"], s9["mb"], c, s9["detector"]) for c in ("hf", "cv2", "gpu")]


def phase_detector(args):
    return [variant(f"detector={d}", "trt-fp16", 4, args.codec, d) for d in stages.DETECTORS]


def phase_pose(args):
    vs = [variant(f"trt-fp16/b{b}", "trt-fp16", b, args.codec, args.detector) for b in (4, 8)]
    vs += [variant(f"trt-hybrid/b{b}", "trt-hybrid", b, args.codec, args.detector) for b in (4, 8, 16)]
    vs += [variant(f"trt-int8/b{b}", "trt-int8", b, args.codec, args.detector) for b in (8, 16)]
    return vs


def new_configs(args) -> list[Variant]:
    """Stage 9's configuration and the upgraded one(s) chosen by the earlier phases."""
    vs = [variant("stage9", STAGE9["backend"], STAGE9["mb"], STAGE9["codec"], STAGE9["detector"]),
          variant("upgraded", args.backend, args.mb, args.codec, args.detector)]
    if args.also_hybrid:
        vs.append(variant("upgraded+hybrid", "trt-hybrid", args.hybrid_mb, args.codec, args.detector))
    return vs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("codec", "detector", "pose", "resolutions", "projection"))
    p.add_argument("--codec", default="gpu", choices=("hf", "cv2", "gpu"), help="The codec phase's choice.")
    p.add_argument("--detector", default="trt16", choices=stages.DETECTORS, help="The detector phase's choice.")
    p.add_argument("--backend", default="trt-fp16", help="The pose phase's choice for the upgraded configuration.")
    p.add_argument("--mb", type=int, default=4)
    p.add_argument("--also-hybrid", action="store_true", help="Also measure the hybrid as an opt-in configuration.")
    p.add_argument("--hybrid-mb", type=int, default=8)
    p.add_argument("--jobs-dir", type=Path, default=stages.default_jobs_dir())
    p.add_argument("--chunk-frames", type=int, default=1000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--warmup-frames", type=int, default=150)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    tag = args.resume or runner.new_tag(f"upgrades-{args.phase}")
    store = runner.Store(tag)
    bench = stages.workload_videos(args.jobs_dir)
    reference = load_run_outputs(RAW / "sync_pytorch.npz")
    report = {"phase": args.phase, "checkpoint_tag": tag, "env": env_fingerprint(),
              "method": "Stage 8 async pipeline, YOLO x4 per call; chunked + guarded units, variants interleaved per chunk"}

    if args.phase in ("codec", "detector", "pose"):
        variants = {"codec": phase_codec, "detector": phase_detector, "pose": phase_pose}[args.phase](args)
        res = measure(args.phase, variants, bench, args, tag, store)
        rows = []
        for var in variants:
            out = res[var.label]["outputs"]
            row = summarize(var, res[var.label])
            row["raw_outputs"] = save_raw(f"{args.phase}_{var.label.replace('/', '_').replace('=', '-')}", out)
            # Every variant against Stage 7's synchronous PyTorch FP16 run: frame-level with
            # Stage 9's box tolerance (exact detector), and people matched by IoU (any detector).
            ref = trim(reference, out)
            if var.cfg.detector == "pt32":
                eq = compare.compare_runs(ref, out, BOX_TOL_BATCHED_DET_PX)
                eq["gate_failures"] = compare.gate_failures(eq, "int8" if "int8" in var.backend or "hybrid" in var.backend else "fp16")
                row["vs_sync_pytorch"] = eq
            m = compare.compare_runs_matched(ref, out)
            m["gate_failures_int8"] = compare.gate_failures(m, "int8")
            row["vs_sync_pytorch_matched"] = m
            rows.append(row)
        base = variants[0]
        for var, row in zip(variants, rows):
            if var is base:
                continue
            a, b = res[base.label]["outputs"], res[var.label]["outputs"]
            if args.phase == "codec":
                rep = compare.compare_runs(a, b, BOX_TOL_BATCHED_DET_PX)
                rep["gate_failures"] = compare.gate_failures(rep, "fp16")
            elif args.phase == "detector":
                rep = compare.compare_runs_matched(a, b)
                rep["gate_failures"] = compare.detector_gate_failures(rep)
            else:
                rep = compare.compare_runs_matched(a, b)
                rep["gate_failures"] = compare.detector_gate_failures(rep) if "fp16" not in var.backend else \
                    compare.gate_failures(rep, "fp16")
            row[f"vs_{base.label}"] = rep
        for row in rows:
            gate = next((row[k]["gate_failures"] for k in row if k.startswith("vs_") and k not in
                         ("vs_sync_pytorch", "vs_sync_pytorch_matched")), [])
            print(f"[{args.phase}] {row['label']:18s} {row['fps']:6.1f} fps  {row['cpu_cores_busy_total']:.1f} cores  "
                  f"latency p50 {row['latency_ms']['p50']:.0f} ms  detect busy {row['stage_busy_share'].get('detect', 0):.2f}  "
                  f"vs PyTorch (matched) {kp_line(row['vs_sync_pytorch_matched'])}  "
                  f"{'gate OK' if not gate else 'GATE: ' + '; '.join(gate)}", flush=True)
        report["variants"] = rows
        name = {"codec": "stage11_codec", "detector": "stage12_detector", "pose": "stage13_pose"}[args.phase]

    elif args.phase == "resolutions":
        videos = other_resolution_videos(args.jobs_dir)
        variants = new_configs(args)
        # The reference: PyTorch FP16 with Stage 7's detector call (YOLO per frame), whose poses
        # match Stage 7's synchronous run to 0.004 px (Stage 8). Outputs only, so it runs once,
        # untimed, before the measured variants.
        ref_var = variant("reference", "pytorch", 4, "hf", "pt32")
        ref_var.cfg = replace(ref_var.cfg, det_batch=1)
        ref = measure("resolutions-ref", [ref_var], videos, args, tag, store)["reference"]["outputs"]
        save_raw("resolutions_reference_pytorch", ref)
        res = measure(args.phase, variants, videos, args, tag, store)
        rows = []
        for var in variants:
            out = res[var.label]["outputs"]
            row = summarize(var, res[var.label])
            row["raw_outputs"] = save_raw(f"resolutions_{var.label}", out)
            m = compare.compare_runs_matched(ref, out)
            m["gate_failures_int8"] = compare.gate_failures(m, "int8")
            m["detector_gate_failures"] = compare.detector_gate_failures(m)
            row["vs_pytorch_reference_matched"] = m
            by_res = collections.defaultdict(list)
            for v in videos:
                by_res[v["resolution"]] += res[var.label]["chunks"][v["job_id"]]
            row["by_resolution"] = {r: {k: agg[k] for k in ("frames", "crops", "wall_s", "fps", "cpu_cores_busy_total",
                                                              "stage_busy_share", "latency_ms")}
                                    for r, cs in by_res.items() for agg in [aggregate(cs, var.cfg)]}
            rows.append(row)
            print(f"[resolutions] {row['label']:16s} {row['fps']:6.1f} fps overall; " +
                  ", ".join(f"{r} {d['fps']:.1f}" for r, d in sorted(row["by_resolution"].items())) +
                  f"; vs PyTorch (matched) {kp_line(m)}", flush=True)
        report["videos"] = [{k: v[k] for k in ("job_id", "fingerprint", "resolution", "mode", "frames")} for v in videos]
        report["variants"] = rows
        name = "stage14_resolutions"

    else:   # projection
        variants = []
        for base in new_configs(args):
            for k in (1, 2, 3, 4):
                v = replace(base, label=f"{base.label}/cap{k}", cfg=replace(base.cfg, max_persons=k))
                variants.append(v)
        res = measure(args.phase, variants, bench, args, tag, store)
        curves = collections.defaultdict(list)
        for var in variants:
            row = summarize(var, res[var.label])
            curves[var.label.split("/")[0]].append({"max_persons": var.cfg.max_persons,
                                                    "crops_per_frame": row["crops_per_frame"], "fps": row["fps"],
                                                    "latency_p95_ms": row["latency_ms"]["p95"],
                                                    "cpu_cores_busy_total": row["cpu_cores_busy_total"]})
        # The fleet: each 720p job video read off its configuration's curve at its own people
        # per frame; every other video measured directly (resolutions phase).
        prof = json.loads((RAW / "workload_profile_videos.json").read_text())
        resol = json.loads((RESULTS / "stage14_resolutions.json").read_text())
        fleet = {}
        for label, pts in curves.items():
            t_720 = f_720 = 0.0
            extrap = 0
            for r in prof:
                if r["resolution"] != "1280x720" or not r["sampled"]:
                    continue
                crops = float(np.mean(np.minimum(r["detected"], stages.MAX_PERSONS)))
                # A video can pose at most MAX_PERSONS per frame, and a fifth of the 720p frames
                # are at exactly that cap, past the benchmark footage's 3.47: extrapolate up to it.
                top = max(p["crops_per_frame"] for p in pts)
                proj = interpolate_fps(pts, max(crops, min(p["crops_per_frame"] for p in pts)),
                                       max_extrapolation=max(stages.MAX_PERSONS - top, 0.0) + 1e-9)
                if proj["fps"] is None:
                    raise SystemExit(f"[projection] {r['job_id']}: {proj['note']}")
                extrap += proj.get("extrapolated", False)
                t_720 += r["frames_decoded"] / proj["fps"]
                f_720 += r["frames_decoded"]
            other = next(v for v in resol["variants"] if v["label"] == label)
            f_other, t_other = other["frames"], other["wall_s"]
            fleet[label] = {"fps_720p_projected": f_720 / t_720, "frames_720p": f_720,
                            "videos_720p_extrapolated": extrap,
                            "fps_other_resolutions_measured": f_other / t_other, "frames_other": f_other,
                            "fps_fleet": (f_720 + f_other) / (t_720 + t_other),
                            "hours_per_1m_frames": 1e6 / ((f_720 + f_other) / (t_720 + t_other)) / 3600}
            print(f"[projection] {label}: 720p {fleet[label]['fps_720p_projected']:.1f} fps (projected), other "
                  f"{fleet[label]['fps_other_resolutions_measured']:.1f} fps (measured), fleet {fleet[label]['fps_fleet']:.1f} fps",
                  flush=True)
        report["curves"] = curves
        report["fleet"] = fleet
        report["configs"] = [v.describe() for v in new_configs(args)]
        name = "stage14_projection"

    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
    print(f"[{args.phase}] wrote {out}")


if __name__ == "__main__":
    main()
