#!/usr/bin/env python3
"""Stage 7: a real, synchronous video pipeline, timed stage by stage.

    python -m pipeline.sync_pipeline                          # all backends, 3 held-out videos
    python -m pipeline.sync_pipeline --backends trt-fp16 --max-frames 120   # smoke run

Per frame, strictly one after another: decode (cv2 / FFmpeg) -> YOLOv8s person detection ->
HF crop + preprocess of up to 4 people -> H2D -> ViTPose++-L forward -> D2H -> HF pose decode.
Nothing overlaps: the GPU idles while the CPU decodes, crops and decodes poses, and the CPU
waits on the GPU in between. That is the baseline Stage 8's asynchronous pipeline is measured
against.

The question is Stage 5's: its deployment breakdown (profiling/profile_deployment_stages.py)
found HF pose decoding at 25.6% of a TensorRT call at batch 1, and said it would be the
bottleneck of a video pipeline. That breakdown had no video decode, no detector and no crop
warp. This one has all of them, on real footage (the Stage 6 held-out videos).

Two passes per (chunk, backend), same discipline as Stages 1 and 5:
  - instrumented: a device sync + timestamp at every stage boundary, giving the breakdown;
  - plain: no intermediate syncs, one wall clock per chunk -- the throughput number, and the
    independent total the instrumented stage sum has to match within STAGE_SUM_TOLERANCE.
Each video is cut into chunks of consecutive frames (pipeline/runner.py), and backends are
interleaved per chunk (trt-fp16, pytorch, trt-int8 on chunk 1, then chunk 2 ...) so the L4's
power-cap clock drift lands on all of them alike.

The box is shared with AutoClipping's production service and other sessions: every unit
waits for a quiet box and runs under pipeline.gpu_guard.GpuMonitor; a unit that overlapped
anything is re-run, and every clean unit is checkpointed (--resume continues a run).

Outputs:
  results/pipeline/sync_<backend>.json    per-backend breakdown, throughput, contention log
  results/pipeline/sync_summary.json      cross-backend: end-to-end speedups, equivalence gates
  results/raw/pipeline/sync_<backend>.npz per-frame boxes/keypoints/scores + per-frame stage
                                          times (gitignored: derived from private footage)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
import transformers
import ultralytics

from baseline import REPO_ROOT
from pipeline import compare, runner, stages
from stage1_benchmark import STAGE_SUM_TOLERANCE

STAGES = ("decode", "detect", "preprocess", "h2d", "pose", "d2h", "postprocess")
RESULTS_DIR = REPO_ROOT / "results" / "pipeline"
RAW_DIR = REPO_ROOT / "results" / "raw" / "pipeline"
STAGE5_POSTPROCESS_SHARE = {1: 0.256, 16: 0.367}   # profiling/results/deployment_stages_b{1,16}.json


class FrameLog:
    """Per-frame outputs + timings of one pass over one video, NaN-padded to MAX_PERSONS."""

    def __init__(self):
        self.rows = {k: [] for k in compare.FIELDS}
        self.stage_ms = {s: [] for s in STAGES}
        self.frame_ms, self.yolo_speed = [], {"preprocess": [], "inference": [], "postprocess": []}

    def add(self, n_detected, boxes, keypoints=None, scores=None):
        P = stages.MAX_PERSONS
        n = len(boxes)
        b = np.full((P, 4), np.nan, np.float32)
        kp = np.full((P, 17, 2), np.nan, np.float32)
        sc = np.full((P, 17), np.nan, np.float32)
        b[:n] = boxes
        if n:
            kp[:n], sc[:n] = keypoints, scores
        for k, v in zip(compare.FIELDS, (n_detected, n, b, kp, sc)):
            self.rows[k].append(v)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        out["n_detected"] = out["n_detected"].astype(np.int16)
        out["n_persons"] = out["n_persons"].astype(np.int16)
        return out


def run_pass(video: dict, detector, processor, backend, instrumented: bool,
             max_frames: int | None, start: int = 0, cap=None) -> tuple[FrameLog, dict]:
    """One pass over frames [start, start + max_frames) of one video (to the end if max_frames
    is None). Instrumented: sync + timestamp at every stage boundary. Plain: no syncs beyond the
    ones the data flow itself needs (download waits on its copy). `cap`: a capture already at
    `start` (runner.open_at); otherwise skipping to `start` happens here, before the clock."""
    if instrumented:
        def tick():
            torch.cuda.synchronize()
            return time.perf_counter()
    else:
        tick = time.perf_counter
    log = FrameLog()
    cap = cap if cap is not None else runner.open_at(video, start)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    frames = 0
    while max_frames is None or frames < max_frames:
        if runner.ABORT.is_set():
            cap.release()
            raise runner.Aborted()
        t0 = tick()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = tick()
        boxes, n_det, speed = stages.detect(detector, frame)
        t2 = tick()
        if len(boxes):
            pv = stages.preprocess(processor, frame, boxes)
            t3 = tick()
            handle = backend.upload(pv)
            t4 = tick()
            out = backend.forward(handle)
            t5 = tick()
            heatmaps = backend.download(out)
            t6 = tick()
            kp, sc = stages.postprocess(processor, heatmaps, boxes)
            t7 = tick()
            log.add(n_det, boxes, kp, sc)
        else:
            t3 = t4 = t5 = t6 = t7 = t2
            log.add(n_det, boxes)
        if instrumented:
            for s, a, b in zip(STAGES, (t0, t1, t2, t3, t4, t5, t6), (t1, t2, t3, t4, t5, t6, t7)):
                log.stage_ms[s].append((b - a) * 1000)
            log.frame_ms.append((tick() - t0) * 1000)
            for k in log.yolo_speed:
                log.yolo_speed[k].append(speed[k])
        frames += 1
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_start
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cap.release()
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    return log, {"frames": frames, "wall_s": wall_s, "cpu_s": cpu_s}


def summarize(instr: FrameLog, instr_stats: dict, plain_stats: dict) -> dict:
    """Breakdown of one (video or aggregate) instrumented pass + its plain pass."""
    frames = instr_stats["frames"]
    n = np.asarray(instr.rows["n_persons"])
    crops = int(n.sum())
    totals_ms = {s: float(np.sum(instr.stage_ms[s])) for s in STAGES}
    frame_total_ms = float(np.sum(instr.frame_ms))
    stage_sum_ms = sum(totals_ms.values())
    plain_ms = plain_stats["wall_s"] * 1000
    rel = abs(stage_sum_ms - plain_ms) / plain_ms
    per_crop = {s: totals_ms[s] / crops if crops else None for s in ("preprocess", "h2d", "pose", "d2h", "postprocess")}
    pose_path = ("h2d", "pose", "d2h", "postprocess")
    gpu_ms = float(np.sum(instr.yolo_speed["inference"])) + totals_ms["h2d"] + totals_ms["pose"] + totals_ms["d2h"]
    hist = np.bincount(np.asarray(instr.rows["n_detected"]), minlength=1)
    return {
        "frames": frames,
        "frames_with_persons": int((n > 0).sum()),
        "crops": crops,
        "crops_per_frame": crops / frames,
        "detected_persons_hist": {str(k): int(v) for k, v in enumerate(hist) if v},
        "frames_capped": int((np.asarray(instr.rows["n_detected"]) > stages.MAX_PERSONS).sum()),
        "plain": {
            "wall_s": plain_stats["wall_s"],
            "fps": frames / plain_stats["wall_s"],
            "crops_per_s": crops / plain_stats["wall_s"],
            "ms_per_frame": plain_ms / frames,
            "cpu_cores_busy": plain_stats["cpu_s"] / plain_stats["wall_s"],
        },
        "instrumented": {
            "wall_s": instr_stats["wall_s"],
            "stage_total_ms": totals_ms,
            "stage_ms_per_frame": {s: totals_ms[s] / frames for s in STAGES},
            "stage_ms_per_crop": per_crop,
            "stage_share": {s: totals_ms[s] / stage_sum_ms for s in STAGES},
            "stage_p50_ms_frames_with_persons": {
                s: float(np.median(np.asarray(instr.stage_ms[s])[n > 0])) if (n > 0).any() else None
                for s in STAGES},
            "unattributed_ms_per_frame": (frame_total_ms - stage_sum_ms) / frames,
            "yolo_speed_ms_per_frame": {k: float(np.mean(v)) for k, v in instr.yolo_speed.items()},
            "cpu_cores_busy": instr_stats["cpu_s"] / instr_stats["wall_s"],
            # The Stage 5 ratio, on the stages Stage 5 had: postprocess / (h2d+exec+d2h+post).
            "postprocess_share_of_pose_path": totals_ms["postprocess"] / sum(totals_ms[s] for s in pose_path),
            # GPU-resident work: YOLO's own inference time (ultralytics' timer, which syncs) +
            # the pose path's copies and forward. Everything else is CPU.
            "gpu_ms_per_frame": gpu_ms / frames,
            "gpu_busy_share": gpu_ms / stage_sum_ms,
        },
        "sync_additivity": {
            "stage_sum_ms": stage_sum_ms, "plain_wall_ms": plain_ms, "rel_diff": rel,
            "tolerance": STAGE_SUM_TOLERANCE, "ok": rel <= STAGE_SUM_TOLERANCE,
        },
    }


def merge_logs(logs: list[FrameLog]) -> FrameLog:
    out = FrameLog()
    for lg in logs:
        for k in out.rows:
            out.rows[k].extend(lg.rows[k])
        for s in STAGES:
            out.stage_ms[s].extend(lg.stage_ms[s])
        out.frame_ms.extend(lg.frame_ms)
        for k in out.yolo_speed:
            out.yolo_speed[k].extend(lg.yolo_speed[k])
    return out


def merge_stats(stats: list[dict]) -> dict:
    return {k: sum(s[k] for s in stats) for k in ("frames", "wall_s", "cpu_s")}


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def env_fingerprint() -> dict:
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "cpu_count": os.cpu_count(),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
        "loadavg_at_start": os.getloadavg(),
        "python": platform.python_version(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=list(stages.BACKENDS), choices=stages.BACKENDS)
    p.add_argument("--jobs-dir", type=Path, default=stages.default_jobs_dir(),
                   help="AutoClipping data/jobs dir holding the Stage 6 held-out source videos.")
    p.add_argument("--max-frames", type=int, default=None, help="Per video; for smoke runs only.")
    p.add_argument("--chunk-frames", type=int, default=1000,
                   help="Frames per measurement unit (see pipeline/runner.py).")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint tag of an interrupted run to continue.")
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    return p.parse_args()


class GpuState(runner.GpuResources):
    """The detector and every pose backend: everything Stage 7 holds on the GPU."""

    def __init__(self, names: list[str], first_frame: np.ndarray):
        self.names, self.first_frame = names, first_frame

    def load(self) -> None:
        self.detector = stages.load_detector()
        for _ in range(5):
            stages.detect(self.detector, self.first_frame)
        self.backends = {name: stages.load_backend(name) for name in self.names}
        for be in self.backends.values():           # every batch size a frame can need
            for n in range(1, stages.MAX_PERSONS + 1):
                be.download(be.forward(be.upload(np.zeros((n, *stages.POSE_INPUT), np.float16))))

    def release(self) -> None:
        del self.detector, self.backends


def main() -> None:
    args = parse_args()
    videos = stages.workload_videos(args.jobs_dir)
    print(f"[stage7] workload: {[v['job_id'] for v in videos]}", flush=True)
    processor = stages.load_processor()
    cap = stages.open_video(videos[0]["path"])
    gpu = GpuState(args.backends, cap.read()[1])
    cap.release()

    def warm() -> None:
        for name in args.backends:
            run_pass(videos[0], gpu.detector, processor, gpu.backends[name], True, args.warmup_frames)

    gpu.warm = warm
    print(f"[stage7] loading + warmup ({args.warmup_frames} frames per backend)", flush=True)
    gpu.ensure_loaded()
    env = env_fingerprint()

    tag = args.resume or runner.new_tag("stage7")
    store = runner.Store(tag)
    units = runner.chunk_units(videos, args.chunk_frames, args.max_frames)
    print(f"[stage7] {len(units)} chunks x {len(args.backends)} backends x 2 passes; checkpoints: {store.dir} "
          f"(resume with --resume {tag})", flush=True)
    done = {}
    for unit in units:
        for name in args.backends:
            for mode in ("plain", "instr"):
                key = f"{name}/{mode}/{unit['key']}"
                if (hit := store.get(key)) is None:
                    t = time.perf_counter()
                    hit = runner.guarded(lambda cap: run_pass(unit["video"], gpu.detector, processor, gpu.backends[name],
                                                              mode == "instr", unit["frames"], unit["start"], cap),
                                         label=key, resources=gpu,
                                         prepare=lambda: runner.open_at(unit["video"], unit["start"]))
                    store.put(key, hit)
                    st = hit[1]
                    print(f"[stage7] {key}: {st['frames'] / st['wall_s']:.1f} fps ({time.perf_counter() - t:.0f}s incl. "
                          f"waiting, SM {st['gpu']['sm_clock_mean_mhz']:.0f} MHz)", flush=True)
                done[key] = hit
    gpu.ensure_loaded()
    backends = gpu.backends

    runs = {name: {mode: {} for mode in ("instr", "plain")} for name in backends}
    unit_stats = {name: {mode: {} for mode in ("instr", "plain")} for name in backends}
    for name in backends:
        for mode in ("instr", "plain"):
            for v in videos:
                hits = [done[f"{name}/{mode}/{u['key']}"] for u in units if u["video"] is v]
                runs[name][mode][v["job_id"]] = (merge_logs([h[0] for h in hits]), merge_stats([h[1] for h in hits]))
                unit_stats[name][mode][v["job_id"]] = [h[1] for h in hits]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    outputs, reports = {}, {}
    for name, be in backends.items():
        instr, plain = runs[name]["instr"], runs[name]["plain"]
        outputs[name] = {j: plain[j][0].arrays() for j in plain}
        instr_outputs = {j: instr[j][0].arrays() for j in instr}
        per_video = {j: summarize(instr[j][0], instr[j][1], plain[j][1]) for j in instr}
        total = summarize(merge_logs([instr[j][0] for j in instr]),
                          merge_stats([instr[j][1] for j in instr]), merge_stats([plain[j][1] for j in plain]))
        raw_path = args.raw_dir / f"sync_{name}.npz"
        raw = {f"{j}__{k}": v for j, arrs in outputs[name].items() for k, v in arrs.items()}
        raw.update({f"{j}__stage_ms__{s}": np.asarray(instr[j][0].stage_ms[s], np.float32)
                    for j in instr for s in STAGES})
        np.savez_compressed(raw_path, **raw)
        reports[name] = {
            "stage": 7,
            "label": "synchronous video pipeline: decode -> detect -> preprocess -> h2d -> pose -> d2h -> postprocess, "
                     "one frame at a time, no overlap",
            "backend": be.describe(),
            "config": {"det_conf": stages.DET_CONF, "max_persons": stages.MAX_PERSONS,
                       "dataset_index": stages.DATASET_INDEX, "detector": "checkpoints/yolov8s.pt (ultralytics, fp32)",
                       "decode": "cv2.VideoCapture (FFmpeg, its default decode threads)",
                       "preprocess": "HF VitPoseImageProcessor (scipy affine warp), cast to fp16 on the CPU",
                       "postprocess": "HF post_process_pose_estimation (DARK)",
                       "max_frames": args.max_frames, "warmup_frames": args.warmup_frames},
            "workload": [{"job_id": v["job_id"], "fingerprint": v["fingerprint"]} for v in videos],
            "total": total,
            "per_video": per_video,
            "stage5_postprocess_share_for_reference": STAGE5_POSTPROCESS_SHARE,
            "determinism_plain_vs_instrumented_identical": compare.identical(outputs[name], instr_outputs),
            "outputs_sha256": stages.outputs_digest(raw),
            "raw_outputs": repo_relative(raw_path),
            "units": unit_stats[name],
            "checkpoint_tag": tag,
            "env": env,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_dir / f"sync_{name}.json").write_text(json.dumps(reports[name], indent=2) + "\n")
        t = total
        print(f"[stage7] {name}: {t['plain']['fps']:.1f} fps end to end, {t['crops_per_frame']:.2f} crops/frame; "
              f"additivity {t['sync_additivity']['rel_diff']:.1%} ({'OK' if t['sync_additivity']['ok'] else 'FAILED'})")
        for s in STAGES:
            print(f"[stage7]   {s:12s} {t['instrumented']['stage_ms_per_frame'][s]:7.2f} ms/frame "
                  f"({t['instrumented']['stage_share'][s]:.1%})")

    summary = {"stage": 7, "backends": {}, "equivalence": {},
               "created_utc": datetime.now(timezone.utc).isoformat()}
    for name, r in reports.items():
        summary["backends"][name] = {
            "fps": r["total"]["plain"]["fps"],
            "pose_ms_per_frame": r["total"]["instrumented"]["stage_ms_per_frame"]["pose"],
            "sync_additivity_ok": r["total"]["sync_additivity"]["ok"],
            "deterministic": r["determinism_plain_vs_instrumented_identical"],
        }
    if "pytorch" in reports:
        pt = reports["pytorch"]["total"]
        for name, r in reports.items():
            if name == "pytorch":
                continue
            s = summary["backends"][name]
            s["end_to_end_speedup_vs_pytorch"] = r["total"]["plain"]["fps"] / pt["plain"]["fps"]
            s["pose_stage_speedup_vs_pytorch"] = (pt["instrumented"]["stage_ms_per_frame"]["pose"]
                                                  / r["total"]["instrumented"]["stage_ms_per_frame"]["pose"])
            rep = compare.compare_runs(outputs["pytorch"], outputs[name])
            precision = name.split("-")[1]
            rep["gate_failures"] = compare.gate_failures(rep, precision)
            rep["gates"] = compare.GATES[precision]
            summary["equivalence"][f"{name}_vs_pytorch"] = rep
            kp = rep["keypoint_err_crop_px"] or {}
            print(f"[stage7] {name} vs pytorch: end to end {s['end_to_end_speedup_vs_pytorch']:.2f}x, pose stage "
                  f"{s['pose_stage_speedup_vs_pytorch']:.2f}x; keypoints median {kp.get('median', float('nan')):.4f} px "
                  f"p95 {kp.get('p95', float('nan')):.4f} px over {rep['crops_compared']} crops "
                  f"({rep['frames_boxes_differ']} frames with different boxes) -- "
                  f"{'gates OK' if not rep['gate_failures'] else 'GATES FAILED: ' + '; '.join(rep['gate_failures'])}")
    (args.output_dir / "sync_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[stage7] wrote {args.output_dir}/sync_*.json and {args.raw_dir}/sync_*.npz")


if __name__ == "__main__":
    main()
