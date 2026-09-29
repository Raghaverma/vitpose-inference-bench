#!/usr/bin/env python3
"""Stage 9, step 1: the crop-count distribution of the real workload.

    python -m pipeline.workload_profile [--every 5]

How many people per frame the pose stage sees decides how much a batching policy pads: a
frame-at-a-time pipeline (Stage 7) runs each frame's n crops on a static engine of batch >= n,
so a 3-person frame pays for 4. This measures n on every AutoClipping job video on this box
-- the production footage this repo's work is for -- with the pipeline's own detector and
selection rule (pipeline/stages.detect: YOLOv8s, conf 0.35, top 4 by area), on every
`--every`-th frame.

Byte-identical re-uploads (the same content fingerprint under another job id) are counted once.
Committed output is aggregate only (results/pipeline/workload_profile.json: histograms by job
mode and resolution); per-video rows, keyed by job id, go to results/raw/pipeline/ (gitignored).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baseline import REPO_ROOT
from calibration.real_corpus import content_fingerprint
from pipeline import stages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", type=Path, default=stages.default_jobs_dir())
    p.add_argument("--every", type=int, default=5, help="Detect on every N-th decoded frame.")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "pipeline" / "workload_profile.json")
    p.add_argument("--raw-output", type=Path, default=REPO_ROOT / "results" / "raw" / "pipeline" / "workload_profile_videos.json")
    return p.parse_args()


def hist(counts) -> dict[str, int]:
    c = collections.Counter(int(x) for x in counts)
    return {str(k): c[k] for k in sorted(c)}


def summarize(ns: np.ndarray) -> dict:
    capped = np.minimum(ns, stages.MAX_PERSONS)
    # Frame-at-a-time padding on the Stage 4/6 static engines (1, 2, 4): slots paid per frame.
    engine = np.select([capped == 0, capped == 1, capped == 2], [0, 1, 2], default=4)
    return {
        "frames": int(len(ns)),
        "detected_hist": hist(ns),
        "crops_hist": hist(capped),
        "mean_detected": float(ns.mean()),
        "mean_crops": float(capped.mean()),
        "frames_capped_pct": float((ns > stages.MAX_PERSONS).mean() * 100),
        "frames_empty_pct": float((ns == 0).mean() * 100),
        "per_frame_engine_pad_fraction": float(1 - capped.sum() / engine.sum()) if engine.sum() else 0.0,
    }


def main() -> None:
    args = parse_args()
    detector = stages.load_detector()
    videos, seen = [], {}
    for src in sorted(args.jobs_dir.glob("*/source.mp4")):
        fp = content_fingerprint(src)
        if fp in seen:
            continue
        seen[fp] = src.parent.name
        meta_path = src.parent / "job_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        videos.append({"job_id": src.parent.name, "path": src, "fingerprint": fp, "mode": meta.get("mode"),
                       "analysis_type": (meta.get("meta") or {}).get("analysis_type")})
    print(f"[workload] {len(videos)} distinct videos (of {len(list(args.jobs_dir.glob('*/source.mp4')))} jobs)", flush=True)

    rows = []
    t_start = time.perf_counter()
    for i, v in enumerate(videos):
        # No contention guard: this counts detections, which don't depend on what else runs.
        cap = stages.open_video(v["path"])
        ns, heights, frame_idx, shape = [], [], 0, None
        while True:
            if frame_idx % args.every:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            shape = frame.shape
            boxes, n_det, _ = stages.detect(detector, frame)
            ns.append(n_det)
            heights.extend(boxes[:, 3].tolist())
            frame_idx += 1
        cap.release()
        ns = np.asarray(ns)
        rows.append({"job_id": v["job_id"], "fingerprint": v["fingerprint"], "mode": v["mode"],
                     "analysis_type": v["analysis_type"], "frames_decoded": frame_idx,
                     "resolution": f"{shape[1]}x{shape[0]}" if shape else None,
                     "sampled": int(len(ns)), "detected": ns.tolist(),
                     "box_height_px_quartiles": np.percentile(heights, [25, 50, 75]).tolist() if heights else None})
        print(f"[workload] {i + 1}/{len(videos)} {v['job_id']} {v['mode']:8s} {rows[-1]['resolution']} "
              f"{frame_idx} frames, mean {ns.mean() if len(ns) else 0:.2f} persons "
              f"({time.perf_counter() - t_start:.0f}s)", flush=True)

    all_ns = np.concatenate([np.asarray(r["detected"]) for r in rows])
    by = lambda key: {k: summarize(np.concatenate([np.asarray(r["detected"]) for r in rows if r[key] == k]))
                      for k in sorted({r[key] for r in rows}, key=str)}
    per_video_mean = np.array([np.mean(np.minimum(r["detected"], stages.MAX_PERSONS)) for r in rows if r["sampled"]])
    report = {
        "stage": 9,
        "purpose": "people per frame on the real workload (every AutoClipping job video on this box), for choosing "
                   "the pose batching policy -- see pipeline/workload_profile.py",
        "method": {"detector": "pipeline.stages.detect (YOLOv8s, conf 0.35, person class)",
                   "max_persons": stages.MAX_PERSONS, "every_nth_frame": args.every,
                   "dedup": "by calibration.real_corpus.content_fingerprint"},
        "videos": len(rows),
        "videos_by_mode": dict(collections.Counter(str(r["mode"]) for r in rows)),
        "videos_by_resolution": dict(collections.Counter(str(r["resolution"]) for r in rows)),
        "frames_decoded": int(sum(r["frames_decoded"] for r in rows)),
        "all": summarize(all_ns),
        "by_mode": by("mode"),
        "by_resolution": by("resolution"),
        "per_video_mean_crops_quantiles": dict(zip(("min", "p25", "p50", "p75", "max"),
                                                   np.percentile(per_video_mean, [0, 25, 50, 75, 100]).tolist())),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(rows) + "\n")
    a = report["all"]
    print(f"[workload] {a['frames']} sampled frames: mean {a['mean_crops']:.2f} crops/frame after the cap, "
          f"{a['frames_capped_pct']:.0f}% of frames capped, per-frame engine padding {a['per_frame_engine_pad_fraction']:.1%}")


if __name__ == "__main__":
    main()
