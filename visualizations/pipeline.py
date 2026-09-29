#!/usr/bin/env python3
"""Stages 7-9: the video pipeline figures.

    python -m visualizations.pipeline

- docs/images/pipeline_sync_breakdown.png (Stage 7): where a synchronous frame's time goes,
  one panel per pose backend, ms per frame by stage. Emphasis form: the pose forward -- the
  stage Stages 2-6 optimized -- in its backend's color, every other stage in one gray, so the
  panels answer "how much of the frame is the part we made faster".
- docs/images/pipeline_async_speedup.png (Stage 8): frames/s of the synchronous and the
  asynchronous pipeline per backend, as a dumbbell (before -> after per item).
- docs/images/pipeline_async_busy.png (Stage 8): each asynchronous stage's busy share, to show
  which one the others wait for.
- docs/images/pipeline_batch_sweep.png + pipeline_batch_latency.png (Stage 9): throughput and
  latency against the pose micro-batch, two figures rather than one dual-axis chart.
- docs/images/workload_crops.png (Stage 9): people per frame on the real workload.

Text stays in ink colors; a colored mark beside it carries backend identity. Backend colors
are _style.py's fixed assignments (PyTorch blue, TensorRT FP16 yellow, TensorRT INT8 magenta).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import cpu_stages as stages
from visualizations._data import REPO_ROOT, load_stage7, load_stage8, load_stage9
from visualizations._style import (BASELINE, BLUE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, MAGENTA, YELLOW,
                                   apply_style, style_axes)

IMAGES = REPO_ROOT / "docs" / "images"
DEEMPHASIS = "#cfd3ce"          # the gray visualizations/stage_breakdown.py uses for context stages
BACKENDS = [("pytorch", "PyTorch FP16", BLUE), ("trt-fp16", "TensorRT FP16", YELLOW),
            ("trt-int8", "TensorRT INT8", MAGENTA)]
STAGE_ROWS = [  # (label, stages summed into the row)
    ("Video decode", ("decode",)),
    ("YOLOv8s detect", ("detect",)),
    ("Crop + preprocess (HF)", ("preprocess",)),
    ("H2D + D2H copies", ("h2d", "d2h")),
    ("ViTPose++-L forward", ("pose",)),
    ("Pose decode (HF)", ("postprocess",)),
]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_sync_breakdown(output_path: Path = IMAGES / "pipeline_sync_breakdown.png") -> Path:
    apply_style()
    data = load_stage7()
    present = [b for b in BACKENDS if b[0] in data["reports"]]
    fig, axes = plt.subplots(1, len(present), figsize=(4.1 * len(present), 3.9), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    xmax = 0.0
    for ax, (key, label, color) in zip(axes, present):
        t = data["reports"][key]["total"]
        per_frame = t["instrumented"]["stage_ms_per_frame"]
        values = [sum(per_frame[s] for s in stages) for _, stages in STAGE_ROWS]
        colors = [color if stages == ("pose",) else DEEMPHASIS for _, stages in STAGE_ROWS]
        y = np.arange(len(STAGE_ROWS))[::-1]
        ax.barh(y, values, height=0.56, color=colors, zorder=3)
        for yi, v in zip(y, values):
            ax.text(v + 0.4, yi, f"{v:.1f}", va="center", ha="left", fontsize=8.5, color=INK_PRIMARY)
        xmax = max(xmax, max(values))
        style_axes(ax, ygrid=False)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in STAGE_ROWS])
        total = sum(values)
        pose_share = per_frame["pose"] / sum(per_frame.values())
        eq = data["summary"]["equivalence"].get(f"{key}_vs_pytorch")
        verdict = "\nfails its pose gates vs PyTorch" if eq and eq["gate_failures"] else ""
        ax.set_title(f"{label}\n{t['plain']['fps']:.1f} frames/s, pose {pose_share:.0%} of the frame{verdict}",
                     fontsize=10.5, loc="left")
        ax.set_xlabel("ms per frame")
    for ax in axes:
        ax.set_xlim(0, xmax * 1.22)
    s = data["reports"][present[0][0]]["total"]
    fig.suptitle("Stage 7: where a synchronous frame's time goes -- 1280x720 nets footage, NVIDIA L4",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK_PRIMARY, y=1.04)
    fig.text(0.01, -0.03,
             f"{s['frames']:,} frames of 3 held-out videos, {s['crops_per_frame']:.2f} people posed per frame (cap 4). "
             f"One frame at a time, stages back to back; instrumented pass, stage sums within "
             f"{max(r['total']['sync_additivity']['rel_diff'] for r in data['reports'].values()):.1%} of the plain pass.",
             ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, output_path)


def _variants(data: dict) -> list[str]:
    """Async variants in det_batch order, e.g. ["det_batch=1", "det_batch=4"]."""
    return sorted(data["variants"], key=lambda k: data["variants"][k]["det_batch"])


def generate_async_speedup(output_path: Path = IMAGES / "pipeline_async_speedup.png") -> Path:
    """Per backend: synchronous (hollow gray) -> async, per-frame YOLO (hollow, backend color) ->
    async, batched YOLO (filled, backend color) frames/s, on one line."""
    apply_style()
    data = load_stage8()
    variants = _variants(data)
    present = [b for b in BACKENDS if b[0] in data["backends"]]
    fig, ax = plt.subplots(figsize=(8.2, 1.2 + 0.85 * len(present)))
    y = np.arange(len(present))[::-1]
    xmax = 0.0
    for yi, (key, label, color) in zip(y, present):
        b = data["backends"][key]
        xs = [b["sync_fps"]] + [b["async"][v]["fps"] for v in variants]
        ax.plot([min(xs), max(xs)], [yi, yi], color=BASELINE, linewidth=2, zorder=2)
        ax.scatter([b["sync_fps"]], [yi], s=70, facecolor="white", edgecolor=INK_MUTED, linewidth=1.6, zorder=3)
        for v in variants[:-1]:
            ax.scatter([b["async"][v]["fps"]], [yi], s=70, facecolor="white", edgecolor=color, linewidth=2, zorder=3)
        last = b["async"][variants[-1]]
        ax.scatter([last["fps"]], [yi], s=85, color=color, edgecolor="white", linewidth=2, zorder=4)
        ax.text(b["sync_fps"] - 1.5, yi, f"{b['sync_fps']:.1f}", ha="right", va="center", fontsize=9,
                color=INK_SECONDARY)
        ax.text(max(xs) + 2.5, yi, f"{last['fps']:.1f}  ({last['speedup_vs_sync']:.2f}x)", ha="left",
                va="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold")
        xmax = max(xmax, *xs)
    style_axes(ax, ygrid=False)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([p[1] for p in present])
    ax.set_xlim(0, xmax * 1.3)
    ax.set_ylim(-0.7, len(present) - 0.3)
    ax.set_xlabel("Frames per second, end to end")
    ax.set_title("Stage 8: synchronous vs asynchronous pipeline, same frames", loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
                          markeredgecolor=INK_MUTED, markeredgewidth=1.6, label="Stage 7 synchronous"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
                          markeredgecolor=INK_SECONDARY, markeredgewidth=2, label="async, YOLO per frame"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=INK_SECONDARY,
                          label=f"async, YOLO {data['variants'][variants[-1]]['det_batch']} frames per call")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    cfg = data["config"]
    fig.text(0.01, -0.12,
             f"Paired per 1000-frame chunk in one process. Async: pose micro-batch {cfg['micro_batch']} across frames, "
             f"{cfg['preprocess_workers']} preprocess + {cfg['postprocess_workers']} pose-decode worker processes, "
             f"separate CUDA streams for detection and pose.", ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, output_path)


BUSY_ROWS = [("decode", "Video decode (1 thread)"), ("detect", "YOLO detect (1 thread)"),
             ("preprocess", "Crop + preprocess (pool)"), ("batcher", "Pose batcher (1 thread)"),
             ("completer", "Pose completer (1 thread)"), ("postprocess", "Pose decode (pool)")]


SATURATED = 0.9


def generate_async_busy(output_path: Path = IMAGES / "pipeline_async_busy.png") -> Path:
    """Each async stage's busy share (busy time / wall / workers), the pose stream's (CUDA
    events) and the GPU's (NVML), one panel per backend, for the batched-YOLO variant; every
    resource at or above SATURATED in the backend color."""
    apply_style()
    data = load_stage8()
    variant = _variants(data)[-1]
    present = [b for b in BACKENDS if b[0] in data["backends"]]
    fig, axes = plt.subplots(1, len(present), figsize=(4.3 * len(present), 3.6), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (key, label, color) in zip(axes, present):
        a = data["backends"][key]["async"][variant]
        rows = [(lbl, a["stage_busy_share"].get(k, 0.0)) for k, lbl in BUSY_ROWS]
        rows.append(("A pose batch in flight (GPU)", a["pose_gpu_busy_share"]))
        rows.append(("GPU, any kernel (NVML)", a["gpu_util_mean_pct"] / 100))
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [v for _, v in rows], height=0.56,
                color=[color if v >= SATURATED else DEEMPHASIS for _, v in rows], zorder=3)
        for yi, (_, v) in zip(y, rows):
            ax.text(v + 0.02, yi, f"{v:.0%}", va="center", ha="left", fontsize=8.5, color=INK_PRIMARY)
        ax.axvline(1.0, color=BASELINE, linewidth=1, zorder=2)
        style_axes(ax, ygrid=False)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Busy share of wall time")
        ax.set_title(f"{label}: {a['fps']:.1f} frames/s", fontsize=10.5, loc="left")
    fig.suptitle(f"Stage 8: what the asynchronous pipeline waits for "
                 f"(YOLO on {data['variants'][variant]['det_batch']} frames per call)", x=0.01,
                 ha="left", fontsize=12.5, fontweight="bold", color=INK_PRIMARY, y=1.04)
    fig.text(0.01, -0.04, f"Colored: {SATURATED:.0%} or more. \"A pose batch in flight\" is event-timed on the pose stream, so "
             "it includes time YOLO's kernels share the GPU; a thread's busy time includes waiting for its own GPU "
             "work (YOLO's syncs).", ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, output_path)


def _sweep_series(rows: list[dict], backend: str, field) -> tuple[list[int], list[float], list[bool]]:
    pts = sorted((r["micro_batch"], field(r), r["eligible"]) for r in rows if r["backend"] == backend)
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def _sweep_figure(field, ylabel: str, title: str, output_path: Path, note: str) -> Path:
    apply_style()
    sweep = load_stage9()["sweep"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, label, color in BACKENDS:
        xs, ys, ok = _sweep_series(sweep["sweep"], key, field)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=2, zorder=3, label=label + ("" if all(ok) else " (fails its gates)"),
                linestyle="-" if all(ok) else (0, (1, 2)))
        ax.scatter(xs, ys, s=48, color=[color if o else "white" for o in ok], edgecolor=color if not all(ok) else "white",
                   linewidth=1.8, zorder=4)
    rec = sweep.get("recommended")
    if rec:
        r = next(r for r in sweep["sweep"] if r["backend"] == rec["backend"] and r["micro_batch"] == rec["micro_batch"])
        ax.annotate("recommended", (r["micro_batch"], field(r)), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5, color=INK_PRIMARY,
                    arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    style_axes(ax)
    sizes = sorted({r["micro_batch"] for r in sweep["sweep"]})
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(b) for b in sizes])
    ax.set_xlabel("Pose micro-batch (crops per forward, packed across frames)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=9)
    fig.text(0.01, -0.03, note, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, output_path)


def generate_batch_sweep() -> list[Path]:
    return [
        _sweep_figure(lambda r: r["fps"], "Frames per second, end to end",
                      "Stage 9: asynchronous pipeline throughput vs pose micro-batch",
                      IMAGES / "pipeline_batch_sweep.png",
                      "Real crops only (padding excluded). A dotted line with hollow points failed its pose gates "
                      "against Stage 7's PyTorch reference and is not eligible."),
        _sweep_figure(lambda r: r["latency_ms"]["p95"], "p95 latency per frame (ms, decode -> poses)",
                      "Stage 9: per-frame latency vs pose micro-batch",
                      IMAGES / "pipeline_batch_latency.png",
                      "Latency from a frame being decoded to its poses being decoded, in the throughput-bound "
                      "offline setting (the decoder runs as fast as the ring allows)."),
    ]


def generate_workload_crops(output_path: Path = IMAGES / "workload_crops.png") -> Path:
    """People posed per frame (after the cap of 4) across the real workload: one series, one hue."""
    apply_style()
    prof = load_stage9()["profile"]
    hist = prof["all"]["crops_hist"]
    ks = sorted(int(k) for k in hist)
    total = sum(hist.values())
    shares = [hist[str(k)] / total for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(ks, shares, width=0.55, color=BLUE, zorder=3)
    for bar, v in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=9,
                color=INK_PRIMARY)
    style_axes(ax)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}" + (" (4 or more detected)" if k == stages.MAX_PERSONS else "") for k in ks])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("People posed in the frame")
    ax.set_ylabel("Share of frames")
    ax.set_ylim(0, max(shares) * 1.2)
    ax.set_title(f"Stage 9: the real workload -- {prof['videos']} job videos, "
                 f"{prof['all']['mean_crops']:.2f} crops per frame", loc="left", fontsize=11.5)
    fig.text(0.01, -0.03,
             f"Every {prof['method']['every_nth_frame']}th frame of every AutoClipping job video on the box "
             f"({prof['all']['frames']:,} frames), YOLOv8s conf 0.35, largest 4 kept; "
             f"{prof['all']['frames_capped_pct']:.0f}% of frames had more than 4 people.",
             ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, output_path)


def generate() -> list[Path]:
    return [generate_sync_breakdown(), generate_async_speedup(), generate_async_busy(),
            *generate_batch_sweep(), generate_workload_crops()]


if __name__ == "__main__":
    for path in generate():
        print(f"[pipeline] wrote {path}")
