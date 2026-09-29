#!/usr/bin/env python3
"""Stages 10-14: figures for human-label accuracy and the pipeline upgrades.

    python -m visualizations.upgrades

- docs/images/coco_int8_by_height.png (Stage 10): INT8 against PyTorch FP16 by crop height,
  as two small multiples on one shared x: agreement (joints moved > 5 px from PyTorch's) and
  accuracy against COCO's human labels (mean OKS change, with its bootstrap 95% interval). Two
  measures of different scale get two panels, never two y-axes.
- docs/images/upgrades_steps.png (Stages 11-13): throughput and CPU of each step from Stage 9's
  configuration, as two panels.
- docs/images/upgrades_resolutions.png (Stage 14): frames/s per resolution, Stage 9's
  configuration against the upgraded one (dumbbell).

Single-series panels carry no legend (the title names the series); INT8 keeps its fixed magenta,
TensorRT FP16 its yellow (_style.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT
from visualizations._style import (BASELINE, BLUE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, MAGENTA, YELLOW,
                                   apply_style, style_axes)

IMAGES = REPO_ROOT / "docs" / "images"
RESULTS = REPO_ROOT / "results"
DEEMPHASIS = "#cfd3ce"


def _load(rel: str, cmd: str) -> dict:
    path = RESULTS / rel
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing -- run: {cmd}")
    return json.loads(path.read_text())


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_coco_int8(output_path: Path = IMAGES / "coco_int8_by_height.png") -> Path:
    apply_style()
    r = _load("evaluation/coco_pose.json", "python -m evaluation.coco_pose")
    sets = [s for s in r["sets"] if s != "original"] + ["original"]
    labels = [s.replace("crop_", "").replace("px", " px") if s != "original"
              else f"as is\n(median {r['sets']['original']['crop_height_px_quantiles']['p50']:.0f} px)" for s in sets]
    p = r["paired_vs_pytorch"]["trt-int8"]
    drift = [p[s]["keypoint_drift_vs_pytorch_crop_px"]["pct_over_5px"] for s in sets]
    oks = [p[s]["mean_oks_diff"] * 100 for s in sets]
    lo = [o - p[s]["ci95"][0] * 100 for o, s in zip(oks, sets)]
    hi = [p[s]["ci95"][1] * 100 - o for o, s in zip(oks, sets)]
    xs = list(range(len(sets)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    a1.bar(xs, drift, width=0.55, color=MAGENTA, zorder=3)
    for x, v in zip(xs, drift):
        a1.text(x, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK_PRIMARY)
    style_axes(a1)
    a1.set_ylabel("Joints > 5 px from PyTorch's (%)")
    a1.set_title("Agreement: INT8 moves small people's joints most", loc="left", fontsize=11.5)
    a1.set_ylim(0, max(drift) * 1.18)

    a2.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)
    a2.errorbar(xs, oks, yerr=[lo, hi], fmt="o", color=MAGENTA, markersize=7, elinewidth=2, capsize=0, zorder=3)
    style_axes(a2)
    a2.set_ylabel("Mean OKS change vs PyTorch (OKS points)")
    a2.set_title("Accuracy vs human labels: INT8 costs only on large people", loc="left", fontsize=11.5)
    for a in (a1, a2):
        a.set_xticks(xs)
        a.set_xticklabels(labels, fontsize=9)
        a.set_xlabel("Crop height (COCO val2017 people, shrunk to it)")
    fig.text(0.01, -0.05, "TensorRT INT8 against PyTorch FP16 on the same 4,149-6,352 annotated people per set. Right: "
             "bootstrap 95% interval; above 0 means INT8 is closer to the human labels. One OKS point = 0.01 OKS.",
             ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, output_path)


def _row(rows: list[dict], label: str) -> dict:
    return next(r for r in rows if r["label"] == label)


def generate_steps(output_path: Path = IMAGES / "upgrades_steps.png") -> Path:
    """Each upgrade step's throughput and CPU, from the phase that measured it (every step is
    paired with its predecessor inside its own interleaved run)."""
    apply_style()
    codec = _load("pipeline/stage11_codec.json", "python -m pipeline.upgrades codec")["variants"]
    det = _load("pipeline/stage12_detector.json", "python -m pipeline.upgrades detector")["variants"]
    pose = _load("pipeline/stage13_pose.json", "python -m pipeline.upgrades pose")["variants"]
    steps = [("Stage 9\n(HF codec, fp32 YOLO,\nTensorRT FP16)", _row(codec, "codec=hf"), YELLOW),
             ("+ GPU codec\n(Stage 11)", _row(codec, "codec=gpu"), YELLOW),
             ("+ TensorRT YOLO\n(Stage 12)", _row(det, "detector=trt16"), YELLOW),
             ("+ FP16/INT8 hybrid,\nmicro-batch 8 (Stage 13,\nopt-in)", _row(pose, "trt-hybrid/b8"), MAGENTA)]
    xs = list(range(len(steps)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    for ax, key, fmt, title, ylabel in (
            (a1, "fps", "{:.1f}", "Throughput", "Frames per second, end to end"),
            (a2, "cpu_cores_busy_total", "{:.1f}", "CPU used by the pipeline", "Cores busy (all its processes)")):
        vals = [r[key] for _, r, _ in steps]
        ax.bar(xs, vals, width=0.55, color=[c for *_, c in steps], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)
        style_axes(ax)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for lbl, *_ in steps], fontsize=8.5)
    fig.text(0.01, -0.06, "Benchmark footage (6,092 frames, 3.5 people per frame), asynchronous pipeline, YOLO on 4 frames "
             "per call. Each step is measured against the one before it in the same interleaved run. Magenta: routes "
             "crops of 64 px and up to INT8 (passes its gates; costs 0.5 COCO AP on large people, Stage 10).",
             ha="left", va="top", fontsize=8.5, color=INK_SECONDARY, wrap=True)
    fig.tight_layout()
    return _save(fig, output_path)


def generate_resolutions(output_path: Path = IMAGES / "upgrades_resolutions.png") -> Path:
    apply_style()
    res = _load("pipeline/stage14_resolutions.json", "python -m pipeline.upgrades resolutions --also-hybrid")
    rows = {r["label"]: r for r in res["variants"]}
    order = sorted(rows["stage9"]["by_resolution"], key=lambda k: -rows["stage9"]["by_resolution"][k]["frames"])
    series = [("stage9", "Stage 9 configuration", BLUE), ("upgraded", "Upgraded (all FP16)", YELLOW),
              ("upgraded+hybrid", "Upgraded + hybrid (opt-in)", MAGENTA)]
    fig, ax = plt.subplots(figsize=(8.2, 0.75 * len(order) + 1.6))
    for y, k in enumerate(order):
        vals = [rows[s]["by_resolution"][k]["fps"] for s, *_ in series if s in rows]
        ax.plot([min(vals), max(vals)], [y, y], color=DEEMPHASIS, linewidth=2, zorder=2)
        for s, label, color in series:
            if s in rows:
                ax.scatter(rows[s]["by_resolution"][k]["fps"], y, s=64, color=color, edgecolor="white",
                           linewidth=1.5, zorder=3, label=label if y == 0 else None)
    style_axes(ax, ygrid=False)
    ax.xaxis.grid(True, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{k}  ({rows['stage9']['by_resolution'][k]['frames']:,} frames)" for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frames per second, end to end")
    ax.set_xlim(0, None)
    ax.set_title("Stage 14: throughput on the job videos that aren't 1280x720", loc="left", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, -0.02, "Every frame of every such job video, asynchronous pipeline, variants interleaved per "
             "1000-frame chunk. Short videos include the pipeline's fill and drain.",
             ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, output_path)


def generate() -> list[Path]:
    return [generate_coco_int8(), generate_steps(), generate_resolutions()]


if __name__ == "__main__":
    for path in generate():
        print(f"[upgrades] wrote {path}")
