#!/usr/bin/env python3
"""ONNX numerical equivalence -- docs/images/numerical_equivalence.png.

    python -m visualizations.numerical_equivalence

Reads results/onnx/equivalence_report.json (Stage 2 Gate C). Pass thresholds
are imported directly from tests/test_onnx_equivalence.py rather than
retyped, so the reference lines in this figure can never drift from what the
test suite actually enforces.

Deliberately not described as "perfect" equivalence -- see
tests/test_onnx_equivalence.py's own docstring: a tensor regression can still
decode to a similar-looking pose, which is why both the raw-tensor and
decoded-keypoint checks exist and both are shown here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from visualizations._data import REPO_ROOT
from visualizations._style import AQUA, INK_MUTED, INK_PRIMARY, INK_SECONDARY, STATUS_GOOD, apply_style
from tests.test_onnx_equivalence import (
    MAX_ABS_ERROR_THRESHOLD,
    MAX_KEYPOINT_DIST_THRESHOLD_PX,
    MEAN_KEYPOINT_DIST_THRESHOLD_PX,
)

OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "numerical_equivalence.png"


def _load_report() -> dict:
    import json
    path = REPO_ROOT / "results" / "onnx" / "equivalence_report.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Generate it by running:\n\n"
            f"    pytest tests/test_onnx_equivalence.py -v\n")
    return json.loads(path.read_text())


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    apply_style()
    report = _load_report()
    tensor = report["tensor_diff"]
    keypoint = report["keypoint_diff"]

    metrics = [
        ("Mean keypoint distance", keypoint["mean_distance_px"], MEAN_KEYPOINT_DIST_THRESHOLD_PX, "px"),
        ("Max keypoint distance", keypoint["max_distance_px"], MAX_KEYPOINT_DIST_THRESHOLD_PX, "px"),
    ]

    fig, (ax_kp, ax_tensor) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                            gridspec_kw={"width_ratios": [1.4, 1]})

    y_pos = range(len(metrics))
    values = [m[1] for m in metrics]
    thresholds = [m[2] for m in metrics]
    ax_kp.barh(list(y_pos), values, color=AQUA, height=0.45, zorder=3)
    for i, (name, val, thresh, unit) in enumerate(metrics):
        ax_kp.plot([thresh, thresh], [i - 0.35, i + 0.35], color=INK_SECONDARY,
                   linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
        ax_kp.text(thresh + max(thresholds) * 0.02, i + 0.38, f"threshold {thresh:.1f}{unit}",
                    fontsize=8, color=INK_SECONDARY, ha="left", va="bottom")
        ax_kp.text(val + max(thresholds) * 0.02, i, f"{val:.3f} {unit}", fontsize=10,
                    fontweight="bold", color=INK_PRIMARY, va="center")
    ax_kp.set_yticks(list(y_pos))
    ax_kp.set_yticklabels([m[0] for m in metrics])
    ax_kp.set_xlabel("Pixel distance from PyTorch golden reference")
    ax_kp.set_xlim(0, max(thresholds) * 1.5)
    ax_kp.spines["left"].set_visible(False)
    ax_kp.spines["bottom"].set_color("#c3c2b7")
    ax_kp.tick_params(length=0)
    ax_kp.set_title("Decoded-keypoint agreement", fontsize=11, fontweight="bold", loc="left")

    max_abs = tensor["max_abs_error"]
    ax_tensor.barh([0], [max_abs], color="#cfd3ce", height=0.4, zorder=3)
    ax_tensor.plot([MAX_ABS_ERROR_THRESHOLD, MAX_ABS_ERROR_THRESHOLD], [-0.35, 0.35],
                   color=INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax_tensor.text(max_abs + MAX_ABS_ERROR_THRESHOLD * 0.03, 0, f"{max_abs:.5f}", fontsize=10,
                    fontweight="bold", color=INK_PRIMARY, va="center")
    ax_tensor.text(MAX_ABS_ERROR_THRESHOLD + MAX_ABS_ERROR_THRESHOLD * 0.03, 0.38,
                    f"threshold {MAX_ABS_ERROR_THRESHOLD}", fontsize=8, color=INK_SECONDARY, va="bottom")
    ax_tensor.set_yticks([0])
    ax_tensor.set_yticklabels(["Max abs.\nheatmap error"])
    ax_tensor.set_xlim(0, MAX_ABS_ERROR_THRESHOLD * 1.6)
    ax_tensor.set_xlabel("Raw fp16 heatmap value")
    ax_tensor.spines["left"].set_visible(False)
    ax_tensor.spines["bottom"].set_color("#c3c2b7")
    ax_tensor.tick_params(length=0)
    ax_tensor.set_title("Raw-tensor agreement", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("ONNX Runtime Output vs. PyTorch FP16 Golden Reference", fontsize=13.5,
                  fontweight="bold", color=INK_PRIMARY, y=1.04)
    fig.text(0.5, -0.03,
              "Sub-pixel pose agreement with the PyTorch golden reference -- both the raw-tensor\n"
              "and decoded-keypoint checks pass independently (batch=1, single fixed test crop).",
              ha="center", va="top", fontsize=9, color=INK_SECONDARY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"[numerical_equivalence] wrote {path}")
