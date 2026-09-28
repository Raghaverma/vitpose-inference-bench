#!/usr/bin/env python3
"""Stage 6 Gate C: INT8 TensorRT engines vs the PyTorch FP16 reference, on real crops.

    pytest tests/test_tensorrt_int8_equivalence.py -v

Unlike Stages 2-3, the gate here is NOT the single golden crop: INT8 error
depends on the input distribution, and the golden crop (plus Gate A's
synthetic corpus, all derived from that one photo) is exactly what let the
first INT8 engine ship broken -- 72 px on golden, and nothing else was ever
measured. The gate is the 400 held-out real crops of
calibration/real_corpus.py (videos never used for calibration), each
compared against its own stored PyTorch FP16 reference heatmap, with the
four INT8 gates in compare_golden.py (heatmap RMSE, and the median / p95 /
share-over-5-px of per-joint keypoint error in model-input pixels).

Every batch size with a tuned INT8 engine is checked, running the 400 crops
through it in batches of B: each slot is compared against that crop's own
standalone reference, so cross-slot contamination fails here the same way
backends/tensorrt.py's FP16 per-slot check catches it.

The golden crop is still reported, and loosely gated, for continuity with
Stages 2-3 -- but as a hard synthetic-lineage case it overstates typical
INT8 drift, so its threshold is a tripwire for a broken engine, not the
accuracy claim.

Only dataset_index 0 (COCO, what production runs) is calibrated or tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from compare_golden import compare_crop_batch, compare_outputs, int8_gate_failures

BATCH_SIZES = [1, 2, 4, 8, 16]
RESULTS = REPO_ROOT / "results" / "tensorrt"
GOLDEN_DIR = REPO_ROOT / "golden"
# Golden crop tripwire, source-frame px (the golden box is ~350 px tall). First measured with
# the shipped recipe: see results/tensorrt/int8_equivalence_report.json. The replaced recipe
# measured 72 px here.
GOLDEN_MEAN_KP_THRESHOLD_PX = 5.0


def _suffix(b: int) -> str:
    return "" if b == 1 else f"_b{b}"


def _available_batches() -> list[int]:
    return [b for b in BATCH_SIZES if (RESULTS / f"int8_engine_metadata{_suffix(b)}.json").is_file()]


@pytest.fixture(scope="module")
def processor():
    proc, _ = load_pose_model(resolve_checkpoint(None), "cpu", torch.float16)
    return proc


@pytest.fixture(scope="module")
def eval_set():
    from calibration.real_corpus import RealCropSet
    try:
        return RealCropSet("eval")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _run_engine(batch: int, pixel_values: np.ndarray) -> np.ndarray:
    from backends.tensorrt import load_engine, run_inference, verify_manifest
    manifest = verify_manifest(RESULTS / f"int8_engine_metadata{_suffix(batch)}.json")
    assert manifest["batch"] == batch
    engine = load_engine(Path(manifest["engine_path"]))
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    di = torch.zeros((batch,), dtype=torch.int64, device="cuda")
    outs = []
    for k in range(0, len(pixel_values), batch):
        x = torch.from_numpy(pixel_values[k:k + batch]).cuda()
        with torch.cuda.stream(stream):
            y = run_inference(engine, context, {"pixel_values": x, "dataset_index": di}, stream.cuda_stream)
        outs.append(y["heatmaps"].float().cpu().numpy())
    return np.concatenate(outs)


@pytest.fixture(scope="module", params=_available_batches() or [None], ids=lambda b: f"batch{b}")
def report(request, eval_set, processor) -> dict:
    batch = request.param
    if batch is None:
        pytest.skip("no INT8 engine built -- python -m conversion.build_int8_engine [--batch-size B]")
    if len(eval_set) % batch:
        pytest.fail(f"{len(eval_set)} eval crops don't divide into batches of {batch}")
    candidate = _run_engine(batch, eval_set.pixel_values)
    result = {
        "candidate": f"TensorRT INT8 (batch={batch}) vs PyTorch FP16",
        "batch": batch,
        "eval_manifest_sha256": eval_set.manifest_sha256,
        "real_crops": compare_crop_batch(eval_set.ref_heatmaps, candidate, processor, eval_set.boxes_xywh,
                                         eval_set.scale_px[:, 1]),
    }
    if batch == 1:
        env = json.loads((GOLDEN_DIR / "env.json").read_text())
        crop = np.load(GOLDEN_DIR / "person_crop.npy").astype(np.float16)
        golden = np.load(GOLDEN_DIR / "pytorch_fp16_output.npy")
        result["golden_crop"] = compare_outputs(golden, _run_engine(1, crop), processor, env["box_xywh"],
                                                candidate_name="TensorRT INT8 (batch=1)", golden_env=env)
    result["gate_failures"] = int8_gate_failures(result["real_crops"])
    path = RESULTS / f"int8_equivalence_report{_suffix(batch)}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    kp = result["real_crops"]["keypoint_err_crop_px"]
    print(f"\n[int8 batch={batch}] median {kp['median']:.3f} px  p95 {kp['p95']:.3f} px  "
          f">5px {kp['pct_over_5px']:.2f}%  rmse {result['real_crops']['tensor_diff']['rmse']:.5f}  -> {path}")
    return result


def test_real_crop_int8_gates(report):
    failures = report["gate_failures"]
    assert not failures, f"batch={report['batch']}: " + "; ".join(failures)


def test_every_crop_scored(report, eval_set):
    assert report["real_crops"]["num_crops"] == len(eval_set)
    assert report["real_crops"]["num_joints_scored"] > 0.5 * len(eval_set) * 17


def test_golden_crop_tripwire(report):
    if report["batch"] != 1:
        pytest.skip("golden crop is checked on the batch-1 engine only")
    mean_px = report["golden_crop"]["keypoint_diff"]["mean_distance_px"]
    assert mean_px < GOLDEN_MEAN_KP_THRESHOLD_PX, (
        f"golden crop mean keypoint distance {mean_px:.2f} px >= {GOLDEN_MEAN_KP_THRESHOLD_PX} px")
