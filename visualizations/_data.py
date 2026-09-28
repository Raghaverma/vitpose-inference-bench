"""Single point of contact between visualizations/*.py and the repo's result
JSON files. No script in this package should call json.load() directly --
every number that ends up in a PNG flows through here, so there is exactly
one place that knows which file each figure needs, what "complete" means for
it, and what to say when it's missing.

Two failure classes this guards against, deliberately differently:

1. Data that COULD exist locally but doesn't yet (Stage 1's
   results/raw/stage1_report.json is gitignored -- a fresh clone won't have
   it until someone runs stage1_benchmark.py). This fails loud with the
   exact command to run. No fallback, no skipped figure, no cached default.

2. Data that doesn't exist ANYWHERE yet because the work hasn't happened
   (TensorRT / Stage 3). This is not a loader error -- see Unmeasured below.
   There is no command that produces it, so crashing would be the wrong
   signal; the plotting scripts must represent it as absent by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class _UnmeasuredType:
    """Sentinel for a metric that has not been benchmarked yet (TensorRT FP16/
    INT8 as of Stage 2). Never compares equal to a number, never formats as
    one -- numeric() below is the only supported way to pull a plottable value
    out of a slot that might hold this, and it raises rather than coerce."""
    def __repr__(self) -> str:
        return "NOT_YET_MEASURED"

    def __bool__(self) -> bool:
        return False


NOT_YET_MEASURED = _UnmeasuredType()


def numeric(value, context: str = "") -> float:
    """The only sanctioned way to pull a float out of something that might be
    NOT_YET_MEASURED. Raises instead of silently plotting 0 or a string."""
    if value is NOT_YET_MEASURED:
        raise TypeError(
            f"Attempted to plot a NOT_YET_MEASURED value{' (' + context + ')' if context else ''} "
            f"as a number. This metric hasn't been benchmarked yet -- render it with "
            f"draw_unmeasured_* helpers in _diagram.py, never as a numeric bar/point.")
    return float(value)


def _require_file(path: Path, how_to_generate: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"\n\nMissing required result file: {path}\n"
            f"{'(this path is gitignored -- not committed, only produced locally)' if 'raw' in str(path) else ''}\n"
            f"Generate it by running:\n\n    {how_to_generate}\n")
    return json.loads(path.read_text())


def assert_positive(value: float, name: str) -> None:
    if not (value > 0):
        raise ValueError(f"Physical-plausibility check failed: {name} = {value!r}, expected > 0. "
                          f"The source JSON parsed fine but contains an implausible value -- "
                          f"treating this as corrupt data rather than plotting it.")


def load_stage0() -> dict:
    """Stage 0 frozen baseline. Git-tracked, always present on a clean clone."""
    path = REPO_ROOT / "results" / "baseline" / "l4_fp16.json"
    data = _require_file(path, "(should be committed -- if missing, something's wrong with the checkout)")
    for field in ("mean_latency_ms", "throughput_fps", "peak_vram_mb"):
        if field not in data:
            raise KeyError(f"{path} is missing required field '{field}'")
        assert_positive(data[field], f"stage0.{field}")
    return data


def load_stage1() -> dict:
    """Stage 1 detailed benchmarking. NOT git-tracked (results/raw/ is
    gitignored) -- only exists locally after running stage1_benchmark.py."""
    path = REPO_ROOT / "results" / "raw" / "stage1_report.json"
    data = _require_file(
        path, f"python stage1_benchmark.py --image samples/sample.jpg\n\n"
              f"    This produces the stage-split timing, noise floor, and batch-size sweep "
              f"that stage_breakdown.py and batch_scaling.py both require.")
    for field in ("noise_floor", "stage_split", "batch_sweep"):
        if field not in data:
            raise KeyError(f"{path} is missing required field '{field}' -- looks like an old or "
                            f"partial run. Re-run stage1_benchmark.py.")
    assert_positive(data["noise_floor"]["mean_ms"], "stage1.noise_floor.mean_ms")
    for stage_name, stats in data["stage_split"]["stages"].items():
        assert_positive(stats["mean_ms"], f"stage1.stage_split.stages.{stage_name}.mean_ms")
    return data


def assert_batch_sweep_complete(stage1_data: dict, required_sizes: list[int]) -> list[dict]:
    sweep = stage1_data["batch_sweep"]
    present = {entry["batch_size"] for entry in sweep}
    missing = sorted(set(required_sizes) - present)
    if missing:
        raise ValueError(
            f"Batch sweep is incomplete: missing batch size(s) {missing} in "
            f"results/raw/stage1_report.json (found {sorted(present)}, need {required_sizes}). "
            f"Refusing to plot a partial sweep as if it were complete -- re-run "
            f"stage1_benchmark.py --batch-sizes {' '.join(map(str, required_sizes))}.")
    for entry in sweep:
        assert_positive(entry["mean_ms"], f"stage1.batch_sweep[batch={entry['batch_size']}].mean_ms")
        assert_positive(entry["fps"], f"stage1.batch_sweep[batch={entry['batch_size']}].fps")
    return sorted(sweep, key=lambda e: e["batch_size"])


def load_stage2() -> dict:
    """Stage 2 ONNX export + validation. Git-tracked."""
    onnx_dir = REPO_ROOT / "results" / "onnx"
    export_metadata = _require_file(onnx_dir / "export_metadata.json",
                                     "python -m conversion.export_onnx")
    structural = _require_file(onnx_dir / "structural_validation.json",
                                "python -m conversion.inspect_onnx")
    equivalence = _require_file(onnx_dir / "equivalence_report.json",
                                 "pytest tests/test_onnx_equivalence.py -v")
    benchmark = _require_file(onnx_dir / "benchmark.json", "python -m backends.onnxruntime")

    for run_name, stats in benchmark["runs"].items():
        assert_positive(stats["mean_ms"], f"stage2.benchmark.runs.{run_name}.mean_ms")
        assert_positive(stats["fps"], f"stage2.benchmark.runs.{run_name}.fps")

    faithful = benchmark["runs"]["unoptimized_graph"]["mean_ms"]
    optimized = benchmark["runs"]["ort_default_optimizations"]["mean_ms"]
    if optimized > faithful:
        raise ValueError(
            f"Physical-plausibility check failed: ORT's own graph optimizations "
            f"(mean {optimized:.2f}ms) claim to be SLOWER than the faithful, unoptimized "
            f"graph (mean {faithful:.2f}ms). Enabling optimizations should never regress "
            f"latency -- treating this as a bad measurement, not a real result.")

    pt_baseline = benchmark.get("pytorch_baseline")
    if pt_baseline is not None:
        stage0 = load_stage0()
        drift = abs(pt_baseline["mean_latency_ms"] - stage0["mean_latency_ms"])
        if drift > 2.0:  # ms -- generous vs. observed run-to-run noise (<1ms), catches real staleness
            raise ValueError(
                f"results/onnx/benchmark.json's embedded PyTorch baseline "
                f"({pt_baseline['mean_latency_ms']:.2f}ms) has drifted {drift:.2f}ms from the "
                f"currently frozen results/baseline/l4_fp16.json ({stage0['mean_latency_ms']:.2f}ms). "
                f"One of these is stale -- re-run baseline.py and/or backends/onnxruntime.py.")

    return {
        "export_metadata": export_metadata,
        "structural_validation": structural,
        "equivalence": equivalence,
        "benchmark": benchmark,
    }


def load_stage3_fp16() -> dict | None:
    """Stage 3 TensorRT FP16. Returns None (not an error) if not yet built --
    unlike Stage 0-2, this artifact is genuinely optional right now: TensorRT
    the figures need to keep working (with TensorRT drawn as unmeasured)
    before this stage lands."""
    path = REPO_ROOT / "results" / "tensorrt" / "fp16.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    stats = data["benchmark"]
    assert_positive(stats["mean_ms"], "stage3_fp16.benchmark.mean_ms")
    assert_positive(stats["fps"], "stage3_fp16.benchmark.fps")

    pt_baseline = data.get("pytorch_baseline")
    if pt_baseline is not None:
        stage0 = load_stage0()
        drift = abs(pt_baseline["mean_latency_ms"] - stage0["mean_latency_ms"])
        if drift > 2.0:
            raise ValueError(
                f"results/tensorrt/fp16.json's embedded PyTorch baseline "
                f"({pt_baseline['mean_latency_ms']:.2f}ms) has drifted {drift:.2f}ms from the "
                f"currently frozen results/baseline/l4_fp16.json ({stage0['mean_latency_ms']:.2f}ms). "
                f"One of these is stale -- re-run baseline.py and/or backends/tensorrt.py.")
    return data


def load_stage6_int8() -> dict | None:
    """Stage 6 TensorRT INT8 (batch 1). None if not built -- the architecture figure then draws
    INT8 as unmeasured. Refuses a benchmark file whose real-crop correctness gate is missing:
    an INT8 latency without its accuracy check is exactly what the first INT8 attempt had."""
    path = REPO_ROOT / "results" / "tensorrt" / "int8.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not data.get("real_crop_correctness"):
        raise ValueError(f"{path} has no real_crop_correctness block -- re-run "
                         f"python -m backends.tensorrt --precision int8")
    assert_positive(data["benchmark"]["mean_ms"], "stage6_int8.benchmark.mean_ms")
    return data


def load_batch_matrix() -> dict:
    """Stage 4's 3-backend x 5-batch-size comparison. Produced by
    aggregate_batch_matrix.py from files backends/pytorch.py,
    backends/onnxruntime.py, and backends/tensorrt.py already wrote --
    fails loud with the aggregation command if missing, same as every
    other loader here."""
    path = REPO_ROOT / "results" / "batch_matrix.json"
    data = _require_file(path, "python aggregate_batch_matrix.py\n\n"
                          "    (after running backends/pytorch.py, backends/onnxruntime.py, and "
                          "backends/tensorrt.py --batch-size B for B in 1,2,4,8,16)")
    for cell in data["cells"]:
        for backend in ("pytorch", "onnxruntime", "tensorrt", "tensorrt_int8"):
            stats = cell.get(backend)
            if stats is not None:
                assert_positive(stats["mean_ms"], f"batch_matrix[batch={cell['batch']}].{backend}.mean_ms")
                assert_positive(stats["fps"], f"batch_matrix[batch={cell['batch']}].{backend}.fps")
    return data
