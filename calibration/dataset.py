#!/usr/bin/env python3
"""Stage 6 Gate A: the calibration-corpus loader.

    python -m calibration.dataset

`CalibrationDataset` reads calibration/manifest.json (written by
calibration/build_calibration_corpus.py) and calibration/samples/*.npy, and
exposes `get_batch(index)` in the exact `{"pixel_values": ..., "dataset_index":
...}` shape backends/tensorrt.py already feeds a live engine -- so a future
`trt.IInt8EntropyCalibrator2` subclass (Gate B, not built here) can call
straight through to this class's `get_batch` without redesigning the input
contract.

Why this file re-verifies the corpus instead of just `json.load`-ing it
------------------------------------------------------------------------
This manifest exists to police exactly one thing: that TensorRT's INT8
calibrator sees enough activations from EVERY one of the backbone MoE's 6
experts (VitPoseNaiveMoe.forward -- see
transformers/models/vitpose_backbone/modeling_vitpose_backbone.py). We
already checked, empirically, whether that's auditable any other way: an
EngineInspector dump of the built FP16 engine
(results/tensorrt/fp16_layer_info_full.json, 180 layers, DETAILED
verbosity) shows builder_optimization_level=3 fuses the entire per-expert
routing -- all 6 experts' Linears plus the shared branch -- into one opaque
"gemm" layer per block (e.g. layer 10, fusing 7 MatMuls) followed by one
opaque "kgen" mask-and-sum kernel (e.g. layer 12,
__myl_MulMulMulMulMulMulAddAddAddAddAddAddConcAdd...), repeating across all
24 transformer blocks, with zero semantic layer names surviving (grepping
the dump for Gather/Select/Where/expert-named layers: zero matches). So
per-expert calibration quality is NOT checkable at the tensor level once
the engine is built (that would need a downstream accuracy check split by
dataset_index -- Gate C/D, not this file's job). This manifest's
per-expert sample-count floor is therefore the ONLY preventive control this
repo has, which is exactly why this loader never trusts it blindly:

  - per_expert_counts is RECOMPUTED from manifest["samples"] here, not read
    off the manifest's own hand-typed field, and any disagreement (a hand
    edit, or a bug in the corpus builder) is a hard refusal naming the
    mismatched expert(s) -- same "recompute and diff, never trust the
    recorded number" discipline conversion/build_engine.py uses for
    onnx_sha256/engine_sha256.
  - Every sample file's sha256 is re-hashed (via conversion/build_engine.py's
    sha256_file, the same helper the corpus builder used to compute the
    value being checked against) and compared to the manifest's recorded
    value, catching a sample silently edited/truncated/swapped since the
    corpus was built -- a mismatch per_expert_counts alone is blind to,
    since a corrupted file still counts toward its expert's tally.
  - Any expert whose (verified) sample count is below
    manifest["min_samples_per_expert"] is a hard SystemExit, not a warning.

None of these gates repair anything. A failure here means the corpus is
untrustworthy; the fix is to delete calibration/samples/ and
calibration/manifest.json and re-run
`python -m calibration.build_calibration_corpus`, not to patch this loader
into accepting what it found.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline import REPO_ROOT
from conversion.build_engine import sha256_file

CALIBRATION_DIR = REPO_ROOT / "calibration"
DEFAULT_MANIFEST_PATH = CALIBRATION_DIR / "manifest.json"

REQUIRED_MANIFEST_KEYS = {
    "num_experts", "min_samples_per_expert", "resolution", "per_expert_counts", "samples",
}


class CalibrationDataset:
    """Gate A's verified view of the calibration corpus. See module
    docstring for exactly what __init__ checks and why -- every check is a
    refuse-and-exit, never a silent repair."""

    def __init__(self, manifest_path: Path | str = DEFAULT_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"\n\nMissing {self.manifest_path}. Build the calibration corpus first:\n\n"
                f"    python -m calibration.build_calibration_corpus\n")
        self.manifest: dict = json.loads(self.manifest_path.read_text())

        missing_keys = REQUIRED_MANIFEST_KEYS - self.manifest.keys()
        if missing_keys:
            raise SystemExit(
                f"[calibration.dataset] REFUSING to load {self.manifest_path}: missing required "
                f"top-level field(s) {sorted(missing_keys)}. This doesn't match the schema "
                f"calibration/build_calibration_corpus.py writes -- it may be stale, hand-edited, "
                f"or from a different generator. Regenerate it; don't hand-patch it.")

        self.num_experts = int(self.manifest["num_experts"])
        self.min_samples_per_expert = int(self.manifest["min_samples_per_expert"])
        self.resolution = tuple(self.manifest["resolution"])
        self.samples: list[dict] = self.manifest["samples"]

        if not self.samples:
            raise SystemExit(
                f"[calibration.dataset] REFUSING to load {self.manifest_path}: 'samples' is "
                f"empty -- an INT8 calibration corpus with zero samples isn't a corpus.")

        # ---- Recompute per_expert_counts from the samples themselves. Requirement 2: never
        # trust manifest["per_expert_counts"] as reported -- it's a hand-computed-at-build-time
        # summary field, exactly the kind of self-reported number this repo's other sha256/
        # field-diff gates (conversion/build_engine.py, backends/tensorrt.py's verify_manifest)
        # refuse to take on faith.
        recomputed_counts: dict[str, int] = {str(i): 0 for i in range(self.num_experts)}
        for entry in self.samples:
            dataset_index = entry.get("dataset_index")
            if not isinstance(dataset_index, int) or not (0 <= dataset_index < self.num_experts):
                raise SystemExit(
                    f"[calibration.dataset] REFUSING to load {self.manifest_path}: sample "
                    f"{entry.get('sample_id', '?')} has dataset_index={dataset_index!r}, outside "
                    f"the valid expert range [0, {self.num_experts}) declared by "
                    f"manifest['num_experts']={self.num_experts}.")
            recomputed_counts[str(dataset_index)] += 1

        declared_counts = {str(k): int(v) for k, v in self.manifest["per_expert_counts"].items()}
        mismatched_experts = sorted(
            expert for expert in recomputed_counts
            if recomputed_counts[expert] != declared_counts.get(expert)
        )
        if mismatched_experts:
            detail = "; ".join(
                f"expert {e}: manifest claims {declared_counts.get(e)!r}, actual sample count "
                f"is {recomputed_counts[e]}" for e in mismatched_experts)
            raise SystemExit(
                f"[calibration.dataset] REFUSING to load {self.manifest_path}: recomputed "
                f"per-expert sample counts disagree with the manifest's own per_expert_counts "
                f"field for expert(s) {mismatched_experts} -- {detail}. This manifest is "
                f"self-inconsistent (hand-edited, or written by an interrupted/buggy corpus "
                f"build); regenerate with calibration/build_calibration_corpus.py rather than "
                f"trusting either number.")
        # From here on, self.per_expert_counts is the VERIFIED count, not the manifest's claim
        # (they're now known equal, but this name makes clear which one downstream code reads).
        self.per_expert_counts = recomputed_counts

        # ---- Requirement 3, and this corpus's ONLY preventive control (see module docstring):
        # refuse to load if any expert is under manifest["min_samples_per_expert"].
        starved_experts = sorted(
            expert for expert, count in self.per_expert_counts.items()
            if count < self.min_samples_per_expert
        )
        if starved_experts:
            detail = ", ".join(
                f"expert {e} has {self.per_expert_counts[e]}" for e in starved_experts)
            raise SystemExit(
                f"[calibration.dataset] REFUSING to load {self.manifest_path}: expert(s) "
                f"{starved_experts} have fewer than min_samples_per_expert="
                f"{self.min_samples_per_expert} calibration samples ({detail}). Per-expert INT8 "
                f"calibration quality is NOT auditable at the tensor level once the engine is "
                f"built (see manifest['moe_coverage_policy_note'] -- the MoE routing is fully "
                f"fused by TensorRT's builder, confirmed via EngineInspector), so this "
                f"sample-count floor is the ONLY preventive control this repo has for it -- an "
                f"under-covered expert is a hard stop here, not a warning. Regenerate: "
                f"python -m calibration.build_calibration_corpus")

        # ---- Per-sample integrity: each sample file must exist and its content must still
        # match the sha256 the corpus build recorded. A count-based gate alone can't catch a
        # sample that was silently edited/truncated/swapped after the corpus was built -- a
        # corrupted file still counts toward its expert's tally, so this is a distinct check,
        # not a restatement of the count gate above.
        corrupted = []
        for entry in self.samples:
            sample_path = REPO_ROOT / entry["path"]
            if not sample_path.is_file():
                raise FileNotFoundError(
                    f"[calibration.dataset] {self.manifest_path} references sample "
                    f"{entry['sample_id']} at {sample_path}, which doesn't exist on disk. "
                    f"Regenerate the corpus: python -m calibration.build_calibration_corpus")
            actual_sha256 = sha256_file(sample_path)
            if actual_sha256 != entry["sha256"]:
                corrupted.append((entry["sample_id"], sample_path, entry["sha256"], actual_sha256))
        if corrupted:
            detail = "; ".join(
                f"sample {sid} ({path}): manifest sha256 {expected}, actual {actual}"
                for sid, path, expected, actual in corrupted)
            raise SystemExit(
                f"[calibration.dataset] REFUSING to load {self.manifest_path}: "
                f"{len(corrupted)} sample file(s) don't match their manifest-recorded sha256 -- "
                f"{detail}. Content has changed since the corpus was built; never trust a "
                f"mismatched artifact -- delete calibration/samples/ and calibration/manifest.json "
                f"and rebuild: python -m calibration.build_calibration_corpus")

        print(f"[calibration.dataset] loaded {len(self.samples)} samples from "
              f"{self.manifest_path} -- per-expert counts (recomputed and sha256-verified, all "
              f">= min_samples_per_expert={self.min_samples_per_expert}): {self.per_expert_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, index: int) -> dict[str, np.ndarray]:
        """Returns {"pixel_values": (1, 3, H, W) float16, "dataset_index": (1,) int64} --
        the exact input-dict shape backends/tensorrt.py's run_inference() already feeds a live
        engine (`{"pixel_values": pixel_values, "dataset_index": dataset_index}`), so a future
        IInt8EntropyCalibrator2.get_batch can bind these straight to the calibration engine's
        input tensors without a shape/dtype translation layer in between."""
        if not (0 <= index < len(self.samples)):
            raise IndexError(
                f"[calibration.dataset] index {index} out of range for a "
                f"{len(self.samples)}-sample dataset.")
        entry = self.samples[index]
        sample_path = REPO_ROOT / entry["path"]
        pixel_values = np.load(sample_path)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]  # (3, H, W) -> (1, 3, H, W): samples are saved
                                                # with the batch dim already dropped (see
                                                # build_calibration_corpus.py's variant_np).
        expected_shape = (1, 3, *self.resolution)
        if pixel_values.shape != expected_shape:
            raise SystemExit(
                f"[calibration.dataset] REFUSING to hand out sample {entry['sample_id']} "
                f"({sample_path}): loaded array has shape {pixel_values.shape}, expected "
                f"{expected_shape} from manifest['resolution']={self.resolution}. This sample "
                f"file doesn't match the manifest that describes it -- integrity check gap, or "
                f"the file was replaced after __init__'s sha256 check ran.")
        dataset_index = np.array([entry["dataset_index"]], dtype=np.int64)
        return {"pixel_values": pixel_values, "dataset_index": dataset_index}


def _smoke_test() -> None:
    dataset = CalibrationDataset(DEFAULT_MANIFEST_PATH)
    print(f"[calibration.dataset] __len__ = {len(dataset)}")
    print(f"[calibration.dataset] per-expert counts: {dataset.per_expert_counts}")

    expected_pixel_shape = (1, 3, *dataset.resolution)
    seen_per_expert = {expert: 0 for expert in dataset.per_expert_counts}
    for i in range(len(dataset)):
        batch = dataset.get_batch(i)
        pixel_values, dataset_index = batch["pixel_values"], batch["dataset_index"]
        if pixel_values.shape != expected_pixel_shape:
            raise SystemExit(
                f"[calibration.dataset] smoke test FAILED: sample {i} pixel_values shape "
                f"{pixel_values.shape} != expected {expected_pixel_shape}.")
        if pixel_values.dtype != np.float16:
            raise SystemExit(
                f"[calibration.dataset] smoke test FAILED: sample {i} pixel_values dtype "
                f"{pixel_values.dtype} != expected float16.")
        if dataset_index.shape != (1,) or dataset_index.dtype != np.int64:
            raise SystemExit(
                f"[calibration.dataset] smoke test FAILED: sample {i} dataset_index shape/dtype "
                f"{dataset_index.shape}/{dataset_index.dtype} != expected (1,)/int64.")
        seen_per_expert[str(int(dataset_index[0]))] += 1

    if seen_per_expert != dataset.per_expert_counts:
        raise SystemExit(
            f"[calibration.dataset] smoke test FAILED: dataset_index values actually returned "
            f"by get_batch() ({seen_per_expert}) don't match __init__'s verified "
            f"per_expert_counts ({dataset.per_expert_counts}).")

    print(f"[calibration.dataset] smoke test OK: all {len(dataset)} samples iterated via "
          f"get_batch(), every pixel_values {expected_pixel_shape} float16, every "
          f"dataset_index (1,) int64, per-expert tallies from get_batch() match __init__'s "
          f"verified per_expert_counts.")


if __name__ == "__main__":
    _smoke_test()
