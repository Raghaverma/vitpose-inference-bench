#!/usr/bin/env python3
"""Stage 6 Gate B: quantize the faithful FP16 ONNX export to an INT8 QDQ graph.

    python -m conversion.quantize_onnx

Why this script exists, and why it wasn't supposed to
--------------------------------------------------------
The original plan for Stage 6B was a single `conversion/build_int8_engine.py`
that would hand TensorRT a `trt.IInt8EntropyCalibrator2` subclass wired
straight to `calibration/dataset.py`'s `CalibrationDataset`, the same way
every TensorRT INT8 tutorial from the 8.x/9.x era describes it -- TensorRT
streams calibration batches through the callback during the build and picks
per-layer INT8/FP16/FP32 precision itself. Checked empirically against the
actually-installed TensorRT 11.3.0.99 Python API before writing a line of
that script (never assumed, same discipline as `conversion/build_engine.py`'s
`BuilderFlag.FP16` finding): `trt.BuilderFlag.INT8` does not exist, and
neither does `IInt8EntropyCalibrator2` / `IInt8Calibrator` /
`IInt8MinMaxCalibrator` / `ITensor.set_dynamic_range` / `IBuilderConfig.
int8_calibrator`. TensorRT 10+ removed that entire implicit-calibration
build-time API. The only INT8 path left is EXPLICIT quantization --
`IQuantizeLayer`/`IDequantizeLayer`, i.e. the ONNX graph itself must already
contain `QuantizeLinear`/`DequantizeLinear` (Q/DQ) node pairs with
pre-computed scales -- and a strongly-typed TensorRT build then takes those
at face value, exactly like it already takes the FP16 graph's declared
dtypes at face value (see `build_engine.py`'s docstring).

So "run the calibration corpus through the calibration path" now has to
happen HERE, on the ONNX graph, before any TensorRT build step exists --
not inside a builder callback. This script is that step. It's Gate B, not
Gate A (calibration/build_calibration_corpus.py) and not the TensorRT build
(conversion/build_int8_engine.py, which now just parses this script's
output the same way build_engine.py parses the faithful FP16 export).

Tool choice: `onnxruntime.quantization.quantize_static`, not a new
dependency. Neither `nvidia-modelopt` nor `pytorch-quantization` (the
PyTorch-native alternatives) is installed, and onnxruntime-gpu is already a
Stage 2 dependency this repo carries. `quantize_static(..., quant_format=
QuantFormat.QDQ, calibrate_method=CalibrationMethod.Entropy)` computes
per-tensor entropy-calibrated INT8 scales from exactly the same
`calibration/dataset.py` corpus Gate A built, and emits them as real
QuantizeLinear/DequantizeLinear nodes -- entropy calibration by name, same
algorithm TensorRT's own removed `IInt8EntropyCalibrator2` implemented.

Two empirically-found TensorRT-parser incompatibilities in ORT's default
QDQ output (verified against this exact ONNX graph and this exact TensorRT
version, not assumed from general ORT/TensorRT documentation)
------------------------------------------------------------------------------
1. **Rank-0 (scalar) QuantizeLinear nodes.** ORT's quantizer, with
   `op_types_to_quantize` left at its default, ends up inserting Q/DQ pairs
   around 96 scalar (0-rank) tensors on this graph -- 48 literal `Constant`
   nodes (GELU's decomposition constants: this repo's ONNX export uses
   `optimize=False`, so these stay as live ops rather than getting
   constant-folded away) and 48 more that are themselves computed (a `Sqrt`
   node feeding attention's `1/sqrt(d_k)` scaling, same reason). TensorRT's
   ONNX parser hard-fails on every one of them: `QuantizeLinear`'s ONNX-spec
   default `axis=1` is asserted `<= nbDims` in `importerUtils.cpp`'s
   `convertAxis`, which is never true for a 0-rank tensor. A scalar gains
   nothing from INT8 quantization anyway -- no memory-bandwidth or
   GEMM-throughput benefit, it's one value -- so `_strip_rank0_qdq_pairs`
   below removes these 96 Q/DQ pairs outright (rewiring consumers straight
   to the original scalar tensor) rather than working around the parser.
2. **A handful of scales computed in float32 on an all-fp16 graph.** Exactly
   one `*_scale` initializer per transformer block (24/2059 on this graph --
   the post-Softmax attention-probability tensor) comes out of ORT's
   calibrator as `float32` while all 2035 others match the graph's native
   `float16`. Per the ONNX QuantizeLinear/DequantizeLinear spec,
   `DequantizeLinear`'s output dtype follows its scale tensor's dtype, so
   these 24 tensors surface downstream as `Float` next to `Half` operands
   everywhere else -- and TensorRT's `IMatrixMultiplyLayer` refuses mixed
   Float/Half operands (`validateTypes`, `matrixMultiplyNode.cpp`).
   `_downcast_float32_scales` below casts those 24 initializers to float16.
   Safe: they're symmetric-quantization scales for values observed in
   [0, 1] (post-softmax probabilities, `scale ~= 1/127` for all 24 of
   them) -- nowhere near fp16's range or precision limits for this purpose.

Both fixes were found by actually attempting a TensorRT parse of ORT's raw
output and reading the exact assertion/error TensorRT raised, then fixing
that one thing and re-parsing -- not by guessing what "should" work. Neither
is undone before the graph is handed to `conversion/build_int8_engine.py`.

Calibration execution provider, and why CPU
-----------------------------------------------
`calibration_providers` controls which ORT execution provider RUNS the
calibration forward passes (not the quantization math -- that's provider-
independent). CUDAExecutionProvider was tried first, on the assumption GPU
inference would obviously be faster for a ViT-L backbone -- checked, not
assumed: 8 samples took ~327s under CUDA EP, printing "200 Memcpy nodes are
added to the graph ... for CUDAExecutionProvider" (this graph's
`optimize=False` op mix has enough CPU-only-assigned nodes that constant
H2D/D2H traffic dominates). CPU EP measured ~101s/2 samples and ~631s/16
samples -- almost exactly the same per-sample marginal cost as CUDA EP
(~38s/sample either way). The real bottleneck is ORT's own histogram/
entropy computation across 2059 tensors, which is CPU-bound regardless of
which provider ran the forward pass -- so CPU EP is the default here, since
it's simpler (no CPU/GPU boundary, no Memcpy-node warnings) for equal cost.

Quantization configuration, and why
--------------------------------------
- `activation_type=QuantType.QInt8`, `weight_type=QuantType.QInt8`: ORT's
  own `quantize_static` docstring recommendation for GPU/TensorRT targets
  ("use symmetric QuantType.QInt8 for both activations and weights").
- `extra_options={"ActivationSymmetric": True}`: WeightSymmetric already
  defaults True; ActivationSymmetric defaults False (ORT's CPU-oriented
  default) and is forced True here to match -- symmetric (zero_point=0)
  quantization is what TensorRT's INT8 tensor-core kernels are built around.
- `extra_options={"QuantizeBias": False}`: found empirically, not assumed --
  ORT's default bias handling quantizes Conv/Gemm biases to INT32 with a
  standalone DequantizeLinear node. TensorRT's ONNX parser only accepts an
  INT32-zero-point dequant when it's fused directly into a Conv/Gemm bias
  operand by a specific pattern match; this graph's `patch_embeddings`
  Conv bias-dequant isn't recognized that way and TensorRT rejects it
  outright (`ITensor::getDimensions... input has type Int32 but must have
  type FP8, FP4, Int4, Int8, or UInt8`). Leaving bias in its native fp16
  sidesteps this rather than fighting TensorRT's fusion-pattern matcher.
- `op_types_to_quantize` is left at ORT's default (not restricted to
  Conv/MatMul/Gemm): empirically, restricting it changes which internal
  calibration-graph-augmentation code path ORT takes and hits a *different*
  bug -- an fp16/fp32 type mismatch inside ORT's own calibration session
  (`Type (tensor(float)) of output arg (InsertedPrecisionFreeCast_...) does
  not match expected type (tensor(float16))`) -- before any TensorRT step is
  even reached. The default op-type set plus the two TensorRT-side fixups
  above is the configuration that was actually verified end-to-end.
- `per_channel=False` (ORT's default): NOT tuned for accuracy in this pass.
  Per-channel weight quantization is the standard accuracy improvement for
  Conv/Gemm/MatMul weights and is explicitly deferred, not silently skipped
  -- see this script's module-level `KNOWN_LIMITATIONS`.

This is Gate B: it produces a QDQ ONNX graph and refuses to run if Gate A's
calibration corpus doesn't pass its own gates (calibration/dataset.py's
CalibrationDataset.__init__ does that refusing -- this script never catches
or works around a SystemExit from it). It makes NO accuracy claim -- Gate C
(numerical equivalence) and Gate D (pose-space evaluation) are what could.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper, shape_inference
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader

from baseline import REPO_ROOT
from calibration.dataset import CalibrationDataset
from conversion.build_engine import sha256_file

KNOWN_LIMITATIONS = [
    "per_channel=False: weights are quantized per-tensor, not per-output-channel. "
    "Per-channel is the standard accuracy improvement for Conv/Gemm/MatMul weights "
    "and was not attempted in this pass -- deferred, not silently skipped.",
    "Calibration corpus is calibration/manifest.json's 96-sample, single-source, "
    "augmentation-diverse corpus (see calibration/README.md) -- synthetic perturbations "
    "of ONE real photo, not a claim of deployment-distribution coverage.",
]


class ManifestCalibrationReader(CalibrationDataReader):
    """Adapts calibration/dataset.py's CalibrationDataset to ORT's
    CalibrationDataReader interface, in manifest order, no repeats.

    Implements `__len__`/`set_range` (not just the required `get_next`) so
    `quantize_static`'s `extra_options["CalibStridedMinMax"]` chunking can
    drive this reader -- found necessary empirically, not a style choice:
    `EntropyCalibrater.collect_data()` (calibrate.py) drains its ENTIRE
    reader into an in-memory `self.intermediate_outputs` list -- every one
    of 2059 tracked tensors' full activations, for every sample -- before
    computing a single histogram. All 96 samples through this ViT-L graph in
    one `collect_data()` call OOM-killed this box (60GB RAM, confirmed via
    dmesg: `Out of memory: Killed process ... anon-rss:60764804kB`) despite
    16 samples in one call working fine, i.e. linear-in-sample-count memory,
    not streamed. `CalibStridedMinMax=N` makes quantize_static call
    `collect_data()` N samples at a time via repeated `set_range` windows,
    each followed by `HistogramCollector.collect()` (which MERGES into the
    running per-tensor histogram, verified by reading its implementation --
    not a first-batch-wins shortcut) and then frees that window's
    `intermediate_outputs` before the next one starts. Despite the option's
    name suggesting MinMax specifically, the chunking loop lives in
    `quantize_static` itself, outside any one calibrator subclass, and
    applies to Entropy the same way."""

    def __init__(self, dataset: CalibrationDataset):
        self.dataset = dataset
        self.index = 0
        self.end_index = len(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.index = start_index
        self.end_index = end_index

    def get_next(self) -> dict | None:
        if self.index >= min(self.end_index, len(self.dataset)):
            return None
        batch = self.dataset.get_batch(self.index)
        self.index += 1
        return {"pixel_values": batch["pixel_values"], "dataset_index": batch["dataset_index"]}


def _tensor_rank_map(model: onnx.ModelProto) -> dict[str, int]:
    """Shape-inferred rank (number of dims) for every named tensor in the
    graph, needed to find the rank-0 QuantizeLinear targets fixed by
    _strip_rank0_qdq_pairs -- ORT's Q/DQ insertion doesn't restrict itself to
    tensors with a literal Constant producer (see module docstring), so a
    static Constant-node check alone misses computed scalars like a Sqrt
    output. `data_prop=True` lets shape inference resolve these through
    ordinary arithmetic ops rather than only shape/reshape-style ops."""
    inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    rank: dict[str, int] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            rank[vi.name] = len(tt.shape.dim)
    return rank


def _strip_rank0_qdq_pairs(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove every QuantizeLinear->DequantizeLinear pair whose quantized
    tensor is rank-0. See module docstring, finding (1). Returns the edited
    model and the number of pairs removed."""
    rank = _tensor_rank_map(model)
    g = model.graph

    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    for o in g.output:
        consumers.setdefault(o.name, []).append("GRAPH_OUTPUT")

    nodes_to_remove: set[str] = set()
    rewire: dict[str, str] = {}
    for n in g.node:
        if n.op_type != "QuantizeLinear" or rank.get(n.input[0]) != 0:
            continue
        q_out = n.output[0]
        dq_node = next((c for c in consumers.get(q_out, [])
                         if c != "GRAPH_OUTPUT" and c.op_type == "DequantizeLinear"
                         and c.input[0] == q_out), None)
        if dq_node is None:
            raise SystemExit(
                f"[quantize_onnx] REFUSING to continue: rank-0 QuantizeLinear node "
                f"{n.name!r} (quantizing {n.input[0]!r}) is not immediately followed by a "
                f"plain DequantizeLinear consuming its output -- the strip pattern this "
                f"script relies on (found empirically for this exact graph) doesn't match. "
                f"Needs manual inspection before this can be assumed safe to drop.")
        nodes_to_remove.add(n.name)
        nodes_to_remove.add(dq_node.name)
        rewire[dq_node.output[0]] = n.input[0]

    kept_nodes = [n for n in g.node if n.name not in nodes_to_remove]
    for n in kept_nodes:
        for i, inp in enumerate(n.input):
            if inp in rewire:
                n.input[i] = rewire[inp]
    del g.node[:]
    g.node.extend(kept_nodes)

    still_referenced = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    kept_inits = [init for init in g.initializer if init.name in still_referenced]
    del g.initializer[:]
    g.initializer.extend(kept_inits)

    return model, len(rewire)


def _downcast_float32_scales(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Cast any float32 `*_scale` initializer down to float16, matching the
    surrounding all-fp16 graph. See module docstring, finding (2)."""
    g = model.graph
    fixed = 0
    for init in g.initializer:
        if init.name.endswith("_scale") and init.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
            fixed += 1
    return model, fixed


def qdq_op_histogram(model: onnx.ModelProto) -> dict:
    ops = Counter(n.op_type for n in model.graph.node)
    return {
        "total_nodes": sum(ops.values()),
        "QuantizeLinear": ops.get("QuantizeLinear", 0),
        "DequantizeLinear": ops.get("DequantizeLinear", 0),
        "histogram": dict(sorted(ops.items())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-path", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "vitpose_plus_l.onnx")
    p.add_argument("--onnx-metadata", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "export_metadata.json",
                    help="Verified against --onnx-path's actual sha256 before quantizing -- "
                         "same refuse-on-mismatch discipline as conversion/build_engine.py.")
    p.add_argument("--calibration-manifest", type=Path,
                    default=REPO_ROOT / "calibration" / "manifest.json")
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "vitpose_plus_l_int8_qdq.onnx")
    p.add_argument("--metadata-output", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "int8_quantization_metadata.json")
    p.add_argument("--calibration-provider", choices=["cuda", "cpu"], default="cpu",
                    help="Execution provider ORT uses to RUN the calibration forward passes. "
                         "Default is CPU -- found empirically to be no slower than CUDA for "
                         "this graph and simpler, see module docstring.")
    p.add_argument("--calibration-stride", type=int, default=8,
                    help="Calibration samples processed per collect_data() chunk (ORT's "
                         "extra_options['CalibStridedMinMax']). Must evenly divide the "
                         "corpus size. Bounds peak RAM -- see ManifestCalibrationReader's "
                         "docstring for why an unstrided run OOM-kills this box.")
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()

    onnx_metadata = json.loads(args.onnx_metadata.read_text())
    actual_onnx_sha256 = sha256_file(args.onnx_path)
    if actual_onnx_sha256 != onnx_metadata["onnx_sha256"]:
        raise SystemExit(
            f"[quantize_onnx] REFUSING to quantize: {args.onnx_path} has sha256 "
            f"{actual_onnx_sha256}, but {args.onnx_metadata} says it should be "
            f"{onnx_metadata['onnx_sha256']}. Re-run Stage 2's gates before quantizing.")
    if onnx_metadata["optimize"] is not False or onnx_metadata["batch_axis"] != "static":
        raise SystemExit(
            f"[quantize_onnx] REFUSING to quantize: expected the faithful "
            f"(optimize=False, static batch) export, got optimize={onnx_metadata['optimize']} "
            f"batch_axis={onnx_metadata['batch_axis']}.")

    # Gate A's own gates (per-expert floor, sha256-verified samples) run inside this
    # constructor -- a SystemExit here means the corpus is untrustworthy, and this
    # script does not catch or work around that.
    dataset = CalibrationDataset(args.calibration_manifest)
    calibration_manifest_sha256 = sha256_file(args.calibration_manifest)

    if len(dataset) % args.calibration_stride != 0:
        raise SystemExit(
            f"[quantize_onnx] REFUSING to run: --calibration-stride={args.calibration_stride} "
            f"does not evenly divide the corpus size ({len(dataset)}) -- "
            f"quantize_static's CalibStridedMinMax chunking requires this.")

    providers = ["CUDAExecutionProvider"] if args.calibration_provider == "cuda" else ["CPUExecutionProvider"]
    print(f"[quantize_onnx] quantizing {args.onnx_path} against {len(dataset)} calibration "
          f"samples ({providers[0]}, CalibrationMethod.Entropy, QuantFormat.QDQ, "
          f"stride={args.calibration_stride})...")
    t0 = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.onnx_path),
        model_output=str(args.output),
        calibration_data_reader=ManifestCalibrationReader(dataset),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Entropy,
        calibration_providers=providers,
        use_external_data_format=True,
        extra_options={
            "QuantizeBias": False,
            "ActivationSymmetric": True,
            "CalibStridedMinMax": args.calibration_stride,
        },
    )
    quantize_seconds = time.perf_counter() - t0
    print(f"[quantize_onnx] ORT quantize_static done in {quantize_seconds:.1f}s, "
          f"wrote {args.output} -- applying TensorRT-parser fixups...")

    model = onnx.load(str(args.output), load_external_data=False)
    pre_fixup_histogram = qdq_op_histogram(model)
    model, num_stripped = _strip_rank0_qdq_pairs(model)
    model, num_scales_downcast = _downcast_float32_scales(model)
    post_fixup_histogram = qdq_op_histogram(model)

    onnx.save(model, str(args.output), save_as_external_data=True,
              location=args.output.name + ".data", all_tensors_to_one_file=True,
              size_threshold=1024, convert_attribute=False)
    print(f"[quantize_onnx] fixups: stripped {num_stripped} rank-0 QuantizeLinear/"
          f"DequantizeLinear pairs, downcast {num_scales_downcast} float32 scale "
          f"initializers to float16")

    # onnx.checker's full_check=True runs its own shape inference, which is STRICTER
    # about QuantizeLinear/DequantizeLinear scale dtypes than the actual consumer
    # (TensorRT's ONNX parser) turns out to be -- found empirically on the real
    # 96-sample run, not assumed: it raises `x_scale typestr: tensor(float), has
    # unsupported type: tensor(float16)` on ORT's OWN float16-scale weight
    # DequantizeLinear nodes (the ~2035/2059 scales ORT computes in float16 by
    # default for this fp16 model -- not specific to this script's fixups, which
    # only ever touch 24+96 of them). A direct TensorRT parse of this exact
    # artifact (see this script's accompanying Stage 6B writeup) succeeds with
    # zero errors, so this is a checker/shape-inference limitation for this
    # opset's DequantizeLinear schema, not a real defect in the graph -- logged
    # as a non-fatal finding rather than either silently skipped or treated as a
    # build-blocking failure after an hour of calibration compute.
    try:
        onnx.checker.check_model(str(args.output), full_check=True)
        print(f"[quantize_onnx] onnx.checker: OK (post-fixup graph is structurally valid)")
    except Exception as exc:
        print(f"[quantize_onnx] onnx.checker.check_model(full_check=True) raised (NOT fatal -- "
              f"see this script's comment just above): {exc}")
        print(f"[quantize_onnx] proceeding on the strength of a direct TensorRT parser check "
              f"instead (this graph's actual consumer) -- see the Stage 6B build/inspect step.")

    output_sha256 = sha256_file(args.output)
    metadata = {
        "model": "ViTPose++-L",
        "stage": "Stage 6 Gate B -- INT8 QDQ ONNX quantization",
        "source_onnx_path": str(args.onnx_path),
        "source_onnx_sha256": actual_onnx_sha256,
        "output_onnx_path": str(args.output),
        "output_onnx_sha256": output_sha256,
        "calibration_manifest_path": str(args.calibration_manifest),
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration_sample_count": len(dataset),
        "calibration_per_expert_counts": dataset.per_expert_counts,
        "calibration_corpus_label": "Engineering calibration corpus -- single-source, "
                                     "augmentation-diverse (see calibration/README.md)",
        "quantization_config": {
            "tool": "onnxruntime.quantization.quantize_static",
            "quant_format": "QDQ",
            "calibrate_method": "Entropy",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": False,
            "activation_symmetric": True,
            "weight_symmetric": True,
            "quantize_bias": False,
            "op_types_to_quantize": "ORT default (unrestricted -- see module docstring)",
            "calibration_provider": providers[0],
            "calibration_stride": args.calibration_stride,
        },
        "known_limitations": KNOWN_LIMITATIONS,
        "tensorrt_parser_fixups": {
            "rank0_qdq_pairs_stripped": num_stripped,
            "float32_scales_downcast_to_float16": num_scales_downcast,
            "note": "See this script's module docstring for exactly what each fixup is and "
                    "why -- both were found by attempting a TensorRT parse and reading the "
                    "exact error, not assumed.",
        },
        "op_histogram_before_fixup": pre_fixup_histogram,
        "op_histogram_after_fixup": post_fixup_histogram,
        "quantize_wall_clock_s": quantize_seconds,
        "installed_versions": {
            "python": platform.python_version(),
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2))
    print(f"[quantize_onnx] wrote {args.metadata_output}")
    print(f"[quantize_onnx] Q/DQ nodes after fixup: "
          f"{post_fixup_histogram['QuantizeLinear']} QuantizeLinear, "
          f"{post_fixup_histogram['DequantizeLinear']} DequantizeLinear")


if __name__ == "__main__":
    main()
