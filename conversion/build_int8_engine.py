#!/usr/bin/env python3
"""Stage 6B: build a static batch-1 INT8 TensorRT engine.

    python -m conversion.build_int8_engine

    python -m conversion.build_int8_engine --batch-size 16

Builds ONLY from conversion/quantize_onnx.py's exact, sha256-pinned QDQ
output (results/onnx/vitpose_plus_l_int8_qdq[_b{B}].onnx) -- Q/DQ on MatMul
inputs only, activation scales calibrated on the real-footage corpus
(calibration/real_corpus.py), itself built from the same faithful FP16
export conversion/build_engine.py uses, per that script's own sha256 gate. This
script does no calibration of its own and imports no calibrator class: as
covered at length in conversion/quantize_onnx.py's module docstring,
TensorRT 11.3.0.99's Python API has no `BuilderFlag.INT8` and no
`IInt8Calibrator*` classes at all -- calibration already happened, on the
ONNX graph, in Gate B. This script's only job is what conversion/
build_engine.py already does for FP16: parse a graph whose tensor dtypes are
already fully declared (here: Half everywhere, except where quantize_onnx.py
inserted QuantizeLinear/DequantizeLinear pairs, which declare Int8) into a
STRONGLY_TYPED network, and let TensorRT take those declarations at face
value. There is no `--mode` precision-selection knob here for the same
reason FP16 doesn't have one on a strongly-typed build: the graph's Q/DQ
placement, not a builder flag, is what makes this an INT8 build.

Before building, this script verifies TWO independent chains of trust, not
one:
  1. the QDQ onnx's sha256 against Gate B's own record of what it wrote
     (results/onnx/int8_quantization_metadata[_b{B}].json).
  2. calibration/real_manifest.json's sha256 against Gate B's own record of
     what it calibrated against -- catching the case where the calibration
     corpus was regenerated (same filename, different content) AFTER Gate B
     ran, which check (1) alone can't see (the QDQ onnx's sha256 wouldn't
     change retroactively just because its calibration inputs did).
A mismatch in either is a hard refusal, same as conversion/build_engine.py's
onnx_sha256 gate -- never a silent rebuild-from-whatever's-on-disk.

The .engine file is NEVER committed (see .gitignore's `*.engine` rule and
conversion/build_engine.py's docstring) -- what's committed is
results/tensorrt/int8_engine_metadata.json: onnx/calibration provenance,
full environment fingerprint (TensorRT/CUDA/driver/GPU/torch), builder
configuration, and a per-layer EngineInspector precision audit.

Per-layer precision audit: THREE execution types, not build_engine.py's
binary Half/Float split
--------------------------------------------------------------------------
"I asked TensorRT for INT8, therefore everything is INT8" is exactly the
unchecked assumption this repo's methodology exists to catch (see Stage 3's
own note that a strongly-typed FP16 build still needed its per-layer
datatypes read back, not assumed, because 1/180 layers there turned out to
stay Float). Read back off the built engine via EngineInspector (DETAILED
profiling verbosity -- LAYER_NAMES_ONLY strips the Datatype field, found
empirically for Stage 3), classifying each layer by the set of its output
tensors' declared Datatype, exactly like build_engine.py's precision_audit
does -- except a real third (and fourth: mixed) category is now possible.
A layer whose outputs are ALL "Int8" is a true INT8-executing layer; ALL
"Half" is FP16; ALL "Float" is FP32; anything else (e.g. a quantize/
dequantize boundary layer straddling Int8 and Half) is its own labeled
bucket, not silently folded into one side -- collapsing "Half+Int8" into
either "FP16" or "INT8" would misrepresent what that layer actually does.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import onnx
import tensorrt as trt
import torch

from baseline import REPO_ROOT
from conversion.build_engine import sha256_file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-path", type=Path, default=None,
                    help="Defaults to results/onnx/vitpose_plus_l_int8_qdq[_b{B}].onnx.")
    p.add_argument("--onnx-metadata", type=Path, default=None,
                    help="Gate B's own record (conversion/quantize_onnx.py), defaults to "
                         "results/onnx/int8_quantization_metadata[_b{B}].json. Verified against "
                         "both --onnx-path's actual sha256 AND the live calibration manifest's "
                         "sha256 before building -- see module docstring.")
    p.add_argument("--calibration-manifest", type=Path,
                    default=REPO_ROOT / "calibration" / "real_manifest.json")
    p.add_argument("--engine-path", type=Path, default=None,
                    help="Defaults to engines/vitpose_b{B}_int8.engine.")
    p.add_argument("--output", type=Path, default=None,
                    help="Defaults to results/tensorrt/int8_engine_metadata[_b{B}].json.")
    p.add_argument("--mode", choices=["smoke", "tuned"], default="tuned",
                    help="smoke = minimal tactic search, fast, NOT benchmarkable. tuned = full "
                         "autotuning, the only mode allowed to produce a results/tensorrt/int8*.json "
                         "benchmark later (Stage 6E). Same convention as conversion/build_engine.py.")
    p.add_argument("--workspace-mb", type=int, default=4096)
    args = p.parse_args()
    sfx = "" if args.batch_size == 1 else f"_b{args.batch_size}"
    args.onnx_path = args.onnx_path or REPO_ROOT / "results" / "onnx" / f"vitpose_plus_l_int8_qdq{sfx}.onnx"
    args.onnx_metadata = (args.onnx_metadata
                          or REPO_ROOT / "results" / "onnx" / f"int8_quantization_metadata{sfx}.json")
    args.engine_path = args.engine_path or REPO_ROOT / "engines" / f"vitpose_b{args.batch_size}_int8.engine"
    args.output = args.output or REPO_ROOT / "results" / "tensorrt" / f"int8_engine_metadata{sfx}.json"
    return args


def gpu_driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip().splitlines()[0]
    except Exception as exc:
        print(f"[build_int8_engine] WARNING: could not read GPU driver version via nvidia-smi: {exc}")
        return None


def classify_layer_precision(layers: list[dict]) -> dict[str, int]:
    """Same technique conversion/build_engine.py's precision_audit uses
    (classify by the set of a layer's OUTPUT tensor Datatypes, since this
    TensorRT version's EngineInspector JSON has no top-level per-layer
    'Precision' field) -- extended to however many distinct dtype
    combinations actually show up, not collapsed to a binary split."""
    counts: dict[str, int] = {}
    for layer in layers:
        output_dtypes = {out.get("Datatype", "UNKNOWN") for out in layer.get("Outputs", [])}
        label = "NO_OUTPUT_TENSOR" if not output_dtypes else "+".join(sorted(output_dtypes))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _dtypes(tensors: list[dict]) -> str:
    return "+".join(sorted({t.get("Datatype", "UNKNOWN") for t in tensors})) or "none"


def classify_layer_io(layers: list[dict]) -> tuple[dict[str, int], int]:
    """What the output-dtype histogram above can't show: an INT8 GEMM reads Int8 and writes
    its dequantized result as Float or Half -- so by output dtype alone it lands in the FP32
    or FP16 bucket (168 'gemm: Int8 -> Float' layers on the batch-1 engine; from batch 4 up
    TensorRT fuses the same GEMMs with their epilogue into 'fusion: Int8 -> Half' layers).
    Returns the '<LayerType>: <input dtypes> -> <output dtypes>' histogram and the number of
    layers whose inputs are all Int8, i.e. that execute on INT8 data."""
    sig: dict[str, int] = {}
    int8_input = 0
    for layer in layers:
        ins, outs = _dtypes(layer.get("Inputs", [])), _dtypes(layer.get("Outputs", []))
        key = f"{layer.get('LayerType', '?')}: {ins} -> {outs}"
        sig[key] = sig.get(key, 0) + 1
        int8_input += ins == "Int8"
    return dict(sorted(sig.items(), key=lambda kv: -kv[1])), int8_input


def build(args: argparse.Namespace) -> dict:
    onnx_metadata = json.loads(args.onnx_metadata.read_text())
    actual_onnx_sha256 = sha256_file(args.onnx_path)
    if actual_onnx_sha256 != onnx_metadata["output_onnx_sha256"]:
        raise SystemExit(
            f"[build_int8_engine] REFUSING to build: {args.onnx_path} has sha256 "
            f"{actual_onnx_sha256}, but {args.onnx_metadata} (Gate B's own record) says it "
            f"should be {onnx_metadata['output_onnx_sha256']}. Re-run "
            f"conversion/quantize_onnx.py before building.")

    actual_manifest_sha256 = sha256_file(args.calibration_manifest)
    if actual_manifest_sha256 != onnx_metadata["calibration_manifest_sha256"]:
        raise SystemExit(
            f"[build_int8_engine] REFUSING to build: {args.calibration_manifest} has sha256 "
            f"{actual_manifest_sha256}, but {args.onnx_metadata} says Gate B calibrated "
            f"against {onnx_metadata['calibration_manifest_sha256']}. The calibration corpus "
            f"changed since the QDQ graph was quantized -- re-run conversion/quantize_onnx.py "
            f"before building an engine from a stale calibration.")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    onnx_opset = {imp.domain or "ai.onnx": imp.version
                  for imp in onnx.load(str(args.onnx_path), load_external_data=False).opset_import}

    print(f"[build_int8_engine] parsing {args.onnx_path} ...")
    with open(args.onnx_path, "rb") as f:
        parse_ok = parser.parse(f.read(), path=str(args.onnx_path))
    if not parse_ok or parser.num_errors > 0:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"[build_int8_engine] ONNX parse FAILED (strict, no silent op "
                          f"fallback):\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * (1 << 20))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.mode == "smoke":
        config.builder_optimization_level = 0
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    else:
        config.builder_optimization_level = 3

    torch.cuda.synchronize()
    vram_free_before, vram_total = torch.cuda.mem_get_info()
    print(f"[build_int8_engine] building ({args.mode} mode, this can take a while)...")
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - t0
    torch.cuda.synchronize()
    vram_free_after_build, _ = torch.cuda.mem_get_info()
    build_phase_vram_delta_mb = (vram_free_before - vram_free_after_build) / 2**20
    if serialized_engine is None:
        raise SystemExit("[build_int8_engine] build_serialized_network returned None -- build failed.")

    args.engine_path.parent.mkdir(parents=True, exist_ok=True)
    args.engine_path.write_bytes(bytes(serialized_engine))
    print(f"[build_int8_engine] wrote {args.engine_path} ({serialized_engine.nbytes / 2**20:.1f} MB, "
          f"{build_seconds:.1f}s)")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    engine_file_size_mb = serialized_engine.nbytes / 2**20
    activation_workspace_mb = engine.device_memory_size_v2 / 2**20
    inspector = engine.create_engine_inspector()
    layer_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = layer_info.get("Layers", [])
    precision_counts = classify_layer_precision(layers)
    total_layers = sum(precision_counts.values())
    int8_layers = precision_counts.get("Int8", 0)
    fp16_layers = precision_counts.get("Half", 0)
    fp32_layers = precision_counts.get("Float", 0)
    mixed_layers = total_layers - int8_layers - fp16_layers - fp32_layers
    io_signatures, int8_input_layers = classify_layer_io(layers)

    gpu_name = torch.cuda.get_device_name(0)
    metadata = {
        "model": "ViTPose++-L",
        "stage": "Stage 6B -- INT8 TensorRT engine build",
        "source": str(args.onnx_path.name),
        "onnx_sha256": actual_onnx_sha256,
        "calibration_manifest_sha256": actual_manifest_sha256,
        "calibration_num_crops": onnx_metadata["calibration_num_crops"],
        "dataset_index_validated": onnx_metadata["dataset_index_validated"],
        "quantization_recipe": onnx_metadata["recipe"],
        "scales_sha256": onnx_metadata["scales_sha256"],
        "engine_path": str(args.engine_path),
        "engine_sha256": sha256_file(args.engine_path),
        "precision_requested": "INT8 (strongly-typed network, explicit Q/DQ graph from "
                                "conversion/quantize_onnx.py -- see that script and this one's "
                                "module docstrings for why there is no BuilderFlag.INT8 step here)",
        "mode": args.mode,
        "benchmarkable": args.mode == "tuned",
        "batch": args.batch_size,
        "opset": onnx_opset,
        "build_wall_clock_s": build_seconds,
        "builder_optimization_level": config.builder_optimization_level,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "vram": {
            "engine_file_size_mb": engine_file_size_mb,
            "activation_workspace_mb": activation_workspace_mb,
            "build_phase_vram_delta_mb": build_phase_vram_delta_mb,
            "note": "same three physically distinct figures as conversion/build_engine.py's "
                    "FP16 metadata -- see that script's docstring.",
        },
        "precision_audit": {
            "total_layers": total_layers,
            "layer_precision_histogram": precision_counts,
            "int8_layers": int8_layers,
            "fp16_layers": fp16_layers,
            "fp32_layers": fp32_layers,
            "mixed_precision_layers": mixed_layers,
            "fraction_int8": int8_layers / total_layers if total_layers else None,
            "fraction_fp16": fp16_layers / total_layers if total_layers else None,
            "fraction_fp32": fp32_layers / total_layers if total_layers else None,
            "fraction_mixed": mixed_layers / total_layers if total_layers else None,
            "note": "The *_layers counts above classify by each layer's OUTPUT tensor Datatype(s), "
                    "same technique as conversion/build_engine.py's FP16 precision_audit -- 'mixed' "
                    "layers (e.g. Half+Int8) straddle a quantize/dequantize boundary. By that measure "
                    "an INT8 GEMM that writes Float counts as FP32; int8_input_layers / "
                    "layer_io_signature_histogram below count what actually reads INT8.",
            "int8_input_layers": int8_input_layers,
            "layer_io_signature_histogram": io_signatures,
        },
        "env": {
            "python": platform.python_version(),
            "tensorrt_version": trt.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu_name,
            "gpu_driver_version": gpu_driver_version(),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2))
    print(f"[build_int8_engine] wrote {args.output}")
    if total_layers:
        print(f"[build_int8_engine] precision audit: {int8_layers}/{total_layers} INT8 "
              f"({int8_layers/total_layers:.1%})  {fp16_layers}/{total_layers} FP16 "
              f"({fp16_layers/total_layers:.1%})  {fp32_layers}/{total_layers} FP32 "
              f"({fp32_layers/total_layers:.1%})  {mixed_layers}/{total_layers} mixed "
              f"({mixed_layers/total_layers:.1%})")
        print(f"[build_int8_engine] full histogram: {precision_counts}")
        print(f"[build_int8_engine] layers reading INT8: {int8_input_layers}/{total_layers}")
    print(f"[build_int8_engine] VRAM: engine {engine_file_size_mb:.1f}MB  "
          f"activation workspace {activation_workspace_mb:.1f}MB  "
          f"build-phase delta {build_phase_vram_delta_mb:.1f}MB")
    return metadata


if __name__ == "__main__":
    build(parse_args())
