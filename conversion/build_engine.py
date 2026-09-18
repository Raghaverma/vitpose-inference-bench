#!/usr/bin/env python3
"""Stage 3: build a TensorRT engine from the Stage 2 ONNX export.

    python -m conversion.build_engine

Builds ONLY from the exact, sha256-pinned faithful ONNX export
(results/onnx/vitpose_plus_l.onnx) -- never from an ORT-optimized graph --
so any TensorRT speedup can't be silently crediting graph rewrites ONNX
Runtime already applied.

A version note found empirically, not assumed from older tutorials:
TensorRT 11.3 has NO `BuilderFlag.FP16` / `BuilderFlag.INT8` at all -- the
classic `config.set_flag(trt.BuilderFlag.FP16)` API from TensorRT 8.x/9.x
tutorials doesn't exist in this version. Precision is now controlled by
building a STRONGLY_TYPED network, which takes the dtypes the ONNX graph
already declares at face value. Since our ONNX was exported from an
explicitly fp16-cast model (Stage 0/2's `torch_dtype=torch.float16`), a
strongly-typed build of it IS the FP16 engine -- there is no separate
"allow fp16" permission step to set. This also changes what "FP16 precision
honesty" means here versus older TensorRT: there's no builder-chosen
per-layer fp16-vs-fp32 selection to audit the way BuilderFlag.FP16 used to
produce -- precision comes from the graph's own declared dtypes. We still
dump the built engine's per-layer information (via EngineInspector) into
the metadata file, because "the graph says fp16" and "the engine actually
executed every layer in fp16" are not automatically the same claim, and
this repo doesn't assume they are without checking.

The .engine file itself is NEVER committed (see .gitignore's `*.engine`
rule) -- it's a GPU/driver/TensorRT-version-locked build artifact. What's
committed is results/tensorrt/engine_metadata*.json: the build manifest
(exact onnx_sha256 it was built from, engine_sha256 of the output, full
environment fingerprint, per-layer precision audit) that lets
backends/tensorrt.py refuse to load a mismatched engine rather than fail
with an opaque deserialization error.

Stage 4 update: `--batch-size B` builds engines/vitpose_b{B}_fp16.engine
from the matching per-batch static ONNX export (results/onnx/vitpose_plus_l
[_b{B}].onnx, built by `conversion/export_onnx.py --batch-size B` from
GENUINELY DISTINCT crops, not repeated copies -- see that script and
golden/build_distinct_batch.py). Each batch size is an independently-tuned
TensorRT build (its own tactic search against the physical GPU) -- these
are 5 separate artifacts, not one engine reused 5 ways, and each gets its
own manifest reflecting that. VRAM is reported as three separate figures,
not one: `engine_file_size_mb` (weights, roughly batch-invariant),
`activation_workspace_mb` (via `engine.device_memory_size_v2` -- verified
against this exact installed TensorRT version, since `IExecutionContext.
device_memory_size` does NOT exist here, another API-naming trap like
`BuilderFlag.FP16`), and `build_phase_vram_delta_mb` (resident memory
consumed during the builder's own autotuning, sampled before/after --
distinct from what the engine needs at inference time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import tensorrt as trt
import torch

from baseline import REPO_ROOT

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-path", type=Path, default=None,
                    help="Defaults to results/onnx/vitpose_plus_l[_b{B}].onnx for the given "
                         "--batch-size.")
    p.add_argument("--onnx-metadata", type=Path, default=None,
                    help="Defaults to results/onnx/export_metadata[_b{B}].json -- its "
                         "onnx_sha256 is verified against the actual ONNX file before "
                         "building, so a stale or hand-edited ONNX can't silently get built "
                         "into an engine.")
    p.add_argument("--engine-path", type=Path, default=None,
                    help="Defaults to engines/vitpose_b{B}_fp16.engine.")
    p.add_argument("--output", type=Path, default=None,
                    help="Defaults to results/tensorrt/engine_metadata[_b{B}].json.")
    p.add_argument("--mode", choices=["smoke", "tuned"], default="tuned",
                    help="smoke = minimal tactic search, fast, NOT benchmarkable (proves the "
                         "ONNX->engine path still works). tuned = full autotuning, the only "
                         "mode allowed to produce a results/tensorrt/fp16*.json benchmark.")
    p.add_argument("--workspace-mb", type=int, default=4096)
    args = p.parse_args()

    B = args.batch_size
    suffix = "" if B == 1 else f"_b{B}"
    if args.onnx_path is None:
        args.onnx_path = REPO_ROOT / "results" / "onnx" / f"vitpose_plus_l{suffix}.onnx"
    if args.onnx_metadata is None:
        args.onnx_metadata = REPO_ROOT / "results" / "onnx" / f"export_metadata{suffix}.json"
    if args.engine_path is None:
        args.engine_path = REPO_ROOT / "engines" / f"vitpose_b{B}_fp16.engine"
    if args.output is None:
        args.output = REPO_ROOT / "results" / "tensorrt" / f"engine_metadata{suffix}.json"
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(args: argparse.Namespace) -> dict:
    onnx_metadata = json.loads(args.onnx_metadata.read_text())
    actual_onnx_sha256 = sha256_file(args.onnx_path)
    if actual_onnx_sha256 != onnx_metadata["onnx_sha256"]:
        raise SystemExit(
            f"[build_engine] REFUSING to build: {args.onnx_path} has sha256 "
            f"{actual_onnx_sha256}, but {args.onnx_metadata} (Stage 2's own export record) "
            f"says it should be {onnx_metadata['onnx_sha256']}. This ONNX file has changed "
            f"since Gates A-D validated it -- re-run the Stage 2 gates before building a "
            f"TensorRT engine from it.")
    if onnx_metadata["optimize"] is not False or onnx_metadata["batch_axis"] != "static":
        raise SystemExit(
            f"[build_engine] REFUSING to build: expected the faithful "
            f"(optimize=False, static batch) export, got optimize={onnx_metadata['optimize']} "
            f"batch_axis={onnx_metadata['batch_axis']}.")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_engine] parsing {args.onnx_path} ...")
    with open(args.onnx_path, "rb") as f:
        parse_ok = parser.parse(f.read(), path=str(args.onnx_path))
    if not parse_ok:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"[build_engine] ONNX parse FAILED (strict, no silent op fallback):\n{errors}")
    if parser.num_errors > 0:
        # parse() can return True while still recording non-fatal warnings/errors --
        # treat any of them as a hard failure rather than a note, per the "strict
        # parser mode" rule: a silently-substituted or dropped op must never pass.
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"[build_engine] ONNX parser reported {parser.num_errors} "
                          f"error(s)/warning(s), refusing to treat as a clean parse:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * (1 << 20))
    # Default profiling verbosity only retains layer NAMES in the built engine --
    # the inspector can't report per-layer precision without DETAILED, found
    # empirically (LAYER_NAMES_ONLY gave back a flat list of name strings, no
    # precision field at all).
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.mode == "smoke":
        config.builder_optimization_level = 0
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    else:
        config.builder_optimization_level = 3  # default max search

    # Build-phase VRAM is a DIFFERENT physical event from run-time VRAM: the
    # builder's own tactic-timing loop allocates/frees scratch memory while
    # racing candidate kernels, which can transiently peak higher than the
    # final serialized engine ever needs at inference. Sampled before/after
    # (not continuously polled -- that needs a background thread and isn't
    # worth the complexity for a before/after delta at this stage).
    torch.cuda.synchronize()
    vram_free_before, vram_total = torch.cuda.mem_get_info()
    print(f"[build_engine] building ({args.mode} mode, this can take a while)...")
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - t0
    torch.cuda.synchronize()
    vram_free_after_build, _ = torch.cuda.mem_get_info()
    build_phase_vram_delta_mb = (vram_free_before - vram_free_after_build) / 2**20
    if serialized_engine is None:
        raise SystemExit("[build_engine] build_serialized_network returned None -- build failed.")

    args.engine_path.parent.mkdir(parents=True, exist_ok=True)
    args.engine_path.write_bytes(bytes(serialized_engine))
    print(f"[build_engine] wrote {args.engine_path} ({serialized_engine.nbytes / 2**20:.1f} MB, "
          f"{build_seconds:.1f}s)")

    # Per-layer precision audit -- the graph declares fp16 (see module docstring),
    # but "declared" and "actually executed as" aren't automatically the same
    # claim without checking. Read it back off the built engine, not assumed.
    #
    # Found empirically (this TensorRT version's engine-inspector JSON has no
    # top-level "Precision" field per layer): precision lives on each tensor as
    # a "Datatype" field ("Half" / "Float" / ...) inside a layer's "Outputs"
    # (its actual computed result). A layer is classified by the set of its
    # output datatypes -- "Half" throughout counts as fp16; anything touching
    # "Float" is flagged, since that's a real fp32 computation, not a labeling
    # artifact (e.g. a gemm's per-channel alpha/beta scale constants commonly
    # stay fp32 even when the main activation path is fp16 -- that's a
    # legitimate mixed-precision detail worth surfacing, not an error).
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    # VRAM is not one number -- these three are physically distinct and don't
    # collapse into each other: weights (~batch-invariant), the activation
    # workspace TensorRT itself declares it needs for THIS batch size, and
    # the build-phase transient sampled above. `device_memory_size_v2` is the
    # real API name for this TensorRT version -- `IExecutionContext.
    # device_memory_size` does not exist here, verified empirically rather
    # than assumed from older-TensorRT documentation (same class of trap as
    # `BuilderFlag.FP16` in this script's module docstring).
    engine_file_size_mb = serialized_engine.nbytes / 2**20
    activation_workspace_mb = engine.device_memory_size_v2 / 2**20
    inspector = engine.create_engine_inspector()
    layer_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = layer_info.get("Layers", [])
    precision_counts: dict[str, int] = {}
    for layer in layers:
        output_dtypes = {out.get("Datatype", "UNKNOWN") for out in layer.get("Outputs", [])}
        if not output_dtypes:
            label = "NO_OUTPUT_TENSOR"
        elif output_dtypes == {"Half"}:
            label = "Half"
        else:
            label = "+".join(sorted(output_dtypes))
        precision_counts[label] = precision_counts.get(label, 0) + 1
    total_layers = sum(precision_counts.values())
    fp16_layers = precision_counts.get("Half", 0)

    gpu_name = torch.cuda.get_device_name(0)
    metadata = {
        "model": "ViTPose++-L",
        "source": str(args.onnx_path.name),
        "onnx_sha256": actual_onnx_sha256,
        "engine_path": str(args.engine_path),
        "engine_sha256": sha256_file(args.engine_path),
        "precision_requested": "FP16 (strongly-typed network, graph-declared dtypes)",
        "mode": args.mode,
        "benchmarkable": args.mode == "tuned",
        "batch": onnx_metadata["static_batch_size"],
        "input_shape": [onnx_metadata["static_batch_size"], 3] + onnx_metadata["static_spatial_shape"],
        "opset": onnx_metadata["opset_version"],
        "build_wall_clock_s": build_seconds,
        "builder_optimization_level": config.builder_optimization_level,
        "vram": {
            "engine_file_size_mb": engine_file_size_mb,
            "activation_workspace_mb": activation_workspace_mb,
            "build_phase_vram_delta_mb": build_phase_vram_delta_mb,
            "note": "three physically distinct figures -- weights (~batch-invariant), "
                    "the activation workspace this batch size needs, and build-time-only "
                    "transient scratch. None of these three is the same thing as "
                    "backends/tensorrt.py's steady-state 'device_vram_mb' inference "
                    "measurement, which also includes the CUDA context's own fixed tax.",
        },
        "precision_audit": {
            "total_layers": total_layers,
            "layer_precision_histogram": precision_counts,
            "fp16_layers": fp16_layers,
            "fraction_fp16": fp16_layers / total_layers if total_layers else None,
        },
        "env": {
            "python": platform.python_version(),
            "tensorrt_version": trt.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": gpu_name,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2))
    print(f"[build_engine] wrote {args.output}")
    print(f"[build_engine] precision audit: {fp16_layers}/{total_layers} layers FP16 "
          f"({metadata['precision_audit']['fraction_fp16']:.1%})"
          if total_layers else "[build_engine] precision audit: no layer info available")
    print(f"[build_engine] VRAM: engine {engine_file_size_mb:.1f}MB  "
          f"activation workspace {activation_workspace_mb:.1f}MB  "
          f"build-phase delta {build_phase_vram_delta_mb:.1f}MB")
    return metadata


if __name__ == "__main__":
    build(parse_args())
