#!/usr/bin/env python3
"""Stage 12: a TensorRT FP16 engine for the pipeline's YOLOv8s person detector.

    python -m conversion.build_yolo_engine

Stages 7-9 run YOLOv8s (checkpoints/yolov8s.pt) in fp32 through PyTorch, on the same GPU as
the pose model; in Stage 8's asynchronous pipeline the detect thread is the one the rest waits
for. This builds a cheaper detector without changing what it detects any more than precision
has to:

1. Export the unchanged weights to ONNX the way ultralytics' exporter prepares a dynamic export
   (fused, Detect head in export mode), with dynamic batch and spatial size (the pipeline
   letterboxes 1280x720 to 384x640, portrait to 640x384, 1180x2556 to 640x320), opset 18.
2. In fp16, except the detection head's tail: the DFL softmax/conv and the anchor arithmetic
   that turns distances into box coordinates (fp32_head_tail). In fp16 those coordinates would
   be quantized to 0.25-0.5 px at letterbox scale (up to ~1 px in a 720p frame). A plain
   `export(half=True)` does exactly that; onnxruntime's fp16 converter with a node block list
   produced an invalid graph (duplicate Cast outputs), so the mixed precision is traced from
   PyTorch instead. Input fp16 (0-1 RGB, as ultralytics feeds an fp16 model), output fp32.
3. Build a STRONGLY_TYPED TensorRT engine (TensorRT 11 takes precision from the graph's
   dtypes; see conversion/build_engine.py) with one optimization profile: batch 1-4, height and
   width 320-640 in multiples of 32, tuned for 4 x 384 x 640 (Stage 8's four 720p frames).

Letterbox, NMS and box scaling stay ultralytics' own: pipeline/stages.py swaps only the
predictor's network call for this engine. Stage 12 then gates the result on detection
agreement and pose quality (evaluation/coco_pose.py --boxes detected, pipeline/detector_check.py).

Outputs: engines/yolov8s_fp16.engine (+ the two ONNX files, all gitignored) and
results/tensorrt/yolo_engine_metadata.json (committed: hashes, profile, per-layer precision).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import onnx
import tensorrt as trt
import torch

from baseline import DEFAULT_DETECTOR, REPO_ROOT
from conversion.build_engine import sha256_file

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_DIR = REPO_ROOT / "engines"
ONNX_PATH = ENGINE_DIR / "yolov8s_fp16_dynamic.onnx"
ENGINE_PATH = ENGINE_DIR / "yolov8s_fp16.engine"
METADATA_PATH = REPO_ROOT / "results" / "tensorrt" / "yolo_engine_metadata.json"
PROFILE = {"min": (1, 3, 320, 320), "opt": (4, 3, 384, 640), "max": (4, 3, 640, 640)}
HEAD = "/model.22/"          # ultralytics' Detect module in YOLOv8s


def fp32_head_tail(model: torch.nn.Module) -> torch.nn.Module:
    """Run the Detect head's tail -- DFL and the anchor arithmetic that produces box
    coordinates -- in fp32 on an otherwise fp16 YOLOv8 model: the DFL conv goes back to fp32
    and the head's decode casts its inputs up first. The per-scale box/class convs stay fp16.
    Used for the engine's ONNX export and for the PyTorch FP16 detector variant alike."""
    from ultralytics.nn.modules.head import Detect

    heads = [m for m in model.modules() if isinstance(m, Detect)]
    if len(heads) != 1:
        raise SystemExit(f"[yolo_engine] expected one Detect head, found {len(heads)}")
    det = heads[0]
    det.dfl.float()
    inner = det._inference

    def _inference(x: dict) -> torch.Tensor:
        return inner({"boxes": x["boxes"].float(), "scores": x["scores"].float(),
                      "feats": [f.float() for f in x["feats"]]})

    det._inference = _inference
    return model


def export_onnx() -> None:
    """Export the way ultralytics' exporter prepares a dynamic ONNX export (fused, Detect in
    export mode, C2f split forward), but with an fp16 body and an fp32 head tail."""
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Detect

    model = YOLO(str(DEFAULT_DETECTOR)).model.to("cuda")
    for prm in model.parameters():
        prm.requires_grad = False
    model = model.eval().float().fuse(imgsz=(640, 640))
    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic, m.export, m.format, m.shape = True, True, "onnx", None
            m.max_det = min(300, sum(int(640 / s) ** 2 for s in model.stride.tolist()))
            m.agnostic_nms, m.xyxy = False, False
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    model = fp32_head_tail(model.half())
    im = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
    for _ in range(2):
        model(im)
    torch.onnx.export(model, im, str(ONNX_PATH), opset_version=18, input_names=["images"], output_names=["output0"],
                      dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "output0": {0: "batch", 2: "anchors"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(ONNX_PATH))


def build(opt_level: int) -> dict:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(ONNX_PATH.read_bytes(), path=str(ONNX_PATH)) or parser.num_errors:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"[yolo_engine] ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.builder_optimization_level = opt_level
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, PROFILE["min"], PROFILE["opt"], PROFILE["max"])
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("[yolo_engine] build failed")
    ENGINE_PATH.write_bytes(bytes(serialized))
    build_s = time.perf_counter() - t0

    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized)
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    hist: dict[str, int] = {}
    for layer in info.get("Layers", []):
        dts = sorted({o.get("Datatype", "UNKNOWN") for o in layer.get("Outputs", [])}) or ["NO_OUTPUT"]
        hist["+".join(dts)] = hist.get("+".join(dts), 0) + 1
    return {"build_wall_clock_s": build_s, "engine_file_size_mb": serialized.nbytes / 2**20,
            "activation_workspace_mb_at_max_shape": engine.device_memory_size_v2 / 2**20,
            "io": {engine.get_tensor_name(i): str(engine.get_tensor_dtype(engine.get_tensor_name(i)))
                   for i in range(engine.num_io_tensors)},
            "layer_output_dtype_histogram": hist}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opt-level", type=int, default=3)
    args = p.parse_args()
    ENGINE_DIR.mkdir(exist_ok=True)
    export_onnx()
    print(f"[yolo_engine] fp16 ONNX, fp32 head tail -> {ONNX_PATH}")
    built = build(args.opt_level)
    meta = {
        "stage": 12,
        "model": "YOLOv8s (ultralytics), person detector of Stages 0-9",
        "source_checkpoint_sha256": sha256_file(DEFAULT_DETECTOR),
        "onnx_sha256": sha256_file(ONNX_PATH),
        "engine_path": str(ENGINE_PATH.relative_to(REPO_ROOT)),
        "engine_sha256": sha256_file(ENGINE_PATH),
        "precision": "FP16 (strongly typed), detection-head tail (DFL + anchor arithmetic) fp32; fp16 input, fp32 output",
        "profile": {k: list(v) for k, v in PROFILE.items()},
        "builder_optimization_level": args.opt_level,
        **built,
        "env": {"python": platform.python_version(), "tensorrt_version": trt.__version__, "torch": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[yolo_engine] wrote {ENGINE_PATH} ({built['engine_file_size_mb']:.1f} MB, {built['build_wall_clock_s']:.0f}s) "
          f"and {METADATA_PATH}")


if __name__ == "__main__":
    main()
