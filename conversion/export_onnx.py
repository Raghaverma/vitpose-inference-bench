#!/usr/bin/env python3
"""Stage 2 Gate A: export ViTPose++-L to ONNX.

    python -m conversion.export_onnx

Exports the model ALONE -- not YOLO, not crop/preprocessing -- using the
exact pixel_values tensor Stage 1's golden reference was built from
(golden/person_crop.npy). Faithfulness over performance for this first
artifact:

  * static (256, 192) spatial shape -- pose preprocessing already produces a
    fixed model input, there's no reason to make height/width dynamic too
  * dynamic batch axis only, via an explicit torch.export.Dim shared between
    pixel_values and dataset_index (not dynamic=True / Dim.AUTO on every
    axis of every input)
  * a pinned opset
  * optimize=False -- no constant folding / dead-code elimination on this
    artifact, so a later numerical or performance regression can't be
    blamed on graph optimization and export in the same breath

A note on independence: transformers.exporters.OnnxExporter is documented
and implemented as a wrapper around `torch.export` -> `torch.onnx.export`
(see its docstring / conversion/inspect_onnx.py's dump of its source) --
it is not a second, independently-implemented export path. A hand-rolled
`torch.onnx.export` call would share the same underlying dynamo/onnxscript
lowering and so would agree with OnnxExporter's output even if that shared
lowering were wrong -- agreement would be false confidence, not a real
witness. Gate C's comparison against the golden PyTorch EAGER-mode output
(tests/test_onnx_equivalence.py) is the actual independent check here,
because eager PyTorch never goes through torch.export/onnxscript at all.

A note on the dynamic batch axis (found empirically, not assumed): an
explicit `torch.export.Dim("batch", min=1, max=...)` on this model raises
`ConstraintViolationError: ... your code specialized it to be a constant (1)`.
Retracing with `Dim.AUTO` "succeeds" but silently falls back to a fully
static graph -- the traced FX graph has literal `[1, ...]` shapes baked into
several `view`/`reshape` calls inside the backbone's windowed self-attention,
not the symbolic batch size. This is a real constraint of the current HF
`VitPoseBackbone` implementation, not a bug in this script: something in the
attention/window-partition path loses the symbolic batch dim during
`torch.export`'s guard solving. Making batch genuinely dynamic would need
patching `modeling_vitpose_backbone.py` (out of scope for "faithful first
export" -- that's exactly the kind of graph surgery Stage 2 defers). This
script therefore exports a STATIC batch=1 graph and records the dynamic
attempt's failure in export_metadata.json so this isn't silently lost.
Batch-size sweeps against ONNX Runtime (Gate D) are deferred until either
the model is patched or a separate static export per batch size is built.

Stage 4 update: `--batch-size B` (B>1) builds exactly that "separate static
export per batch size" -- using `golden/distinct_crops.npy[:B]` (GENUINELY
DISTINCT crops with a mixed dataset_index per slot, built by
`golden/build_distinct_batch.py`), never `.repeat()`, so the resulting ONNX
graph gets a real chance to reveal a cross-batch-element bug during Gate C
instead of one that's structurally invisible under repeated-copy batches.
For B>1 the dynamic-batch attempt is skipped outright (not re-attempted and
allowed to fail again) -- it's a known, already-documented failure for this
model, and there's nothing new to learn from re-hitting it 4 more times.
--batch-size 1 (the default) is UNCHANGED from the original Gate A: it still
attempts the dynamic export first and writes to the same default paths, so
Stage 2's already-validated batch=1 artifact is never touched by this update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxscript
import torch
import transformers
from transformers.exporters import OnnxConfig, OnnxExporter

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint

OPSET_VERSION = 18
MAX_BATCH = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "golden")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-index", type=int, default=0,
                    help="Only used for --batch-size 1 (golden/person_crop.npy has no "
                         "per-slot dataset_index of its own). Ignored for --batch-size > 1, "
                         "which uses golden/distinct_batch_env.json's per-slot values instead.")
    p.add_argument("--batch-size", type=int, default=1,
                    help="1 (default) = original Gate A behavior, unchanged. >1 = a "
                         "genuinely-distinct-crops static export for that batch size "
                         "(golden/build_distinct_batch.py must have been run first).")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "onnx")
    p.add_argument("--onnx-path", type=Path, default=None,
                    help="Defaults to results/onnx/vitpose_plus_l.onnx for batch=1, "
                         "results/onnx/vitpose_plus_l_b{B}.onnx otherwise.")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    device = args.device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    B = args.batch_size

    onnx_path = args.onnx_path or (
        REPO_ROOT / "results" / "onnx" / "vitpose_plus_l.onnx" if B == 1
        else REPO_ROOT / "results" / "onnx" / f"vitpose_plus_l_b{B}.onnx")

    if B == 1:
        golden_env = json.loads((args.golden_dir / "env.json").read_text())
        person_crop = np.load(args.golden_dir / "person_crop.npy")  # (1, 3, 256, 192) fp16
        pixel_values = torch.from_numpy(person_crop).to(device=device, dtype=dtype)
        dataset_index = torch.full((1,), args.dataset_index, dtype=torch.long, device=device)
        per_slot_dataset_index = [args.dataset_index]
    else:
        golden_env = json.loads((args.golden_dir / "distinct_batch_env.json").read_text())
        distinct_crops = np.load(args.golden_dir / "distinct_crops.npy")  # (16, 3, 256, 192)
        if B > distinct_crops.shape[0]:
            raise SystemExit(f"[export_onnx] --batch-size {B} exceeds the "
                              f"{distinct_crops.shape[0]} slots golden/build_distinct_batch.py built.")
        per_slot_dataset_index = [s["dataset_index"] for s in golden_env["slots"][:B]]
        pixel_values = torch.from_numpy(distinct_crops[:B]).to(device=device, dtype=dtype)
        dataset_index = torch.tensor(per_slot_dataset_index, dtype=torch.long, device=device)

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[export_onnx] loading ViTPose++-L from {checkpoint} ({device}), batch={B}")
    _processor, model = load_pose_model(checkpoint, device, dtype)
    model.eval()

    sample_inputs = {"pixel_values": pixel_values, "dataset_index": dataset_index}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exporter = OnnxExporter()
    dynamic_export_error = None

    if B == 1:
        batch_axis = "dynamic"
        batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH)
        dynamic_shapes = {"pixel_values": {0: batch_dim}, "dataset_index": {0: batch_dim}}
        print(f"[export_onnx] attempting export with a dynamic batch axis "
              f"(opset={OPSET_VERSION}, optimize=False)...")
        try:
            config = OnnxConfig(
                output_path=str(onnx_path),
                dynamic_shapes=dynamic_shapes,
                opset_version=OPSET_VERSION,
                optimize=False,
                external_data=True,  # fp16 checkpoint is ~1.7GB, close enough to the 2GB
                                      # protobuf limit that embedding it isn't worth the risk
            )
            onnx_program = exporter.export(model, sample_inputs, config=config)
        except torch._dynamo.exc.UserError as exc:
            dynamic_export_error = str(exc).splitlines()[0]
            print(f"[export_onnx] dynamic batch export FAILED: {dynamic_export_error}")
            print("[export_onnx] falling back to a static batch=1 export -- see this "
                  "script's module docstring for the (empirically-found) root cause.")
            batch_axis = "static"
            config = OnnxConfig(output_path=str(onnx_path), dynamic_shapes=None,
                                 opset_version=OPSET_VERSION, optimize=False, external_data=True)
            onnx_program = exporter.export(model, sample_inputs, config=config)
    else:
        # B>1: don't re-attempt the dynamic export -- it's a known, already-
        # documented failure for this model (see module docstring). Go
        # straight to a static export at this specific batch size.
        batch_axis = "static"
        dynamic_export_error = "skipped for batch>1 -- already documented as failing (see batch=1 export)"
        print(f"[export_onnx] static export at batch={B} (opset={OPSET_VERSION}, optimize=False)...")
        config = OnnxConfig(output_path=str(onnx_path), dynamic_shapes=None,
                             opset_version=OPSET_VERSION, optimize=False, external_data=True)
        onnx_program = exporter.export(model, sample_inputs, config=config)
    print(f"[export_onnx] wrote {onnx_path} (batch_axis={batch_axis})")

    # Immediate in-memory sanity check -- NOT Gate C (that's the rigorous,
    # per-batch-size, golden-reference comparison in
    # tests/test_onnx_equivalence.py). This just catches an export that's
    # obviously broken before we bother writing metadata and moving to Gate B.
    with torch.no_grad():
        pytorch_out = model(pixel_values=pixel_values, dataset_index=dataset_index)
    # ONNXProgram.__call__ returns Sequence[torch.Tensor] (positional, not a
    # dict) -- the graph has exactly one output, named "heatmaps" (confirmed
    # via onnx.load(...).graph.output).
    onnx_outputs = onnx_program(**sample_inputs)
    output_name = "heatmaps"
    onnx_heatmaps = onnx_outputs[0]
    sanity_max_abs_err = (onnx_heatmaps.float().cpu() - pytorch_out.heatmaps.float().cpu()).abs().max().item()
    print(f"[export_onnx] immediate sanity check: output '{output_name}', "
          f"max_abs_err vs PyTorch = {sanity_max_abs_err:.6f} (informal -- see Gate C for the real check)")

    metadata = {
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path) if onnx_path.is_file() else None,
        "output_name": output_name,
        "opset_version": OPSET_VERSION,
        "optimize": False,
        "batch_axis": batch_axis,
        "dynamic_batch_export_attempted": B == 1,
        "dynamic_batch_export_error": dynamic_export_error,
        "dynamic_axes": ({"pixel_values": {"0": "batch"}, "dataset_index": {"0": "batch"}}
                          if batch_axis == "dynamic" else None),
        "static_spatial_shape": [pixel_values.shape[2], pixel_values.shape[3]],
        "static_batch_size": None if batch_axis == "dynamic" else int(pixel_values.shape[0]),
        "max_batch": MAX_BATCH if batch_axis == "dynamic" else int(pixel_values.shape[0]),
        "per_slot_dataset_index": per_slot_dataset_index,
        "distinct_crops_used": B > 1,
        "checkpoint": checkpoint,
        "golden_env": golden_env,
        "immediate_sanity_max_abs_err": sanity_max_abs_err,
        "exporter": {
            "class": "transformers.exporters.OnnxExporter",
            "not_independent_of": "torch.export -> torch.onnx.export (see module docstring)",
            "tested_versions": OnnxExporter.tested_versions,
            "min_versions": OnnxExporter.min_versions,
        },
        "installed_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxscript": onnxscript.__version__,
        },
    }
    metadata_path = args.output_dir / ("export_metadata.json" if B == 1 else f"export_metadata_b{B}.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[export_onnx] wrote {metadata_path}")


if __name__ == "__main__":
    main()
