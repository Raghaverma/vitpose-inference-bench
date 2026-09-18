#!/usr/bin/env python3
"""Stage 2 Gate B: structural validation of the exported ONNX graph.

    python -m conversion.inspect_onnx

Checks the exported .onnx from conversion/export_onnx.py: does it load, is
the graph structurally valid (onnx.checker), and does it look like what we
think we exported (input/output names, dtypes, shapes, opset, operator
histogram, initializer count/size)? No inference happens here -- that's
Gate C (tests/test_onnx_equivalence.py) and Gate D (backends/onnxruntime.py).

The operator histogram doubles as the control fixture the export-integrity
side of Stage 2 needs: results/onnx/structural_validation.json is small and
diffable, so a future re-export (new opset, new transformers version, a
flipped optimize=True) that silently changes the graph's op composition
shows up as a diff here before anyone has to debug a numerical regression.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
from onnx import numpy_helper

from baseline import REPO_ROOT

ELEM_TYPE_NAMES = {v: k for k, v in onnx.TensorProto.DataType.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-path", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "vitpose_plus_l.onnx")
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "results" / "onnx" / "structural_validation.json")
    return p.parse_args()


def dim_to_str(dim) -> str | int:
    if dim.HasField("dim_value"):
        return dim.dim_value
    if dim.HasField("dim_param") and dim.dim_param:
        return dim.dim_param
    return "?"


def tensor_spec(value_info) -> dict:
    t = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": ELEM_TYPE_NAMES.get(t.elem_type, f"UNKNOWN({t.elem_type})"),
        "shape": [dim_to_str(d) for d in t.shape.dim],
    }


def main() -> None:
    args = parse_args()
    if not args.onnx_path.is_file():
        raise SystemExit(f"{args.onnx_path} not found -- run conversion/export_onnx.py first "
                          "(Gate A must pass before Gate B).")

    report: dict = {"onnx_path": str(args.onnx_path), "checks": {}}

    # 1. Does it load, and is the graph structurally valid?
    try:
        # load_external_data=False: keep initializers pointing at the .onnx_data
        # sidecar instead of inlining ~800MB+ of weights into this process just
        # to report their size -- onnx.checker below reads the graph structure
        # either way and doesn't need the tensor bytes resident.
        model = onnx.load(str(args.onnx_path), load_external_data=False)
        report["checks"]["loads"] = True
    except Exception as exc:
        report["checks"]["loads"] = False
        report["checks"]["load_error"] = str(exc)
        args.output.write_text(json.dumps(report, indent=2))
        raise SystemExit(f"[inspect_onnx] FAILED to load {args.onnx_path}: {exc}")

    try:
        # Pass the PATH, not the already-loaded ModelProto: a ModelProto in
        # memory carries no directory context, so the checker can't resolve
        # external-data tensors (like the position embeddings) relative to
        # the .onnx file's own directory unless it's given that path directly.
        onnx.checker.check_model(str(args.onnx_path), full_check=True)
        report["checks"]["checker_passed"] = True
    except onnx.checker.ValidationError as exc:
        report["checks"]["checker_passed"] = False
        report["checks"]["checker_error"] = str(exc)
        args.output.write_text(json.dumps(report, indent=2))
        raise SystemExit(f"[inspect_onnx] FAILED onnx.checker.check_model: {exc}")

    # 2. Opset.
    report["opset_imports"] = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    report["ir_version"] = model.ir_version

    # 3. Inputs / outputs (name, dtype, shape -- including any symbolic dims).
    report["inputs"] = [tensor_spec(i) for i in model.graph.input]
    report["outputs"] = [tensor_spec(o) for o in model.graph.output]

    # 4. Operators -- the diffable control fixture.
    op_counts = Counter(node.op_type for node in model.graph.node)
    report["operators"] = {
        "unique_op_types": len(op_counts),
        "total_nodes": sum(op_counts.values()),
        "histogram": dict(sorted(op_counts.items())),
    }

    # 5. Initializers (weights) -- count and total size, not values (that's
    # what Gate C's checkpoint-vs-graph parity would need if we ever bake a
    # full weight-by-weight diff; out of scope for this pass).
    total_bytes = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init) if not init.data_location else None
        if arr is not None:
            total_bytes += arr.nbytes
        else:
            # external_data=True -- size lives in the external_data entries.
            for kv in init.external_data:
                if kv.key == "length":
                    total_bytes += int(kv.value)
    report["initializers"] = {
        "count": len(model.graph.initializer),
        "total_size_mb": round(total_bytes / 2**20, 1),
        "external_data": any(init.data_location for init in model.graph.initializer),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"[inspect_onnx] loads: OK, onnx.checker: OK")
    print(f"[inspect_onnx] opset: {report['opset_imports']}")
    print(f"[inspect_onnx] inputs:")
    for i in report["inputs"]:
        print(f"    {i['name']:15s} {i['dtype']:10s} {i['shape']}")
    print(f"[inspect_onnx] outputs:")
    for o in report["outputs"]:
        print(f"    {o['name']:15s} {o['dtype']:10s} {o['shape']}")
    print(f"[inspect_onnx] operators: {report['operators']['unique_op_types']} unique types, "
          f"{report['operators']['total_nodes']} nodes total")
    print(f"[inspect_onnx] initializers: {report['initializers']['count']} tensors, "
          f"{report['initializers']['total_size_mb']:.1f} MB "
          f"(external_data={report['initializers']['external_data']})")
    print(f"[inspect_onnx] wrote {args.output}")


if __name__ == "__main__":
    main()
