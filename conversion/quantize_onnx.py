#!/usr/bin/env python3
"""Stage 6 Gate B: insert INT8 Q/DQ nodes into the faithful FP16 ONNX export.

    python -m conversion.quantize_onnx                    # batch 1
    python -m conversion.quantize_onnx --batch-size 16    # any static export
    python -m conversion.quantize_onnx --recalibrate      # redo the scales from the real corpus

TensorRT 11.3 has no implicit INT8 calibration (no BuilderFlag.INT8, no
IInt8Calibrator*): INT8 only exists as EXPLICIT Q/DQ pairs in the graph,
which a strongly-typed build (conversion/build_int8_engine.py) takes at face
value. This script decides where those pairs go and what their scales are.

Why the first version of this script was replaced
---------------------------------------------------
It ran `onnxruntime.quantization.quantize_static` with ORT's default op set,
entropy-calibrated on Gate A's synthetic corpus (96 perturbations of one
photo), and its engine was wrong: 72 px mean keypoint error on the golden
crop, 12 px mean / 48.7 px p95 (256x192 crop px) on 400 real crops. Running
the SAME Q/DQ graph in ONNX Runtime gave the same error (76.6 px on golden),
so the fault was the recipe, not the TensorRT build. ORT's default quantized
nearly every tensor: residual Adds, LayerNorm gamma/beta, every Linear bias,
Softmax, the MoE mask arithmetic and the heatmap output itself. Bisecting
that graph (removing Q/DQ by consumer type) showed that keeping Q/DQ ONLY on
MatMul inputs brings the error to 0.44 px median on real crops, and
per-channel weights plus real-footage calibration to 0.32 px. That output
also carried float16 scales at opset 18, which the ONNX spec does not allow
(float16 Q/DQ scales arrived in opset 19): the old "onnx.checker limitation"
note was wrong -- ORT itself refused to load that graph.

The recipe
----------
- Q/DQ on the inputs of every MatMul, and nothing else. Per block that is
  12 Linear MatMuls (q, k, v, attention output, fc1, fc2 and the 6 MoE
  experts) plus the 2 attention act x act MatMuls (q.k^T and probs.V).
  LayerNorm, GELU, Softmax, residual Adds, biases, the MoE mask, the
  patch-embedding Conv and the deconvolution head stay FP16.
- Weights: per-output-channel symmetric INT8, scale = absmax / 127 of each
  column of the (transposed) weight. No calibration.
- Activations: per-tensor symmetric INT8, scale = amax / 127 with amax =
  the --percentile (default 99.99) of |x| over the real calibration crops
  (calibration/real_corpus.py, train videos only), collected in PyTorch with
  forward hooks. Softmax probabilities use amax = 1.0: they are bounded by 1,
  and clipping the largest attention weights is the one thing a percentile
  would do there. (ORT can't serve as the collector: exposing intermediate
  outputs of this fp16 graph trips ORT 1.30's InsertedPrecisionFreeCast
  type error.)
- Keeping the attention act x act MatMuls in FP16 is more accurate
  (0.30 vs 0.39 px median) but costs a lot of speed at batch 16 (445 vs
  524 crops/s): TensorRT fuses the quantized attention far better. Keeping
  fc2 + experts in FP16 drops the speedup to 1.05x. Both stay INT8 here.
- Only expert 0 (COCO, what production runs) is calibrated and validated.
  The engine still accepts other dataset_index values; their accuracy is
  not measured.

Scale cache
-----------
The scales are written to results/onnx/int8_calibration_scales.json
(committed) together with the sha256 of calibration/real_manifest.json they
came from. Every batch size's export reuses them (the exports share tensor
names; the activation distribution does not depend on batch size), and a
rebuild without the private footage on disk uses the cache. --recalibrate,
or a cache whose manifest sha256 no longer matches, recomputes them.

This is Gate B: it makes no accuracy claim. Gate C is
tests/test_tensorrt_int8_equivalence.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from conversion.build_engine import sha256_file

SCALES_PATH = REPO_ROOT / "results" / "onnx" / "int8_calibration_scales.json"
REAL_MANIFEST = REPO_ROOT / "calibration" / "real_manifest.json"
LINEAR_ROLES = {  # weight-name fragment -> role of the activation that Linear reads
    ".attention.attention.query.": "qkv_in",
    ".attention.attention.key.": "qkv_in",
    ".attention.attention.value.": "qkv_in",
    ".attention.output.dense.": "proj_in",
    ".mlp.fc1.": "fc1_in",
    ".mlp.fc2.": "fc2_in",
    ".mlp.experts.": "fc2_in",
}
ACT_ROLES = ("qkv_in", "proj_in", "fc1_in", "fc2_in", "q_scaled", "k_scaled", "v", "attn_probs")
# 12 = q, k, v, attention output, fc1, fc2 + 6 MoE experts; 6 = the MoE-fused export
WEIGHT_MATMULS_PER_BLOCK = {12, 6}
NUM_BLOCKS = 24
HIST_BINS = 4096


@dataclass
class QuantPoint:
    tensor: str
    kind: str                      # "weight" | "act"
    role: str
    block: int
    consumers: list = field(default_factory=list)   # [(node_name, input_index)]
    weight_init: str | None = None
    perm: list | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--onnx-path", type=Path, default=None)
    p.add_argument("--onnx-metadata", type=Path, default=None,
                   help="Stage 2's export record; its onnx_sha256 is verified against --onnx-path.")
    p.add_argument("--scales", type=Path, default=SCALES_PATH)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--metadata-output", type=Path, default=None)
    args = p.parse_args()
    sfx = "" if args.batch_size == 1 else f"_b{args.batch_size}"
    onnx_dir = REPO_ROOT / "results" / "onnx"
    args.onnx_path = args.onnx_path or onnx_dir / f"vitpose_plus_l{sfx}.onnx"
    args.onnx_metadata = args.onnx_metadata or onnx_dir / f"export_metadata{sfx}.json"
    args.output = args.output or onnx_dir / f"vitpose_plus_l_int8_qdq{sfx}.onnx"
    args.metadata_output = args.metadata_output or onnx_dir / f"int8_quantization_metadata{sfx}.json"
    return args


# ---------------------------------------------------------------- graph analysis

def analyze_graph(model: onnx.ModelProto) -> tuple[list[QuantPoint], float]:
    """Every MatMul input -> a QuantPoint. Refuses on any structure it doesn't recognise,
    since a silently unquantized (or wrongly calibrated) MatMul is exactly the kind of
    unchecked assumption Gate C would only catch indirectly.

    Handles both exports this recipe has met: Stage 2's (weights as named initializers
    behind a runtime Transpose, 12 Linear MatMuls per block with the 6 MoE experts) and a
    torch.onnx dynamo export of the MoE-FUSED model (AutoClipping's
    scripts/export_vitpose_trt.py: 6 Linear MatMuls per block, the weight Transpose folded
    into an anonymous `val_*` initializer, bias Adds with the bias first, and the SDPA
    q/k scaling folded to a constant)."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for x in n.input:
            cons[x].append(n)
    inits = {t.name: t for t in g.initializer}

    def weight_of(t):
        n = prod.get(t)
        if n is not None and n.op_type == "Transpose" and n.input[0] in inits:
            perm = list(helper.get_attribute_value(next(a for a in n.attribute if a.name == "perm")))
            return n.input[0], perm
        if t in inits:
            return t, None
        return None, None

    def linear_name(matmul):
        """Name of the backbone Linear a weight MatMul implements: its weight's name, or --
        when the exporter folded the weight into an anonymous initializer -- the name of the
        bias its output is added to."""
        w, _ = weight_of(matmul.input[1])
        if w is None:
            return None
        if "encoder.layer." in w:
            return w
        for c in cons[matmul.output[0]]:
            if c.op_type == "Add":
                bias = next((x for x in c.input if x in inits and x.endswith(".bias")), None)
                if bias is not None:
                    return bias
        return w

    def scalar_value(t):
        """Value of a scalar constant expression (initializer / Constant / Sqrt / CastLike of
        one), or None if `t` isn't one."""
        if t in inits:
            a = numpy_helper.to_array(inits[t])
            return float(a) if a.size == 1 else None
        n = prod.get(t)
        if n is None:
            return None
        if n.op_type == "Constant":
            a = numpy_helper.to_array(n.attribute[0].t)
            return float(a) if a.size == 1 else None
        if n.op_type == "CastLike":
            return scalar_value(n.input[0])
        if n.op_type == "Sqrt":
            v = scalar_value(n.input[0])
            return None if v is None else float(np.sqrt(v))
        return None

    def data_input(n):
        """The activation operand of an elementwise op (the one that isn't a constant)."""
        if n.op_type in ("Add", "Mul"):
            for x in n.input:
                if x not in inits and scalar_value(x) is None and prod.get(x, n).op_type != "Constant":
                    return x
        return n.input[0]

    def block_of(name):
        return int(name.split("encoder.layer.")[1].split(".")[0])

    def trace_to_linear(t):
        """Follow the data path up through reshapes/transposes/scaling/bias to the Linear
        MatMul that produced it; returns that Linear's name."""
        for _ in range(16):
            n = prod.get(t)
            if n is None:
                return None
            if n.op_type == "MatMul":
                return linear_name(n)
            if n.op_type not in ("Mul", "Transpose", "Reshape", "Add"):
                return None
            t = data_input(n)
        return None

    points: dict[str, QuantPoint] = {}

    def add(tensor, kind, role, block, consumer, idx, weight=None, perm=None):
        qp = points.setdefault(tensor, QuantPoint(tensor, kind, role, block, weight_init=weight, perm=perm))
        if (qp.role, qp.block) != (role, block):
            raise SystemExit(f"[quantize_onnx] tensor {tensor} feeds MatMuls with different roles "
                             f"({qp.role}@{qp.block} vs {role}@{block}) -- unexpected graph structure.")
        qp.consumers.append((consumer.name, idx))

    attention_scaling = None
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        w, perm = weight_of(n.input[1])
        if w is not None:
            name = linear_name(n)
            role = next((r for frag, r in LINEAR_ROLES.items() if frag in name), None)
            if role is None or "encoder.layer." not in name:
                raise SystemExit(f"[quantize_onnx] REFUSING: MatMul {n.name} reads weight {w} ({name}), "
                                 f"which is not a backbone Linear this recipe knows.")
            b = block_of(name)
            add(n.input[1], "weight", "weight", b, n, 1, weight=w, perm=perm)
            add(n.input[0], "act", role, b, n, 0)
            continue
        # activation x activation: q.k^T or probs.V
        a0 = prod.get(n.input[0])
        if a0 is not None and a0.op_type == "Softmax":
            wv = trace_to_linear(n.input[1])
            if wv is None or ".value." not in wv:
                raise SystemExit(f"[quantize_onnx] REFUSING: {n.name} (probs.V) second input does not "
                                 f"trace back to a value projection.")
            b = block_of(wv)
            add(n.input[0], "act", "attn_probs", b, n, 0)
            add(n.input[1], "act", "v", b, n, 1)
            continue
        wq, wk = trace_to_linear(n.input[0]), trace_to_linear(n.input[1])
        if wq is None or wk is None or ".query." not in wq or ".key." not in wk:
            raise SystemExit(f"[quantize_onnx] REFUSING: act x act MatMul {n.name} is neither q.k^T "
                             f"nor probs.V as far as input tracing can tell ({wq}, {wk}).")
        b = block_of(wq)
        add(n.input[0], "act", "q_scaled", b, n, 0)
        add(n.input[1], "act", "k_scaled", b, n, 1)
        # The exporters decompose SDPA as (q*sqrt(s)) @ (k^T*sqrt(s)); calibration applies the
        # same sqrt(s), so check s is the attention module's scaling (head_dim ** -0.5).
        mul = prod[n.input[0]]
        c = next((v for v in (scalar_value(x) for x in mul.input) if v is not None), None) \
            if mul.op_type == "Mul" else None
        if c is None:
            raise SystemExit(f"[quantize_onnx] REFUSING: can't read the SDPA scaling constant feeding "
                             f"{mul.name}; calibration's q/k scaling would be unverified.")
        s = c * c
        if attention_scaling is not None and abs(s - attention_scaling) > 1e-6:
            raise SystemExit(f"[quantize_onnx] REFUSING: blocks disagree on the SDPA scaling ({s} vs "
                             f"{attention_scaling}).")
        attention_scaling = s

    pts = list(points.values())
    per_block = collections.Counter((p.block, p.kind) for p in pts)
    act_roles = collections.defaultdict(set)
    for p in pts:
        if p.kind == "act":
            act_roles[p.block].add(p.role)
    layouts = {per_block[(b, "weight")] for b in range(NUM_BLOCKS)}
    for b in range(NUM_BLOCKS):
        if (len(layouts) != 1 or per_block[(b, "weight")] not in WEIGHT_MATMULS_PER_BLOCK
                or act_roles[b] != set(ACT_ROLES)):
            raise SystemExit(f"[quantize_onnx] REFUSING: block {b} has {per_block[(b, 'weight')]} weight "
                             f"MatMuls (expected the same one of {sorted(WEIGHT_MATMULS_PER_BLOCK)} in every "
                             f"block) and activation roles {sorted(act_roles[b])} (expected {sorted(ACT_ROLES)}).")
    return pts, attention_scaling


# ---------------------------------------------------------------- calibration

class _AbsCollector:
    """Exact percentile of |x| over every element of every calibration crop: pass 1 finds
    the global max, pass 2 histograms |x| into HIST_BINS bins over [0, max]."""

    def __init__(self):
        self.max = 0.0
        self.hist = None

    def observe(self, x: torch.Tensor, phase: str) -> None:
        a = x.detach().float().abs()
        if phase == "max":
            self.max = max(self.max, float(a.max()))
        else:
            h = torch.histc(a, bins=HIST_BINS, min=0.0, max=self.max if self.max > 0 else 1.0).double()
            self.hist = h if self.hist is None else self.hist + h

    def percentile(self, q: float) -> float:
        c = torch.cumsum(self.hist, 0).cpu().numpy()
        return float((np.searchsorted(c, q / 100.0 * c[-1]) + 1) / HIST_BINS * self.max)


def calibrate(percentile: float, attention_scaling: float) -> dict:
    from calibration.real_corpus import RealCropSet
    calib = RealCropSet("calib")
    torch.backends.cudnn.benchmark = False
    _, model = load_pose_model(resolve_checkpoint(None), "cuda", torch.float16)
    model.eval()
    layers = model.backbone.encoder.layer
    if len(layers) != NUM_BLOCKS:
        raise SystemExit(f"[quantize_onnx] model has {len(layers)} blocks, graph analysis expects {NUM_BLOCKS}.")
    col = {(b, r): _AbsCollector() for b in range(NUM_BLOCKS) for r in ACT_ROLES if r != "attn_probs"}
    phase = {"p": "max"}
    hooks = []

    for b, layer in enumerate(layers):
        attn = layer.attention.attention
        # the graph may carry sqrt(s) as an fp16 constant (0.12497 for 0.125): compare at fp16 precision
        if abs(attn.scaling - attention_scaling) > 1e-3 * attn.scaling:
            raise SystemExit(f"[quantize_onnx] block {b}: module scaling {attn.scaling} != graph's "
                             f"SDPA constant {attention_scaling}.")

        def attn_hook(mod, inputs, b=b):
            x = inputs[0]
            shape = (x.shape[0], -1, mod.num_attention_heads, mod.attention_head_size)
            c = mod.scaling ** 0.5
            col[(b, "qkv_in")].observe(x, phase["p"])
            col[(b, "q_scaled")].observe(mod.query(x).view(shape).transpose(1, 2) * c, phase["p"])
            col[(b, "k_scaled")].observe(mod.key(x).view(shape).transpose(1, 2) * c, phase["p"])
            col[(b, "v")].observe(mod.value(x).view(shape).transpose(1, 2), phase["p"])

        def input_hook(role, b=b):
            return lambda mod, inputs: col[(b, role)].observe(inputs[0], phase["p"])

        hooks += [attn.register_forward_pre_hook(attn_hook),
                  layer.attention.output.dense.register_forward_pre_hook(input_hook("proj_in")),
                  layer.mlp.fc1.register_forward_pre_hook(input_hook("fc1_in")),
                  layer.mlp.fc2.register_forward_pre_hook(input_hook("fc2_in"))]

    t0 = time.perf_counter()
    with torch.inference_mode():
        for phase["p"] in ("max", "hist"):
            for k in range(0, len(calib), 16):
                x = torch.from_numpy(calib.pixel_values[k:k + 16]).cuda()
                model(pixel_values=x, dataset_index=torch.zeros(len(x), dtype=torch.int64, device="cuda"))
    for h in hooks:
        h.remove()
    amax = {str(b): {r: (1.0 if r == "attn_probs" else col[(b, r)].percentile(percentile)) for r in ACT_ROLES}
            for b in range(NUM_BLOCKS)}
    del model
    torch.cuda.empty_cache()
    return {
        "percentile": percentile,
        "attn_probs_amax": "1.0 (softmax output bound), not calibrated",
        "attention_scaling": attention_scaling,
        "calibration_manifest": str(REAL_MANIFEST.relative_to(REPO_ROOT)),
        "calibration_manifest_sha256": calib.manifest_sha256,
        "calibration_num_crops": len(calib),
        "dataset_index": 0,
        "calibration_wall_clock_s": round(time.perf_counter() - t0, 1),
        "amax": amax,
    }


def load_or_calibrate(args, attention_scaling: float) -> tuple[dict, str]:
    manifest_sha = sha256_file(REAL_MANIFEST) if REAL_MANIFEST.is_file() else None
    if args.scales.is_file() and not args.recalibrate:
        cached = json.loads(args.scales.read_text())
        if cached["calibration_manifest_sha256"] != manifest_sha:
            print(f"[quantize_onnx] {args.scales} was calibrated against a different "
                  f"{REAL_MANIFEST.name} -- recalibrating.")
        elif cached["percentile"] != args.percentile:
            print(f"[quantize_onnx] {args.scales} used percentile {cached['percentile']}, "
                  f"--percentile is {args.percentile} -- recalibrating.")
        else:
            return cached, "cache"
    scales = calibrate(args.percentile, attention_scaling)
    args.scales.write_text(json.dumps(scales, indent=2) + "\n")
    print(f"[quantize_onnx] calibrated {NUM_BLOCKS * len(ACT_ROLES)} activation scales on "
          f"{scales['calibration_num_crops']} real crops in {scales['calibration_wall_clock_s']}s "
          f"-> {args.scales}")
    return scales, "calibrated"


# ---------------------------------------------------------------- Q/DQ insertion

def insert_qdq(model: onnx.ModelProto, points: list[QuantPoint], amax: dict) -> int:
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    by_name = {n.name: n for n in g.node}
    producer_idx = {o: k for k, n in enumerate(g.node) for o in n.output}
    after = collections.defaultdict(list)
    for p in points:
        if p.kind == "weight":
            w = numpy_helper.to_array(inits[p.weight_init]).astype(np.float32)
            wt = np.transpose(w, p.perm) if p.perm else w          # the MatMul's [in, out] operand
            scale = (np.maximum(np.abs(wt).max(axis=0), 1e-8) / 127.0).astype(np.float16)
            kw = {"axis": wt.ndim - 1}
        else:
            scale = np.array(max(amax[str(p.block)][p.role], 1e-6) / 127.0, dtype=np.float16)
            kw = {}
        s_name, z_name = f"{p.tensor}__int8_scale", f"{p.tensor}__int8_zp"
        g.initializer.extend([numpy_helper.from_array(scale, s_name),
                              numpy_helper.from_array(np.zeros(scale.shape, np.int8), z_name)])
        q = helper.make_node("QuantizeLinear", [p.tensor, s_name, z_name], [p.tensor + "__int8_q"],
                             name=p.tensor + "__QuantizeLinear", **kw)
        dq = helper.make_node("DequantizeLinear", [p.tensor + "__int8_q", s_name, z_name],
                              [p.tensor + "__int8_dq"], name=p.tensor + "__DequantizeLinear", **kw)
        after[producer_idx.get(p.tensor, -1)] += [q, dq]
        for node_name, idx in p.consumers:
            by_name[node_name].input[idx] = p.tensor + "__int8_dq"
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(after.get(-1, []))                 # graph-input producers (none today)
    for k, n in enumerate(nodes):
        g.node.append(n)
        g.node.extend(after.get(k, []))
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = max(o.version, 19)       # float16 Q/DQ scales are only legal from opset 19
    return len(points)


def main() -> None:
    args = parse_args()
    export_meta = json.loads(args.onnx_metadata.read_text())
    actual = sha256_file(args.onnx_path)
    if actual != export_meta["onnx_sha256"]:
        raise SystemExit(f"[quantize_onnx] REFUSING to quantize: {args.onnx_path} has sha256 {actual}, "
                         f"but {args.onnx_metadata} says {export_meta['onnx_sha256']}. Re-run Stage 2's gates.")
    if export_meta["optimize"] is not False or export_meta["batch_axis"] != "static":
        raise SystemExit(f"[quantize_onnx] REFUSING: expected the faithful (optimize=False, static batch) "
                         f"export, got optimize={export_meta['optimize']} batch_axis={export_meta['batch_axis']}.")

    model = onnx.load(str(args.onnx_path))
    points, attention_scaling = analyze_graph(model)
    scales, scales_source = load_or_calibrate(args, attention_scaling)
    n_pairs = insert_qdq(model, points, scales["amax"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_name = args.output.name + ".data"
    (args.output.parent / data_name).unlink(missing_ok=True)       # onnx.save appends otherwise
    onnx.save(model, str(args.output), save_as_external_data=True, location=data_name,
              all_tensors_to_one_file=True, size_threshold=1024)
    onnx.checker.check_model(str(args.output), full_check=True)
    print(f"[quantize_onnx] {n_pairs} Q/DQ pairs -> {args.output}; onnx.checker (full_check) OK")

    ops = collections.Counter(n.op_type for n in model.graph.node)
    roles = collections.Counter(p.role for p in points)
    metadata = {
        "model": "ViTPose++-L",
        "stage": "Stage 6 Gate B -- INT8 Q/DQ insertion (MatMul inputs only)",
        "batch": args.batch_size,
        "source_onnx_path": str(args.onnx_path),
        "source_onnx_sha256": actual,
        "output_onnx_path": str(args.output),
        "output_onnx_sha256": sha256_file(args.output),
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "recipe": {
            "quantized_op": "MatMul inputs only (Linear x weight and attention act x act)",
            "kept_fp16": "LayerNorm, GELU, Softmax, residual Adds, biases, MoE mask, patch-embedding "
                         "Conv, deconvolution head, heatmap output",
            "weights": "per-output-channel symmetric INT8, absmax / 127",
            "activations": f"per-tensor symmetric INT8, p{scales['percentile']} of |x| / 127 on real crops; "
                           f"softmax probabilities amax = 1.0",
            "zero_point": 0,
        },
        "qdq_pairs": n_pairs,
        "qdq_pairs_by_role": dict(sorted(roles.items())),
        "scales_path": str(args.scales.relative_to(REPO_ROOT)),
        "scales_sha256": sha256_file(args.scales),
        "scales_source": scales_source,
        "calibration_manifest_sha256": scales["calibration_manifest_sha256"],
        "calibration_num_crops": scales["calibration_num_crops"],
        "dataset_index_validated": 0,
        "op_histogram": dict(sorted(ops.items())),
        "installed_versions": {"python": platform.python_version(), "onnx": onnx.__version__,
                               "torch": torch.__version__},
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[quantize_onnx] wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
