#!/usr/bin/env python3
"""Stage 4 prerequisite: a batch of GENUINELY DISTINCT crops, for testing
whether the model handles batch>1 correctly -- not just whether it accepts
the right shape.

    python -m golden.build_distinct_batch

Every "batch" measurement in this project so far (Stage 1's PyTorch sweep,
any future ONNX Runtime/TensorRT batch>1 benchmark) used B COPIES of one
crop via pixel_values.repeat(B,1,1,1). That is blind to cross-batch-element
bugs: if a batch of B identical inputs produces B identical outputs, that's
true whether or not information leaked between batch slots, because every
slot's "correct" answer is identical to every other slot's anyway. This
model has two real, specific reasons to worry about that -- both already
surfaced empirically in this project, not hypothetically:

  1. MoE routing: the backbone selects an expert per sample via a
     `dataset_index` gather/broadcast (`indices.view(-1, 1, 1)` then
     `hidden_state * (indices == i)` per expert). A batch with a MIXED
     dataset_index per slot is the direct test of whether that broadcast
     is actually per-sample or accidentally shared across the batch.
  2. Windowed self-attention: this is the exact mechanism that already
     broke dynamic-batch ONNX export (torch.export specialized the batch
     dim to a literal constant during tracing inside this attention path
     -- see conversion/export_onnx.py). A model that has trouble keeping
     batch symbolic during tracing is a reasonable place to also suspect
     it might have trouble keeping batch elements independent at runtime.

This script builds 16 slots: slot 0 is the REAL golden crop
(golden/person_crop.npy) at dataset_index=0, unchanged, so every existing
Stage 1-3 golden-reference test stays valid. Slots 1-15 are deterministic,
substantially-different perturbations of that same crop (brightness scale,
additive offset, spatial roll -- applied to the already-normalized
pixel_values tensor, not raw pixels, since that's the actual model input
this project's backends all consume), each assigned a DIFFERENT
dataset_index (cycling through all 6 MoE experts). Each of the 16 slots is
run through the model INDIVIDUALLY (batch=1) under the same pinned-cuDNN-
determinism, proven-bit-stable discipline as stage1_benchmark.py's
generate_golden_reference -- these 16 individual results are the ground
truth every future batch>1 benchmark's per-slot outputs get checked
against, at every batch size, for every backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from stage1_benchmark import env_fingerprint

GOLDEN_DIR = REPO_ROOT / "golden"
NUM_SLOTS = 16
NUM_EXPERTS = 6
N_VERIFY_RUNS = 5


def make_variant(base: torch.Tensor, slot: int) -> torch.Tensor:
    """Deterministic, substantially-different perturbation of `base`
    (slot 0 returns `base` unchanged). Operates on the already-normalized
    pixel_values tensor -- not simulating realistic photometric variation,
    just guaranteeing each slot's content actually differs, and by enough
    that a cross-slot mix-up would be numerically obvious."""
    if slot == 0:
        return base.clone()
    gen = torch.Generator(device="cpu").manual_seed(1000 + slot)
    brightness = 0.85 + 0.30 * torch.rand((), generator=gen).item()   # [0.85, 1.15]
    offset = (torch.rand((), generator=gen).item() - 0.5) * 0.2        # [-0.1, 0.1]
    shift_h = int(torch.randint(-24, 25, (), generator=gen).item())    # up to +-24px of 256
    shift_w = int(torch.randint(-18, 19, (), generator=gen).item())    # up to +-18px of 192
    variant = base * brightness + offset
    variant = torch.roll(variant, shifts=(shift_h, shift_w), dims=(2, 3))
    return variant.to(base.dtype)


@torch.no_grad()
def run_single(model, pixel_values: torch.Tensor, dataset_index: int, device: str) -> torch.Tensor:
    kwargs = {"dataset_index": torch.full((1,), dataset_index, dtype=torch.long, device=device)}
    outputs = model(pixel_values=pixel_values, **kwargs)
    return outputs.heatmaps.detach().clone()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    checkpoint = resolve_checkpoint(None)
    print(f"[build_distinct_batch] loading ViTPose++-L from {checkpoint} ({device})")
    _processor, model = load_pose_model(checkpoint, device, dtype)
    model.eval()

    base_crop = torch.from_numpy(np.load(GOLDEN_DIR / "person_crop.npy")).to(device=device, dtype=dtype)
    dataset_indices = [slot % NUM_EXPERTS for slot in range(NUM_SLOTS)]
    dataset_indices[0] = 0  # slot 0 must match the existing golden reference exactly

    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    crops, outputs, slot_meta = [], [], []
    try:
        for slot in range(NUM_SLOTS):
            crop = make_variant(base_crop, slot)
            ds_idx = dataset_indices[slot]

            runs = [run_single(model, crop, ds_idx, device).cpu() for _ in range(N_VERIFY_RUNS)]
            ref = runs[0]
            for r in runs[1:]:
                if not torch.equal(r, ref):
                    raise SystemExit(
                        f"[build_distinct_batch] slot {slot} is NOT bit-stable across "
                        f"{N_VERIFY_RUNS} runs -- treating as a build failure, not saving "
                        f"an untrustworthy fixture.")

            crops.append(crop[0].cpu())  # drop the (1, ...) leading batch dim before stacking
            outputs.append(ref[0])
            slot_meta.append({"slot": slot, "dataset_index": ds_idx,
                               "is_real_crop": slot == 0,
                               "verified_bit_identical_runs": N_VERIFY_RUNS})
            print(f"[build_distinct_batch] slot {slot:2d}  dataset_index={ds_idx}  "
                  f"{'(real crop)' if slot == 0 else '(synthetic variant)'}  bit-stable OK")
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark

    # Verify slot 0 matches the EXISTING golden reference exactly -- this fixture
    # must be a strict superset of what Stage 1-3 already validated, not a
    # parallel, possibly-inconsistent one.
    existing_golden = np.load(GOLDEN_DIR / "pytorch_fp16_output.npy")[0]  # drop its (1, ...) dim too
    if not np.array_equal(outputs[0].numpy(), existing_golden):
        raise SystemExit(
            "[build_distinct_batch] slot 0's output does NOT match the existing "
            "golden/pytorch_fp16_output.npy -- refusing to save a fixture that's "
            "inconsistent with Stage 1-3's already-established ground truth.")
    print("[build_distinct_batch] slot 0 confirmed identical to golden/pytorch_fp16_output.npy")

    # Cheap but real assertion that slots are actually distinct, not accidentally
    # collapsed back to copies of each other by the perturbation logic.
    crops_stacked = torch.stack(crops)
    for i in range(1, NUM_SLOTS):
        if torch.equal(crops_stacked[i], crops_stacked[0]):
            raise SystemExit(f"[build_distinct_batch] slot {i} is byte-identical to slot 0 -- "
                              f"perturbation failed to produce a distinct crop.")

    np.save(GOLDEN_DIR / "distinct_crops.npy", crops_stacked.numpy())
    np.save(GOLDEN_DIR / "distinct_pytorch_fp16_outputs.npy", torch.stack(outputs).numpy())

    env = env_fingerprint(device)
    env["num_slots"] = NUM_SLOTS
    env["num_experts"] = NUM_EXPERTS
    env["slots"] = slot_meta
    env["crops_shape"] = list(crops_stacked.shape)
    env["outputs_shape"] = list(torch.stack(outputs).shape)
    (GOLDEN_DIR / "distinct_batch_env.json").write_text(json.dumps(env, indent=2))

    print(f"[build_distinct_batch] wrote {GOLDEN_DIR / 'distinct_crops.npy'}, "
          f"{GOLDEN_DIR / 'distinct_pytorch_fp16_outputs.npy'}, "
          f"{GOLDEN_DIR / 'distinct_batch_env.json'}")


if __name__ == "__main__":
    main()
