#!/usr/bin/env python3
"""Stage 6 Gate A: build the TensorRT INT8 entropy-calibration corpus.

    source .venv/bin/activate && python3 -m calibration.build_calibration_corpus

Why this corpus exists, and what it can and can't guarantee
-------------------------------------------------------------
ViTPose++-L's backbone MLP routes every token through a 6-expert MoE
(`VitPoseNaiveMoe.forward` in
transformers/models/vitpose_backbone/modeling_vitpose_backbone.py): each of
the 6 experts' Linear layers runs on the FULL hidden_state unconditionally,
then gets multiplied by a `(dataset_index == i)` mask and summed. INT8
entropy calibration needs activation statistics from every branch that will
actually execute at inference time, so a calibration corpus that only ever
exercises expert 0 (COCO) would leave experts 1-5's Linear layers calibrated
on whatever statistics happen to leak through masked-out (zeroed) forward
passes -- not on real per-expert activation distributions.

We already checked whether this is even auditable after the fact. TensorRT's
EngineInspector (DETAILED profiling verbosity), run against the existing
engines/vitpose_b1_fp16.engine and dumped to
results/tensorrt/fp16_layer_info_full.json (180 layers total), shows
builder_optimization_level=3 fuses the entire MoE block into opaque kernels:
layer 10 is a "gemm" layer fusing 7 MatMuls (node_MatMul_92 + the 6 experts'
MatMuls) into one kernel, and layer 12 is a "kgen" layer named
__myl_MulMulMulMulMulMulAddAddAddAddAddAddConcAdd... fusing the six
per-expert masks and their summation into one kernel -- repeating across all
24 transformer blocks. Zero semantic layer names survive (grepping the full
dump for Gather/Select/Where/expert-named layers: zero matches). So:
per-expert calibration quality is NOT verifiable at the tensor level once
the engine is built. This script's per-expert sample-count gate (>=
MIN_SAMPLES_PER_EXPERT, enforced below and recorded in the manifest) is
therefore the ONLY preventive control available before engine build -- a
downstream accuracy check split by dataset_index (Gate C/D, not this
script's job) is the only way to actually catch a badly-calibrated expert
after the fact.

Content source (read before trusting these numbers)
-----------------------------------------------------
This repo has exactly one real photo (samples/sample.jpg, gitignored),
reduced to one detected+preprocessed crop (golden/person_crop.npy). There is
no public pose dataset or private sports footage integrated yet. Every one
of this corpus's 96 samples is a synthetic perturbation of that ONE crop --
this is explicitly a first-pass placeholder, not a claim of scene/pose
diversity. See calibration/manifest.json's "content_source_note" for the
full caveat (kept in the manifest itself, not just here, since that's the
artifact a future INT8-calibration step will actually load).

Sample construction
--------------------
- 96 samples, dataset_index assigned ROUND-ROBIN as `sample_id % 6` (not
  grouped in blocks of 16), so any prefix of the corpus is roughly
  expert-balanced too -- TensorRT's calibrator streams samples in batches,
  and a block-grouped ordering would present it with long expert-skewed
  windows even though the final aggregate count is balanced.
- Two augmentation families, alternating by sample index (even ->
  "classic_photometric_spatial", odd -> "motion_blur"). NOTE, worth stating
  plainly rather than leaving as a silent gotcha: because 6 (the expert
  cycle) is divisible by 2 (the family cycle), this pairing is NOT
  independent of dataset_index -- experts 0/2/4 land exclusively on
  "classic_photometric_spatial" samples and experts 1/3/5 exclusively on
  "motion_blur" ones. Each expert still gets 16 samples (the manifest gate
  this script enforces), just not a mix of both families. That's a real
  coverage gap in this first-pass corpus, not a bug in the round-robin/
  alternation logic -- both are implemented exactly as specified. Flagging
  it here so a future corpus revision (e.g. offsetting the family index by
  an odd stride) doesn't have to rediscover it by reading the numbers.
- "classic_photometric_spatial" reuses golden/build_distinct_batch.py's
  make_variant() math exactly (brightness scale, additive offset, spatial
  roll on the already-normalized tensor) -- reimplemented locally rather
  than imported, because make_variant() hardcodes its seed as `1000 + slot`
  and this corpus needs a disjoint seed range (`2000 + sample_id`) so the
  two fixture generators' RNG streams never overlap even coincidentally.
- "motion_blur" is new: de-normalize pixel_values to ~[0,1] pixel space
  (invert the processor's per-channel `(x - mean) / std`), convolve with a
  directional line kernel (angle in [0, 360) deg, length in [3, 9] px --
  a first-pass placeholder range, NOT measured from real fast-bowling /
  floodlight footage, same caveat as the content-source note above),
  re-normalize with the same mean/std. The kernel/convolution are
  hand-built with torch (F.conv2d, per-channel depthwise, seed
  `3000 + sample_id`) -- no new dependency.

Distinctness and determinism discipline
------------------------------------------
Every sample's forward pass is verified bit-identical across N_VERIFY_RUNS
reruns under pinned cudnn determinism before it's trusted -- the same
discipline golden/build_distinct_batch.py uses for its slots, applied here
too rather than assumed to carry over. Every sample's decoded pose is also
checked against the UNPERTURBED base crop run through the SAME expert (not
just expert 0), via compare_golden.decode_keypoints -- isolating the
augmentation's effect from whichever expert a sample happens to route
through. A sample whose pose barely moves (< MIN_KEYPOINT_SHIFT_PX) is
functionally inert calibration input and gets refused (SystemExit), the
same way build_distinct_batch.py refuses a non-distinct slot -- this script
never silently keeps a degenerate sample. sha256 collisions across saved
sample files are checked too, as a cheap guard against a seed/RNG mistake
silently collapsing two "different" samples into byte-identical content.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from compare_golden import decode_keypoints
from conversion.build_engine import sha256_file
from stage1_benchmark import env_fingerprint

GOLDEN_DIR = REPO_ROOT / "golden"
CALIBRATION_DIR = REPO_ROOT / "calibration"
SAMPLES_DIR = CALIBRATION_DIR / "samples"
MANIFEST_PATH = CALIBRATION_DIR / "manifest.json"

NUM_EXPERTS = 6
MIN_SAMPLES_PER_EXPERT = 16
NUM_SAMPLES = 96
RESOLUTION = (256, 192)
N_VERIFY_RUNS = 5
# Low but non-zero distinctness floor: this is "did the augmentation do
# anything at all to the decoded pose," not a numerical-equivalence bound --
# mirrors this repo's existing sub-pixel-agreement scale (see compare_golden.py
# and Stage 2/3's near-zero equivalence reports) for "these two things are
# numerically close" vs. what we want here, "these two things are NOT the
# same," so the bar sits just above 0.
MIN_KEYPOINT_SHIFT_PX = 0.05

# build_distinct_batch.py owns seeds 1000 + slot (slot in [0, 16)). This
# corpus's two families get their own disjoint ranges purely so the two
# scripts' RNG streams can never be confused for each other when reading
# either one's output later -- there is no shared state between them.
CLASSIC_SEED_BASE = 2000
MOTION_BLUR_SEED_BASE = 3000
MOTION_BLUR_ANGLE_RANGE_DEG = (0.0, 360.0)
MOTION_BLUR_LENGTH_RANGE_PX = (3.0, 9.0)


def classic_variant(base: torch.Tensor, sample_id: int) -> tuple[torch.Tensor, dict]:
    """Identical math to golden/build_distinct_batch.py's make_variant() for
    slot != 0 (brightness scale, additive offset, spatial roll on the
    normalized tensor) -- see that function's docstring for why this
    particular three-op combination is used. Seeded from CLASSIC_SEED_BASE,
    not make_variant()'s own `1000 + slot`, per the module docstring."""
    seed = CLASSIC_SEED_BASE + sample_id
    gen = torch.Generator(device="cpu").manual_seed(seed)
    brightness = 0.85 + 0.30 * torch.rand((), generator=gen).item()   # [0.85, 1.15]
    offset = (torch.rand((), generator=gen).item() - 0.5) * 0.2        # [-0.1, 0.1]
    shift_h = int(torch.randint(-24, 25, (), generator=gen).item())    # up to +-24px of 256
    shift_w = int(torch.randint(-18, 19, (), generator=gen).item())    # up to +-18px of 192
    variant = base * brightness + offset
    variant = torch.roll(variant, shifts=(shift_h, shift_w), dims=(2, 3))
    params = {"seed": seed, "brightness": brightness, "offset": offset,
              "shift_h_px": shift_h, "shift_w_px": shift_w}
    return variant.to(base.dtype), params


def _line_kernel(angle_deg: float, length_px: float, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    """Rasterize a `length_px`-long line segment at `angle_deg` into a
    square, sum-normalized 2D kernel -- a directional motion-blur PSF.
    Bilinearly splats oversampled points along the line into the kernel
    grid rather than snapping each point to its nearest pixel: nearest-pixel
    snapping would make the kernel's actual blur direction quantize toward
    0/45/90-degree increments regardless of the requested angle, which
    defeats the point of sampling angle uniformly from [0, 360).
    """
    k = int(2 * math.ceil(length_px / 2) + 3)  # odd, with margin for the line's extent
    kernel = torch.zeros((k, k), dtype=torch.float64)
    center = (k - 1) / 2.0
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    n_samples = max(int(length_px * 4), 8)  # oversample so the splat is smooth, not sparse
    for t in torch.linspace(-length_px / 2, length_px / 2, n_samples).tolist():
        x, y = center + t * dx, center + t * dy
        x0, y0 = math.floor(x), math.floor(y)
        for xi, yi in ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)):
            if 0 <= xi < k and 0 <= yi < k:
                w = (1 - abs(x - xi)) * (1 - abs(y - yi))
                if w > 0:
                    kernel[int(yi), int(xi)] += w
    kernel = kernel / kernel.sum()
    return kernel.to(device=device, dtype=dtype)


def motion_blur_variant(base: torch.Tensor, sample_id: int, image_mean: tuple[float, ...],
                         image_std: tuple[float, ...]) -> tuple[torch.Tensor, dict]:
    """Directional motion blur, applied in ~[0,1] pixel space -- NOT on the
    already-normalized tensor directly. Blur is a pixel-domain optical
    phenomenon (light integrating across a moving sensor during exposure);
    convolving the per-channel-rescaled normalized tensor instead would make
    the same kernel blend channels in the wrong proportions, since
    normalization rescales each channel by a different std. So: de-normalize
    (invert the processor's `(x - mean) / std`), convolve, re-normalize with
    the same mean/std -- same processor.image_mean/image_std this repo
    already uses everywhere else for this tensor's normalization contract.

    Computed in fp32 throughout (base is typically fp16): F.conv2d on fp16
    CPU tensors isn't implemented, and fp16 would also lose precision in the
    kernel's small bilinear-splat weights. Only the final result is cast
    back to `base`'s dtype, matching baseline.py's own pattern of computing
    a numerically-sensitive step in fp32 and casting the input/output at the
    boundary (see run_pose()'s heatmap decode there).
    """
    gen = torch.Generator(device="cpu").manual_seed(MOTION_BLUR_SEED_BASE + sample_id)
    lo, hi = MOTION_BLUR_ANGLE_RANGE_DEG
    angle_deg = lo + (hi - lo) * torch.rand((), generator=gen).item()
    lo, hi = MOTION_BLUR_LENGTH_RANGE_PX
    length_px = lo + (hi - lo) * torch.rand((), generator=gen).item()

    device = base.device
    mean_t = torch.tensor(image_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(image_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    pixel_01 = base.float() * std_t + mean_t  # invert VitPoseImageProcessor's (x - mean) / std
    kernel = _line_kernel(angle_deg, length_px, device, torch.float32)
    channels = pixel_01.shape[1]
    weight = kernel.expand(channels, 1, *kernel.shape).contiguous()  # depthwise: same PSF/channel
    pad = kernel.shape[-1] // 2
    padded = F.pad(pixel_01, (pad, pad, pad, pad), mode="replicate")  # avoid a zero-pad dark halo
    blurred = F.conv2d(padded, weight, groups=channels)

    variant = (blurred - mean_t) / std_t
    params = {"seed": MOTION_BLUR_SEED_BASE + sample_id, "angle_deg": angle_deg,
              "length_px": length_px, "kernel_size": kernel.shape[-1]}
    return variant.to(base.dtype), params


def make_sample(base_crop: torch.Tensor, sample_id: int, family: str,
                 image_mean: tuple[float, ...], image_std: tuple[float, ...]
                 ) -> tuple[torch.Tensor, dict]:
    if family == "classic_photometric_spatial":
        return classic_variant(base_crop, sample_id)
    if family == "motion_blur":
        return motion_blur_variant(base_crop, sample_id, image_mean, image_std)
    raise ValueError(f"unknown augmentation family: {family!r}")


@torch.no_grad()
def run_forward(model, pixel_values: torch.Tensor, dataset_index: int, device: str) -> torch.Tensor:
    kwargs = {"dataset_index": torch.full((1,), dataset_index, dtype=torch.long, device=device)}
    outputs = model(pixel_values=pixel_values, **kwargs)
    return outputs.heatmaps.detach().clone()


def run_stable(model, pixel_values: torch.Tensor, dataset_index: int, device: str,
               label: str) -> torch.Tensor:
    """N_VERIFY_RUNS-bit-identical forward pass -- same discipline as
    golden/build_distinct_batch.py's run_single()/verification loop, applied
    here per this script's own module docstring rather than assumed to
    carry over automatically. Refuses (does not silently accept) any
    fixture whose forward pass isn't deterministic under pinned cudnn
    settings."""
    runs = [run_forward(model, pixel_values, dataset_index, device).cpu() for _ in range(N_VERIFY_RUNS)]
    ref = runs[0]
    for r in runs[1:]:
        if not torch.equal(r, ref):
            raise SystemExit(
                f"[build_calibration_corpus] {label} is NOT bit-stable across "
                f"{N_VERIFY_RUNS} runs -- treating as a build failure, not saving an "
                f"untrustworthy calibration sample.")
    return ref


def mean_keypoint_shift_px(processor, base_heatmap: torch.Tensor, sample_heatmap: torch.Tensor,
                            box_xywh: list[float]) -> float:
    """Mean Euclidean per-joint pixel distance between the decoded poses of
    two heatmaps, via compare_golden.decode_keypoints (not a new decode
    implementation) -- the same "did the pose actually move" question
    compare_golden.py's keypoint_diff answers for backend-equivalence
    checks, reused here to answer "did this augmentation actually do
    anything," not to test numerical equivalence (see the manifest's
    "purpose" field)."""
    base_pose = decode_keypoints(processor, base_heatmap, box_xywh)
    sample_pose = decode_keypoints(processor, sample_heatmap, box_xywh)
    base_kpts = np.array(base_pose["keypoints"], dtype=np.float64)
    sample_kpts = np.array(sample_pose["keypoints"], dtype=np.float64)
    return float(np.linalg.norm(sample_kpts - base_kpts, axis=1).mean())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    checkpoint = resolve_checkpoint(None)
    print(f"[build_calibration_corpus] loading ViTPose++-L from {checkpoint} ({device}, "
          f"{dtype})")
    processor, model = load_pose_model(checkpoint, device, dtype)
    model.eval()

    base_crop_path = GOLDEN_DIR / "person_crop.npy"
    base_crop_np = np.load(base_crop_path)
    if base_crop_np.ndim == 3:  # tolerate a saved (3, 256, 192) as well as (1, 3, 256, 192)
        base_crop_np = base_crop_np[None]
    base_crop = torch.from_numpy(base_crop_np).to(device=device, dtype=dtype)
    expected_shape = (1, 3, *RESOLUTION)
    if tuple(base_crop.shape) != expected_shape:
        raise SystemExit(
            f"[build_calibration_corpus] REFUSING to build: {base_crop_path} has shape "
            f"{tuple(base_crop.shape)}, expected {expected_shape}. This script's "
            f"augmentations (roll shifts, motion-blur kernel padding) assume this exact "
            f"resolution -- re-check golden/person_crop.npy before proceeding.")

    env_golden = json.loads((GOLDEN_DIR / "env.json").read_text())
    box_xywh = env_golden["box_xywh"]
    image_mean, image_std = processor.image_mean, processor.image_std
    source_image_sha256 = sha256_file(base_crop_path)

    # Never silently patch/backfill a stale corpus -- delete and regenerate,
    # matching this repo's build_engine.py-style "refuse and delete-then-
    # rebuild" discipline for any artifact that might be self-inconsistent
    # (e.g. a previous partial run leaving stale sample_*.npy files around
    # that a new, possibly-different NUM_SAMPLES run wouldn't overwrite).
    stale = sorted(SAMPLES_DIR.glob("sample_*.npy")) if SAMPLES_DIR.is_dir() else []
    if stale or MANIFEST_PATH.exists():
        print(f"[build_calibration_corpus] clearing {len(stale)} stale sample file(s) and any "
              f"existing manifest before regenerating...")
        for f in stale:
            f.unlink()
        MANIFEST_PATH.unlink(missing_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    try:
        print(f"[build_calibration_corpus] computing per-expert base-crop reference heatmaps "
              f"({NUM_EXPERTS} experts x {N_VERIFY_RUNS} verify runs)...")
        base_heatmaps = {
            expert: run_stable(model, base_crop, expert, device, label=f"base crop @ expert {expert}")
            for expert in range(NUM_EXPERTS)
        }

        samples_meta = []
        per_expert_counts = {str(i): 0 for i in range(NUM_EXPERTS)}
        seen_sha256: dict[str, int] = {}
        for sample_id in range(NUM_SAMPLES):
            dataset_index = sample_id % NUM_EXPERTS
            family = "classic_photometric_spatial" if sample_id % 2 == 0 else "motion_blur"
            variant, aug_params = make_sample(base_crop, sample_id, family, image_mean, image_std)

            heatmap = run_stable(model, variant, dataset_index, device,
                                  label=f"sample {sample_id} (family={family}, expert={dataset_index})")
            shift_px = mean_keypoint_shift_px(processor, base_heatmaps[dataset_index], heatmap, box_xywh)
            if shift_px < MIN_KEYPOINT_SHIFT_PX:
                raise SystemExit(
                    f"[build_calibration_corpus] sample {sample_id} (family={family}, "
                    f"dataset_index={dataset_index}) shifted the decoded pose by only "
                    f"{shift_px:.4f}px relative to the unperturbed base crop run through the "
                    f"SAME expert -- below the {MIN_KEYPOINT_SHIFT_PX}px distinctness floor. "
                    f"Treating this as a degenerate, functionally-inert augmentation and "
                    f"refusing to save it, the same way golden/build_distinct_batch.py refuses "
                    f"a non-distinct slot -- a calibration corpus with a sample that doesn't "
                    f"actually probe the model differently isn't worth the sample-count "
                    f"credit it would claim in per_expert_counts.")

            variant_np = variant[0].cpu().numpy().astype(np.float16)  # drop batch dim; match
                                                                        # person_crop.npy's on-disk dtype
            sample_path = SAMPLES_DIR / f"sample_{sample_id:04d}.npy"
            np.save(sample_path, variant_np)
            sample_sha256 = sha256_file(sample_path)
            if sample_sha256 in seen_sha256:
                raise SystemExit(
                    f"[build_calibration_corpus] sample {sample_id} is byte-identical (sha256 "
                    f"{sample_sha256}) to sample {seen_sha256[sample_sha256]} -- a seed or RNG "
                    f"mistake collapsed two supposedly-distinct samples into one. Refusing to "
                    f"save a corpus with a hidden duplicate.")
            seen_sha256[sample_sha256] = sample_id

            samples_meta.append({
                "sample_id": sample_id,
                "path": f"calibration/samples/sample_{sample_id:04d}.npy",
                "dataset_index": dataset_index,
                "sha256": sample_sha256,
                "augmentation_family": family,
                "augmentation_params": aug_params,
                "keypoint_shift_px": shift_px,
            })
            per_expert_counts[str(dataset_index)] += 1
            print(f"[build_calibration_corpus] sample {sample_id:3d}/{NUM_SAMPLES}  "
                  f"expert={dataset_index}  family={family:<24s}  shift={shift_px:7.3f}px  OK")
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark

    short = [i for i, c in per_expert_counts.items() if c < MIN_SAMPLES_PER_EXPERT]
    if short:
        raise SystemExit(
            f"[build_calibration_corpus] REFUSING to write manifest.json: expert(s) {short} "
            f"got fewer than MIN_SAMPLES_PER_EXPERT={MIN_SAMPLES_PER_EXPERT} samples "
            f"({per_expert_counts}) -- the round-robin assignment should make this "
            f"unreachable for NUM_SAMPLES={NUM_SAMPLES}, NUM_EXPERTS={NUM_EXPERTS}; if it "
            f"triggered, something upstream (a refused-and-skipped sample?) broke the "
            f"round-robin invariant this gate exists to catch.")

    content_source_note = (
        "This repo has exactly one real source photo (samples/sample.jpg, gitignored) "
        "reduced to one detected+preprocessed crop (golden/person_crop.npy). No public pose "
        "dataset or private footage is integrated yet. This corpus is entirely synthetic "
        "augmentation of that one crop, explicitly a first-pass placeholder pending real "
        "data (public dataset crops and/or the user's own private sports footage). If/when "
        "private footage is added, the raw image/video files MUST be gitignored (follow the "
        "existing samples/ and checkpoints/ .gitignore precedent) and this field must be "
        "updated to document the real source, sample count, and sampling strategy -- never "
        "commit private footage itself."
    )
    moe_coverage_policy_note = (
        "TensorRT's EngineInspector (DETAILED profiling verbosity), run against the built "
        "engine engines/vitpose_b1_fp16.engine and dumped to "
        "results/tensorrt/fp16_layer_info_full.json (180 layers total), shows the backbone "
        "MLP's 6-expert MoE routing (VitPoseNaiveMoe.forward in "
        "transformers/models/vitpose_backbone/modeling_vitpose_backbone.py: each expert's "
        "Linear runs on the full hidden_state unconditionally, then is multiplied by a "
        "(dataset_index==i) mask and summed) is COMPLETELY FUSED by "
        "builder_optimization_level=3 -- e.g. layer 10 is a 'gemm' layer fusing 7 MatMuls "
        "(node_MatMul_92 + all 6 experts' MatMuls) into one kernel, and layer 12 is a 'kgen' "
        "layer (__myl_MulMulMulMulMulMulAddAddAddAddAddAddConcAdd...) fusing the six "
        "per-expert masks and their summation into one opaque kernel, repeating across all "
        "24 transformer blocks. Zero semantic layer names survive (grepping the full dump "
        "for Gather/Select/Where/expert-named layers: zero matches). CONCLUSION: per-expert "
        "calibration quality is NOT auditable at the tensor level after the engine is built. "
        "This manifest's per_expert_counts gate (intended to be enforced by "
        "calibration/dataset.py -- not yet implemented as of this corpus build; this note "
        "records the design intent, not a claim that enforcement code already exists) is "
        "therefore the ONLY preventive control available before engine build; a downstream "
        "accuracy check split by dataset_index (Gate C/D, not this corpus-generation step) "
        "is the only way to actually catch a badly-calibrated expert."
    )

    manifest = {
        "model": "ViTPose++-L",
        "purpose": "TensorRT INT8 entropy calibration corpus (Stage 6 Gate A) -- NOT a "
                   "validation/accuracy fixture. Do not use for numerical-equivalence testing.",
        "num_experts": NUM_EXPERTS,
        "min_samples_per_expert": MIN_SAMPLES_PER_EXPERT,
        "num_samples": len(samples_meta),
        "resolution": list(RESOLUTION),
        "content_source": "synthetic_augmentation_of_single_real_crop",
        "content_source_note": content_source_note,
        "source_image_sha256": source_image_sha256,
        "moe_coverage_policy_note": moe_coverage_policy_note,
        "augmentation_families": ["classic_photometric_spatial", "motion_blur"],
        "per_expert_counts": per_expert_counts,
        "samples": samples_meta,
        "env": env_fingerprint(device),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[build_calibration_corpus] wrote {len(samples_meta)} samples to {SAMPLES_DIR} and "
          f"{MANIFEST_PATH}")
    print(f"[build_calibration_corpus] per_expert_counts: {per_expert_counts}")


if __name__ == "__main__":
    main()
