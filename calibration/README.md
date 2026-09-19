# Stage 6, Gate A — TensorRT INT8 calibration corpus

**Label this corpus explicitly wherever it's referenced downstream:
"Engineering calibration corpus — single-source, augmentation-diverse."**
Not "representative of the deployment distribution" — it isn't, and no
language in this repo (this file, `manifest.json`, or any script that
consumes it) should imply otherwise. It exists to validate the INT8
calibration *machinery* (per-expert coverage, entropy-histogram collection,
the TensorRT build path) against real activation statistics, not to make a
claim about production pose/lighting/scene diversity. See
[Source data](#source-data-synthetic-augmentation-of-one-real-crop-not-a-dataset)
below for exactly what that means -- this corpus is a placeholder pending a
held-out, multi-source pose-quality evaluation set (not yet built).

## What Gate A is, and isn't

Gate A builds and documents the entropy-calibration corpus a TensorRT INT8
build will read activation statistics from. That is the entire scope of
this stage.

Gate A explicitly does **not**:
- **Build the INT8 engine.** No `IInt8EntropyCalibrator2`, no
  `config.set_flag(trt.BuilderFlag.INT8)`, no `trtexec`/`build_engine.py`
  INT8 path exists yet. That's Gate B.
- **Claim anything about accuracy.** Nothing in this corpus or its manifest
  measures whether an INT8 engine built from it would match the FP16
  engine's output, on this repo's `compare_golden.py` thresholds or any
  other. That's Gate C (numerical equivalence) / Gate D (the
  `EngineInspector` layer-wise precision audit, following the same pattern
  Stage 3 already used for FP16 — see `results/tensorrt/engine_metadata.json`
  and the finding cited below).

`calibration/manifest.json`'s own `"purpose"` field states this directly:
*"TensorRT INT8 entropy calibration corpus (Stage 6 Gate A) -- NOT a
validation/accuracy fixture. Do not use for numerical-equivalence testing."*
A `keypoint_shift_px` recorded per sample (see below) measures only whether
an augmentation moved the decoded pose at all relative to the unperturbed
crop — it is a **distinctness check**, not an accuracy metric, and it is
not comparable to Stage 2/3's `compare_golden.py` equivalence thresholds
(those measure agreement between two backends on the *same* input; this
measures difference between two *inputs* on the same backend).

## The corpus, by the numbers

Pulled directly from `calibration/manifest.json` (96 sample entries,
1 manifest) — nothing below is hand-typed:

| Field | Value |
|---|---|
| `num_samples` | 96 |
| `resolution` | 256×192 (matches `golden/person_crop.npy`'s shape) |
| `num_experts` | 6 |
| `min_samples_per_expert` | 16 |
| `per_expert_counts` | `{0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16}` — exactly at the floor, for every expert |
| `augmentation_families` | `classic_photometric_spatial` (48 samples), `motion_blur` (48 samples) |
| on-disk sample dtype/shape | float16, `(3, 256, 192)` — already-normalized `pixel_values`, batch dim dropped |
| `keypoint_shift_px` range | 2.87px – 496.83px across all 96 samples, all above the 0.05px distinctness floor |
| sample sha256 collisions | 0 / 96 — every saved sample file is byte-distinct |
| total on-disk size | ~28MB (`calibration/samples/sample_0000.npy` … `sample_0095.npy`) |

`per_expert_counts` sitting exactly at `min_samples_per_expert` (not above
it) is expected, not a near-miss: `NUM_SAMPLES=96` and a round-robin
`dataset_index = sample_id % 6` assignment make 16/expert an exact,
deterministic outcome, not a statistical one.

**A real coverage gap, not hidden:** `NUM_EXPERTS=6` is divisible by the
family-alternation period (2), so `dataset_index` and
`augmentation_family` are perfectly correlated across the whole corpus —
experts 0/2/4 are exclusively `classic_photometric_spatial` and experts
1/3/5 exclusively `motion_blur`; no expert ever sees both families. This
falls directly out of implementing "`dataset_index = sample_id % 6`" and
"family alternates by `sample_id % 2`" exactly as specified — neither rule
is a bug — but it means each expert's 16 samples vary along only one
augmentation axis, not two. `build_calibration_corpus.py`'s module
docstring flags this in the same terms; a future corpus revision (e.g.
offsetting the family index by an odd stride) would need to fix it
deliberately, not assume round-robin coverage implies family-mixed
coverage per expert.

## Source data: synthetic augmentation of one real crop, not a dataset

This repo has exactly one real source photo (`samples/sample.jpg`,
gitignored), reduced to one detected+preprocessed crop
(`golden/person_crop.npy`). There is no public pose dataset and no real
sports footage integrated into this repo yet. **All 96 samples in this
corpus are synthetic perturbations of that single crop** — the manifest
records this plainly as `"content_source":
"synthetic_augmentation_of_single_real_crop"`, and `source_image_sha256`
(`68c2972b411e93bfd8b2d4b075c131f767e7f0caab99f12ca29c7ade07f7d3c7`, pinned
via `sha256_file` from `conversion/build_engine.py`) ties every sample back
to that one crop, byte-for-byte.

This is an explicit, known limitation of this first pass, not a hidden one:
a 96-sample corpus built from one photo's motion-blur/brightness/roll
perturbations does not represent the pose, lighting, or motion diversity of
real deployment footage. Entropy calibration statistics gathered from it
are calibration statistics for *this crop's augmented neighborhood*, not
for the deployment distribution the INT8 engine will actually see frames
from. See [Follow-on work](#follow-on-work-not-done-here) for what closes
this gap.

## MoE `dataset_index` coverage policy, and why it's enforced in the manifest, not the engine

ViTPose++-L's backbone MLP routes every token through a 6-expert MoE
(`VitPoseNaiveMoe.forward` in
`transformers/models/vitpose_backbone/modeling_vitpose_backbone.py`): each
of the 6 experts' `Linear` layers runs on the **full** `hidden_state`
unconditionally, then gets multiplied by a `(dataset_index == i)` mask and
summed. INT8 entropy calibration needs activation statistics from every
branch that will actually execute at inference time — a corpus that only
ever exercises expert 0 would leave experts 1–5's `Linear` layers
calibrated on whatever statistics leak through masked-out (zeroed) forward
passes, not real per-expert activation distributions. That is the entire
reason `per_expert_counts` and `min_samples_per_expert` exist as manifest
fields in the first place.

**We checked whether this is auditable after the engine is built, instead
of assuming either way.** TensorRT's `EngineInspector` (DETAILED profiling
verbosity) was run against the existing `engines/vitpose_b1_fp16.engine`
and dumped to `results/tensorrt/fp16_layer_info_full.json` (180 layers
total, committed). It shows `builder_optimization_level=3` **completely
fuses** the MoE block into opaque kernels: layer 10 is a `gemm` layer
fusing 7 MatMuls (`node_MatMul_92` — the shared branch — plus all 6
experts' MatMuls) into one kernel, and layer 12 is a `kgen` layer named
`__myl_MulMulMulMulMulMulAddAddAddAddAddAddConcAdd...` fusing the six
per-expert masks and their summation into one opaque kernel — this exact
pattern repeats across all 24 transformer blocks. Grepping the full dump
for `Gather`/`Select`/`Where`/expert-named layers returns **zero matches**;
no semantic per-expert layer names survive the build.

**Conclusion: per-expert calibration quality is not verifiable at the
tensor level once the engine is built.** There is no post-hoc inspection
that can tell you "expert 3's Linear was calibrated on too few samples" by
looking at the built INT8 engine — the tensor it lived in no longer exists
as a distinct, individually-quantizable object. That makes this manifest's
`per_expert_counts` gate the **only preventive control** available before
engine build. `calibration/dataset.py` is the enforcement point:
`CalibrationDataset.__init__` recomputes `per_expert_counts` from
`manifest["samples"]` itself (never trusts the manifest's own cached
field — a mismatch is its own `SystemExit`, independent of the floor
check) and refuses to load, naming the exact short expert(s), if any count
is under `min_samples_per_expert`. `build_calibration_corpus.py`'s
generation-time check is the first line of defense (currently unreachable
given `NUM_SAMPLES=96` / `NUM_EXPERTS=6`'s exact round-robin arithmetic,
but stops silently shipping a skewed corpus if that arithmetic ever
changes); `calibration/dataset.py` is the second, independent one, run
every time the corpus is loaded rather than only at generation time — see
[How to run it](#how-to-run-it) for both, plus the adversarial tests that
confirmed each one actually fires. A downstream accuracy check split by
`dataset_index` (Gate C/D, not Gate A's job) remains the only way to
actually *catch* a badly-calibrated expert after the fact — the manifest
gate can only prevent the input from being obviously wrong going in.

## Private-footage policy (for when real data replaces this placeholder)

This corpus is a first-pass placeholder pending real data — either public
pose-dataset crops or the user's own private sports footage. When either
lands:

- **Raw image/video files must be gitignored**, the same way this repo
  already treats `samples/` and `checkpoints/` (see `.gitignore`) — never
  commit a private photo or video file itself.
- **Only the manifest gets committed**: sample count, resolution, source
  description, pose/`dataset_index` distribution, and sampling strategy —
  the same shape of information `calibration/manifest.json` already
  records for the synthetic corpus, updated to describe the real source
  instead of `synthetic_augmentation_of_single_real_crop`.
- The manifest's `content_source` and `content_source_note` fields are the
  two fields that must change; `per_expert_counts` /
  `min_samples_per_expert` stay as the same enforced gate regardless of
  where the underlying frames came from.

## How to run it

```bash
source .venv/bin/activate && python3 -m calibration.build_calibration_corpus
```

Builds (or rebuilds) the corpus end-to-end. Deletes any existing
`calibration/samples/*.npy` and `calibration/manifest.json` first — never
patches an existing corpus in place, matching this repo's
delete-and-regenerate convention (`conversion/build_engine.py`'s manifest
pattern). Refuses (`SystemExit`) and stops the run on any of:
- a sample's forward pass isn't bit-identical across `N_VERIFY_RUNS=5`
  reruns under pinned cuDNN determinism;
- a sample's decoded pose shifts by less than `MIN_KEYPOINT_SHIFT_PX=0.05`
  relative to the unperturbed base crop run through the *same* expert
  (functionally-inert augmentation);
- two saved sample files hash to the same sha256 (hidden duplicate); or
- any expert ends up below `MIN_SAMPLES_PER_EXPERT=16` once all 96 samples
  are generated.

None of these fired on the actual run this corpus was built from.

```bash
source .venv/bin/activate && python3 -m calibration.dataset
```

The load-time gate an INT8 calibrator (Gate B, not built yet) would import:
`CalibrationDataset` recomputes `per_expert_counts` from the manifest's
`samples` list itself, refuses (`SystemExit`, naming the exact expert(s))
if any count disagrees with the manifest's cached field or falls below
`min_samples_per_expert`, and re-hashes every sample file against its
manifest-recorded sha256 before handing it out via `get_batch(index)` (the
`{"pixel_values": ..., "dataset_index": ...}` shape
`backends/tensorrt.py` already uses). Adversarially tested against two
broken `/tmp`-only manifests (never the real one) — an honestly-reported
short expert, and a manifest whose cached `per_expert_counts` *lies* about
an actually-short expert — and confirmed both raise `SystemExit` naming
expert 3 specifically; a clean run against the real manifest was checked as
a positive control so the gate isn't just refusing everything indiscriminately.

```bash
source .venv/bin/activate && python3 -m calibration.check_disjoint
```

A standalone re-check, independent of `build_calibration_corpus.py`,
that this corpus was never also used as a golden/validation fixture. Two
checks, both must be read together:
1. **Exact-bytes**: sha256 every file under `calibration/samples/` against
   every `golden/*.npy`'s raw array bytes — refuses on any overlap.
   Adversarially tested by planting a byte-identical copy of
   `golden/person_crop.npy` into a `/tmp`-only scratch corpus and
   confirming the script actually caught it before being trusted against
   the real corpus (which passes clean: 0 collisions).
2. **Provenance**: exact-byte disjointness passing does *not* mean these
   corpora probe independent real-world content. This check compares
   `calibration/manifest.json`'s `source_image_sha256` against
   `golden/person_crop.npy`'s actual hash — right now they **match**
   (both this corpus and every golden fixture trace back to the repo's one
   real photo), so the script prints a loud, non-fatal `WARNING` rather
   than silently passing: see [Source data](#source-data-synthetic-augmentation-of-one-real-crop-not-a-dataset)
   above for what closes this.

## Follow-on work (not done here)

- **Additional augmentation families.** Only `classic_photometric_spatial`
  and `motion_blur` are implemented. Glare, JPEG artifacts, occlusion
  patches, and geometric warps are not — each would need its own
  coverage-assertion and `compare_golden.py`-based keypoint-shift check
  before being trusted, the same discipline this corpus already applies to
  its two existing families.
- **Swapping the content source from synthetic-only to real footage** —
  public pose-dataset crops and/or the user's own private sports footage,
  per the private-footage policy above. This is the change that would
  actually close the "one photo, synthetically perturbed" gap flagged
  throughout this document.
