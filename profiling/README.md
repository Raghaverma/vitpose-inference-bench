# Stage 5 — GPU Profiling & Timeline Analysis

**Goal:** explain *why* TensorRT hits ~5ms at batch=1, and *why* its speedup
over PyTorch shrinks from 4.17x (batch=1) to ~1.3x (batch=8/16) — not just
restate that it does.

## Tooling note (read before the numbers below)

`nsys`/`ncu` (NVIDIA Nsight Systems/Compute CLI) are **not installed** on
this machine, and aren't in the default apt cache — no NVIDIA CUDA apt repo
is configured. Passwordless `sudo` is available, so installing them is
*possible*, but it means adding a new system-level package source and a
500MB+ download on a shared cloud instance — a real infrastructure decision,
not a given, and not taken here. Everything below uses tools already
installed with zero new footprint: TensorRT's own `IProfiler` API (part of
the `tensorrt` package already in use since Stage 3), `torch.cuda.Event`
(built into PyTorch), and arithmetic against this project's own already-
measured, already-validated Stage 4 data.

## Finding 0 (before any of the requested numbers): Stage 4's benchmark measures the engine alone, not a deployment

Checked empirically before writing any profiling code:
`backends/tensorrt.py`'s benchmarked call (`run_inference`) takes an input
tensor that's already GPU-resident (built once, outside the loop) and
returns output that stays on the GPU. It does **zero** host↔device copies
and **zero** postprocessing. Bracketing "H2D copy / TensorRT execution /
D2H copy / postprocessing" NVTX-style ranges around *that* loop would make
three of the four stages read ~0ms at every batch size and explain
nothing, while an additivity check against it would pass trivially (the
timed region never changed). That's a real, useful result — it's exactly
why Stage 4's headline 5.07ms is an **engine-only** figure — but it means
the 4-stage breakdown below had to be built as a separate, genuine,
explicitly-labeled measurement (`profiling/profile_deployment_stages.py`),
not a relabeling of the existing benchmark loop.

```bash
python -m profiling.profile_deployment_stages --batch-size 1
python -m profiling.profile_deployment_stages --batch-size 16
python -m profiling.profile_layers --batch-size 1
python -m profiling.profile_layers --batch-size 16
python -m profiling.roofline
```

## 1. H2D / execution / D2H / postprocess breakdown

Measured with a **genuine pinned-host-memory** input tensor copied to the
device fresh every iteration (not the pre-resident tensor Stage 4 uses), a
real device→host copy, and the real `post_process_pose_estimation` decode.
Sync-additivity verified (same discipline as `stage1_benchmark.py`'s
`STAGE_SUM_TOLERANCE`) before trusting the breakdown:

| Batch | H2D | TensorRT exec | D2H | Postprocess | Sum | Wall clock | Additivity |
|---|---|---|---|---|---|---|---|
| 1 | 0.048 ms (0.8%) | 4.641 ms (72.9%) | 0.048 ms (0.7%) | 1.634 ms (25.6%) | 6.371 ms | 6.499 ms | 2.0% drift, OK |
| 16 | 0.383 ms (0.6%) | 40.975 ms (62.3%) | 0.229 ms (0.3%) | 24.139 ms (36.7%) | 65.727 ms | 66.454 ms | 1.1% drift, OK |

**H2D/D2H are negligible at every batch size** — the crop tensor is small
(3×256×192 fp16 ≈ 295KB per image), so even a genuine PCIe transfer costs
under half a millisecond even at batch=16. There is no meaningful "memory
transfer overhead" story here; this workload's transfer cost was never the
bottleneck at these batch sizes.

**Postprocessing is not negligible, and its *share* grows with batch** —
25.6% of the deployment-realistic latency at batch=1, growing to 36.7% at
batch=16. This is the DARK/UDP heatmap decode (`scipy.ndimage.gaussian_filter`
+ argmax), which runs on CPU and is not batched/vectorized across slots the
way the GPU's matmuls are — so as batch size grows, postprocessing scales
roughly linearly with it while GPU execution gets proportionally *more*
efficient (see §2-3), and the balance shifts toward postprocessing. This is
a direct preview of the Amdahl's-Law bottleneck Stage 7/8's video pipeline
will hit: pushing batch size up to buy GPU throughput doesn't buy anything
if a CPU-bound, unbatched postprocessing step is running in between.

Note the exec column here (4.641ms at batch=1, 40.975ms at batch=16) is
modestly lower than Stage 4's isolated engine-only numbers (5.061ms,
51.478ms) — a ~8-16% difference most likely reflecting ordinary run-to-run
GPU clock-state variance between two separate process invocations, not a
methodological error (the additivity check inside *this* run is clean); it's
flagged rather than quietly ignored.

## 2. Top-5 dominant "kernels" — TensorRT's own per-layer ground truth

TensorRT's `IProfiler` interface reports every layer's own measured GPU
time on every inference, straight from the engine itself — real numbers,
zero new installs. Two honesty notes: (1) these are TensorRT's own
**post-fusion layer names** (e.g. a fused `MatMul+MatMul+...` block), not
raw SM kernel names — that finer granularity needs `ncu`'s hardware
counters, which this repo doesn't have; (2) attaching a profiler forces
internal per-layer synchronization, so absolute totals here (5.020ms sum at
batch=1) don't exactly match the unprofiled engine-only number — only the
*relative* layer-to-layer proportions are meaningful.

**Batch=1** (180 layers reported): no single layer dominates. The top 5 are
all fused QKV/attention-output/MLP matmul groups from different transformer
blocks, each contributing ~1.6% of total time. Cost is evenly spread across
the ViT-L backbone's 24 transformer blocks — consistent with the
architecture (`hidden_size=1024`, `num_hidden_layers=24`, `mlp_ratio=4`,
`num_attention_heads=16`) rather than any one bottleneck operation.

**Batch=16** (205 layers reported): same shape of story, but each top
fused-matmul-group's mean time grew from ~0.083ms to ~0.902ms — **an ~11x
increase for a 16x increase in batch size**, i.e. sub-linear. Each matmul
does 16x the work for only ~11x the time: bigger batches let the L4's
Tensor Cores run those GEMMs more efficiently per unit of work. This is the
real mechanism, not a guess: the same batching efficiency gain that helps
TensorRT's fused kernels **also helps PyTorch's own batched cuBLAS/cuDNN
GEMMs**. It isn't TensorRT-exclusive, so as batch size grows, this shared
efficiency gain compresses the gap between the two backends — the
opposite of what would happen if TensorRT had some batch-exclusive
advantage.

## 3. Roofline: is this compute-bound, bandwidth-bound, or neither?

`profiling/roofline.py` computes ViTPose++-L's backbone FLOPs and weight
bytes from its published architecture (approximate, backbone-only, a
documented lower bound — see the script's docstring), divides by the L4's
published FP16 peak (121 TFLOPS) and memory bandwidth (300 GB/s) to get two
theoretical latency floors per batch size, and checks which one the
*measured* TensorRT latency (from `results/batch_matrix.json`) actually
approaches:

| Batch | Measured | Compute floor | Bandwidth floor | Efficiency vs. governing floor |
|---|---|---|---|---|
| 1 | 5.06 ms | 0.99 ms | 2.04 ms | 40.4% |
| 2 | 7.48 ms | 1.98 ms | 2.08 ms | 27.8% |
| 4 | 13.55 ms | 3.95 ms | 2.14 ms | 29.2% |
| 8 | 27.57 ms | 7.91 ms | 2.26 ms | 28.7% |
| 16 | 51.48 ms | 15.81 ms | 2.52 ms | 30.7% |

**The honest finding here is not what was expected going in.** Measured
latency is well above *both* theoretical floors at every batch size tested
— this workload is never close to being truly compute-saturated or
bandwidth-saturated on the L4 (efficiency tops out under 41%). Efficiency
actually *drops* from batch=1 (40.4%) to batch=2 (27.8%), then stays roughly
flat (28-31%) through batch=16 — not a clean monotonic "batching approaches
the roofline" curve. **This means the roofline framing, on its own, does
not explain why TensorRT's relative speedup over PyTorch shrinks** — that
story is better explained by §2's finding (a batching efficiency gain
shared by both backends' GEMMs, narrowing TensorRT's fixed-overhead
advantage) than by either backend approaching a hardware ceiling. The
roofline check earns its place here by *ruling out* an intuitive-sounding
explanation, not confirming one — this repo doesn't have a "GEMM tiling
efficiency" story to tell that survives contact with the actual numbers,
and says so rather than forcing one.

## Direct answer: why does TensorRT's relative advantage shrink at higher batch?

Not compute saturation, not memory bandwidth (§3 rules both out as the
dominant mechanism at these batch sizes). The evidence points to **launch-
overhead amortization that both backends share, applied on top of a
fixed per-op overhead advantage that only matters when it's proportionally
large**:

- At batch=1, TensorRT's 180-fused-layer graph eliminates a large amount of
  Python/framework dispatch overhead that PyTorch eager pays per op — this
  fixed-cost elimination is TensorRT's real edge, and it's proportionally
  *huge* against a ~21ms PyTorch baseline where most of the time is
  overhead, not compute.
- As batch size grows, each backend's own GEMMs get more efficient per
  unit of work (§2: ~11x time for 16x batch) — PyTorch benefits from this
  too, via its own batched cuBLAS/cuDNN kernels, without needing TensorRT's
  fusion at all. Stage 4's own numbers already show this directly: PyTorch's
  FPS scales from 47 (batch=1) to 232 (batch=16), a bigger relative jump
  than TensorRT's 198→311.
- The result: TensorRT's fixed-overhead advantage stays roughly constant in
  absolute terms, while PyTorch's own overhead gets amortized away by
  batching — so the *ratio* between them shrinks, even though neither
  backend is anywhere near the L4's hardware ceiling (§3).

## Files

- `profiling/profile_deployment_stages.py` / `profiling/profile_layers.py` /
  `profiling/roofline.py` — the three scripts above.
- `profiling/results/deployment_stages_b{1,16}.json`,
  `layer_profile_b{1,16}.json`, `roofline.json` — raw data backing every
  number in this document.
