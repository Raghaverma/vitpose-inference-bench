# ViTPose++-L on an NVIDIA L4: how far can inference be optimized?

**Research question:** how far can ViTPose++-L inference be optimized on an
NVIDIA L4 while preserving pose accuracy?

The plan is PyTorch baseline → ONNX → TensorRT (FP16/INT8) → video pipeline,
each stage gated on the previous one being correct, not just fast. This repo
is built incrementally, stage by stage; only Stage 0 exists so far.

## Stage 0 — baseline (this stage)

`baseline.py` loads ViTPose++-L from a local checkpoint, runs it on a single
image, sanity-checks the output, then benchmarks latency / FPS / peak VRAM
on repeated forward passes.

ViTPose is top-down, so a YOLO detector supplies the person box; only the
ViTPose forward pass itself is timed — the detector is just how this stage
gets a realistic crop.

### Getting the checkpoint

The checkpoint is not in this repo (see `.gitignore`). Download the public
ViTPose++-L weights from Hugging Face:

```bash
huggingface-cli download usyd-community/vitpose-plus-large \
    --local-dir checkpoints/vitpose-plus-large
```

If you skip this, `baseline.py` falls back to the same hub id directly
(`transformers` will download and cache it on first use).

You'll also need person-detector weights (any YOLOv8 `.pt` works) at
`checkpoints/yolov8s.pt`, or pass `--detector path/to/weights.pt`. Without
either, ultralytics will download its own default `yolov8s.pt`.

### Running it

```bash
pip install -r requirements.txt
python baseline.py --image path/to/a/photo/with/a/person.jpg
```

Outputs land in `results/`:
- `baseline_sample.jpg` — the image with detected keypoints/skeleton drawn on it
- `baseline_report.json` — checkpoint/device/dtype, the decoded keypoints, and
  the benchmark stats (mean/p50/p95 latency, FPS, peak VRAM)

### What "correct" means here

`verify_output()` in `baseline.py` checks hard invariants on the decoded
keypoints (finite values, scores in a sane range, coordinates roughly inside
the image) — enough to catch a broken weight port, dtype cast, or a
transposed heatmap↔image coordinate mapping. It is not a pose-quality/accuracy
metric; COCO-style accuracy evaluation is Stage 1.

## Roadmap

- **Stage 1** — PyTorch baseline: accuracy (COCO val) and latency/VRAM in more
  depth than Stage 0's smoke test.
- **Stage 2** — export to ONNX, verify numerically against PyTorch, run under
  ONNX Runtime (CUDA EP).
- **Stage 3** — ONNX → TensorRT, FP16 and INT8 (with calibration).
- **Stage 4** — video pipeline: batching, an async pipeline, GPU profiling,
  and optimization based on what profiling shows.

## Repo conventions

Checkpoints, exported models, and raw result dumps are not committed —
see `.gitignore`. Everything under `checkpoints/` must be obtained locally
per the instructions above.
