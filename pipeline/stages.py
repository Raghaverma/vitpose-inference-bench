"""One implementation of every video-pipeline stage, shared by Stage 7's synchronous pipeline
(pipeline/sync_pipeline.py) and Stage 8's asynchronous one (pipeline/async_pipeline.py), so
the two differ only in how the stages are scheduled, never in what a stage computes.

Per frame: decode -> detect -> preprocess -> h2d -> pose -> d2h -> postprocess.

The CPU stages (person selection, HF preprocess, HF decode) live in pipeline/cpu_stages.py so
Stage 8's worker processes can import them without TensorRT or ultralytics; they're re-exported
here. Held fixed from Stages 0-1: the YOLOv8s detector (checkpoints/yolov8s.pt, person class,
conf 0.35, fp32), the HF preprocessing and the HF decoding. What changes against Stage 1's
single image: every person up to MAX_PERSONS instead of the largest one, since this is
multi-person footage, and a choice of pose backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline import DEFAULT_DETECTOR, REPO_ROOT, load_pose_model, resolve_checkpoint
from pipeline.cpu_stages import (DATASET_INDEX, DET_CONF, HEATMAP_SHAPE, MAX_PERSONS,  # noqa: F401 (re-exports)
                                 POSE_INPUT, crop_height_px, postprocess, preprocess, select_persons)
from pipeline.cpu_stages import load_processor as _load_processor

BACKENDS = ("trt-fp16", "pytorch", "trt-int8")


# ---- workload -----------------------------------------------------------------------------------

def default_jobs_dir() -> Path:
    return REPO_ROOT.parent / "AutoClipping" / "data" / "jobs"


def workload_videos(jobs_dir: Path, split: str = "val") -> list[dict]:
    """The Stage 6 held-out videos (calibration/real_manifest.json, `split`), located under an
    AutoClipping jobs dir and checked against the manifest's content fingerprints -- so a
    result file names footage by job id + fingerprint, never by a private path, and a rerun
    on different bytes refuses instead of silently measuring other footage."""
    from calibration.real_corpus import MANIFEST_PATH, content_fingerprint

    manifest = json.loads(MANIFEST_PATH.read_text())
    out = []
    for job_id, meta in sorted(manifest["videos"].items()):
        if split != "all" and meta["split"] != split:
            continue
        path = jobs_dir / job_id / "source.mp4"
        if not path.is_file():
            raise SystemExit(f"[pipeline] {path} not found -- pass --jobs-dir <AutoClipping data/jobs>")
        fp = content_fingerprint(path)
        if fp != meta["fingerprint"]:
            raise SystemExit(f"[pipeline] REFUSING: {job_id} fingerprint {fp} != manifest {meta['fingerprint']}")
        out.append({"job_id": job_id, "path": path, "fingerprint": fp})
    return out


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    return cap


def load_processor():
    return _load_processor(resolve_checkpoint(None))


# ---- detect -------------------------------------------------------------------------------------

def load_detector():
    from ultralytics import YOLO
    return YOLO(str(DEFAULT_DETECTOR))


def detect(detector, frame_bgr: np.ndarray, max_persons: int = MAX_PERSONS) -> tuple[np.ndarray, int, dict]:
    """(xywh boxes, persons detected before the cap, ultralytics' own pre/infer/post ms)."""
    result = detector(frame_bgr, verbose=False, conf=DET_CONF, classes=[0], device="cuda")[0]
    xyxy = result.boxes.xyxy.cpu().numpy()
    return select_persons(xyxy, max_persons), len(xyxy), dict(result.speed)


@torch.inference_mode()
def detect_frames(detector, frames: list[np.ndarray], max_persons: int = MAX_PERSONS) -> list[tuple[np.ndarray, int]]:
    """detect() without ultralytics' timing wrappers, for Stage 8. Its predictor brackets every
    stage with ops.Profile, whose timer calls torch.cuda.synchronize() -- a DEVICE-wide sync,
    which in a two-stream pipeline makes each detection wait for the pose stream's work. This
    calls the same predictor's own preprocess / inference / postprocess (the exact methods
    stream_inference runs, so the same letterbox, model and NMS), whose only syncs are the
    current-stream ones their data flow needs. detect() must have run once first, to build the
    predictor with this pipeline's arguments; pipeline/async_pipeline.py checks the boxes against
    Stage 7's, which came from detect(), bit for bit."""
    p = detector.predictor
    if p is None or p.args.conf != DET_CONF or list(p.args.classes) != [0]:
        raise RuntimeError("call stages.detect() once before detect_frames() to set up the predictor")
    with p._lock:
        p.batch = ([""] * len(frames), frames, [""] * len(frames))
        im = p.preprocess(frames)
        results = p.postprocess(p.inference(im), im, frames)
    out = []
    for r in results:
        xyxy = r.boxes.xyxy.cpu().numpy()
        out.append((select_persons(xyxy, max_persons), len(xyxy)))
    return out


# ---- pose backends ------------------------------------------------------------------------------
# Each backend splits a forward into upload (pinned host -> device), forward and download
# (device -> pinned host), all on its own CUDA stream and none of them host-synchronizing
# except download, so a caller can time the three separately (with explicit syncs) or run
# them back to back.

class TorchPose:
    """PyTorch FP16 ViTPose++-L (the Stage 0 model), any batch size."""

    name = "pytorch"

    def __init__(self, max_batch: int = MAX_PERSONS):
        _, self.model = load_pose_model(resolve_checkpoint(None), "cuda", torch.float16)
        self.stream = torch.cuda.Stream()
        self.host_in = torch.empty((max_batch, *POSE_INPUT), dtype=torch.float16).pin_memory()
        self.host_out = torch.empty((max_batch, *HEATMAP_SHAPE), dtype=torch.float16).pin_memory()
        self.dataset_index = torch.full((max_batch,), DATASET_INDEX, dtype=torch.int64, device="cuda")
        self.max_batch = max_batch

    def upload(self, pixel_values: np.ndarray):
        n = len(pixel_values)
        self.host_in[:n].numpy()[...] = pixel_values
        with torch.cuda.stream(self.stream):
            return self.host_in[:n].to("cuda", non_blocking=True)

    @torch.inference_mode()
    def forward(self, device_in: torch.Tensor) -> torch.Tensor:
        n = device_in.shape[0]
        with torch.cuda.stream(self.stream):
            return self.model(pixel_values=device_in, dataset_index=self.dataset_index[:n]).heatmaps

    def download(self, device_out: torch.Tensor) -> np.ndarray:
        n = device_out.shape[0]
        with torch.cuda.stream(self.stream):
            self.host_out[:n].copy_(device_out, non_blocking=True)
        self.stream.synchronize()
        return self.host_out[:n].numpy().copy()

    def describe(self) -> dict:
        return {"backend": self.name, "batching": "one forward per frame, batch = persons in frame"}


class TrtPose:
    """TensorRT ViTPose++-L from the Stage 4 (FP16) / Stage 6 (INT8) static-batch engines.
    The engines have fixed batch sizes, so a frame's n crops run on the smallest engine with
    batch >= n, pad slots zeroed (Stage 4 verified slots are independent; zeroing keeps the
    pad content, and so the output, the same every call). forward() goes through
    run_inference(), the Stage 3-6 validated call, which ends in a device sync; enqueue() is
    Stage 8's sync-free equivalent."""

    def __init__(self, precision: str, batch_sizes: tuple[int, ...] = (1, 2, 4)):
        from backends.tensorrt import load_engine, verify_manifest

        self.name = f"trt-{precision}"
        self.precision = precision
        self.stream = torch.cuda.Stream()
        self.engines, self.manifests = {}, {}
        prefix = "int8_engine" if precision == "int8" else "engine"
        for b in batch_sizes:
            sfx = "" if b == 1 else f"_b{b}"
            manifest = verify_manifest(REPO_ROOT / "results" / "tensorrt" / f"{prefix}_metadata{sfx}.json")
            if manifest["batch"] != b:
                raise SystemExit(f"[pipeline] {prefix}_metadata{sfx}.json is batch {manifest['batch']}, expected {b}")
            engine = load_engine(Path(manifest["engine_path"]))
            self.engines[b] = {
                "engine": engine,
                "context": engine.create_execution_context(),
                "input": torch.zeros((b, *POSE_INPUT), dtype=torch.float16, device="cuda"),
                "dataset_index": torch.full((b,), DATASET_INDEX, dtype=torch.int64, device="cuda"),
            }
            self.manifests[b] = {"engine_sha256": manifest["engine_sha256"], "batch": b}
        self.batch_sizes = sorted(self.engines)
        self.max_batch = self.batch_sizes[-1]
        self.host_in = torch.empty((self.max_batch, *POSE_INPUT), dtype=torch.float16).pin_memory()
        self.host_out = torch.empty((self.max_batch, *HEATMAP_SHAPE), dtype=torch.float16).pin_memory()

    def buffer_contexts(self, n_buffers: int) -> dict[int, list]:
        """n_buffers execution contexts per engine, for a caller with several batches in flight
        (Stage 8). Created once and shared by every caller: runs are sequential, and each context
        carries its own activation memory (5-68 MB per engine, Stage 4's workspace figures)."""
        cache = self.__dict__.setdefault("_buffer_contexts", {})
        for b, e in self.engines.items():
            ctxs = cache.setdefault(b, [])
            while len(ctxs) < n_buffers:
                ctxs.append(e["engine"].create_execution_context())
        return cache

    def engine_batch(self, n: int) -> int:
        for b in self.batch_sizes:
            if b >= n:
                return b
        raise ValueError(f"{n} crops > largest engine batch {self.max_batch}")

    def upload(self, pixel_values: np.ndarray):
        n = len(pixel_values)
        b = self.engine_batch(n)
        slot = self.engines[b]["input"]
        self.host_in[:n].numpy()[...] = pixel_values
        with torch.cuda.stream(self.stream):
            slot[:n].copy_(self.host_in[:n], non_blocking=True)
            if n < b:
                slot[n:].zero_()
        return (b, n)

    def forward(self, handle) -> torch.Tensor:
        from backends.tensorrt import run_inference

        b, n = handle
        e = self.engines[b]
        out = run_inference(e["engine"], e["context"],
                            {"pixel_values": e["input"], "dataset_index": e["dataset_index"]},
                            self.stream.cuda_stream)["heatmaps"]
        return out[:n]

    def enqueue(self, b: int, device_in: torch.Tensor, device_out: torch.Tensor, stream: torch.cuda.Stream,
                context=None) -> None:
        """Enqueue one forward of the batch-b engine on `stream`, reading device_in (b, 3, 256, 192)
        and writing device_out (b, 17, 64, 48), with no host or device sync -- the caller orders
        it with events. Same bindings as run_inference(), which also checks contiguity. A caller
        with several batches in flight passes one execution context per buffer, so no context is
        re-bound while an earlier enqueue on it may still be running."""
        e = self.engines[b]
        if not (device_in.is_contiguous() and device_out.is_contiguous()):
            raise ValueError("[pipeline] TensorRT bindings must be contiguous")
        ctx = context or e["context"]
        ctx.set_tensor_address("pixel_values", device_in.data_ptr())
        ctx.set_tensor_address("dataset_index", e["dataset_index"].data_ptr())
        ctx.set_tensor_address("heatmaps", device_out.data_ptr())
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"[pipeline] execute_async_v3 failed (batch {b})")

    def download(self, device_out: torch.Tensor) -> np.ndarray:
        n = device_out.shape[0]
        with torch.cuda.stream(self.stream):
            self.host_out[:n].copy_(device_out, non_blocking=True)
        self.stream.synchronize()
        return self.host_out[:n].numpy().copy()

    def describe(self) -> dict:
        return {"backend": self.name, "engines": [self.manifests[b] for b in self.batch_sizes],
                "batching": "one forward per frame on the smallest static engine >= persons in "
                            "frame, pad slots zeroed"}


def load_backend(name: str, batch_sizes: tuple[int, ...] = (1, 2, 4)):
    if name == "pytorch":
        return TorchPose(max_batch=max(batch_sizes))
    if name in ("trt-fp16", "trt-int8"):
        return TrtPose(name.split("-")[1], batch_sizes)
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")


def outputs_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()
