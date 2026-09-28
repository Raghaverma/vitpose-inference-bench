#!/usr/bin/env python3
"""Stage 6: the REAL-footage INT8 calibration + evaluation corpus.

    python -m calibration.real_corpus build --distill-dir <AutoClipping distill dataset>
    python -m calibration.real_corpus check

Why this exists next to Gate A's synthetic corpus
---------------------------------------------------
calibration/manifest.json (Gate A) is 96 synthetic perturbations of ONE photo.
It validated the calibration machinery, but it is not a basis for INT8 scales
or for an accuracy claim, and it hid the fact that the first INT8 engine was
broken: that engine was only ever compared against one golden crop. This
corpus is real cricket net footage -- person crops from the production
pipeline this repo's optimizations are for (AutoClipping) -- split into:

  - calib: 256 crops sampled from the TRAIN videos, used only to set INT8
    activation scales (conversion/quantize_onnx.py);
  - eval:  400 crops sampled from the held-out VAL videos, with their
    PyTorch FP16 reference heatmaps, used only to measure INT8 drift
    (tests/test_tensorrt_int8_equivalence.py, backends/tensorrt.py
    --precision int8).

Source: an AutoClipping `distill/build_dataset.py` output directory
(manifest.jsonl + shard_*.npz). Its crops are the deployed crop warp
(pose_codec, antialiased, straight from the source frame) around the
production person detector's boxes; this script copies those uint8 crops and
their crop transforms (center, scale_px) and nothing else -- the distill
teacher's flip-TTA heatmaps are NOT used as a reference here, since INT8 is
judged against the plain FP16 forward it replaces.

Split: BY VIDEO, never by frame (frames of one clip are near-duplicates, so
a per-frame split would leak). The rule is AutoClipping's own
`distill.data.video_split` (sha1 of the source path < 0.1 -> val), so this
corpus's eval videos are the distill student's val videos too. Because that
rule hashes a PATH, and byte-identical re-uploads under different job ids do
exist on that box, `build` also fingerprints every source video's content
(sha1 of its first 16 MiB + its size) and refuses if any train video shares
a fingerprint with any val video.

Private-footage policy (calibration/README.md): the crops are people in
private footage, so calibration/real/*.npz is gitignored. Only
calibration/real_manifest.json is committed -- counts per job id, split,
sampling seed, frame indices, content fingerprints and the sha256 of every
gitignored array file -- and the loader below refuses to hand out data whose
sha256 no longer matches it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from baseline import REPO_ROOT, load_pose_model, resolve_checkpoint
from conversion.build_engine import sha256_file

CORPUS_DIR = REPO_ROOT / "calibration" / "real"
MANIFEST_PATH = REPO_ROOT / "calibration" / "real_manifest.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
PADDING = 1.25          # ViTPose's box padding; scale_px = padded, aspect-matched box size
VAL_FRAC = 0.1          # AutoClipping distill.data.video_split default
SEED = 0
N_CALIB, N_EVAL = 256, 400


def video_split(video_path: str, val_frac: float = VAL_FRAC) -> str:
    """AutoClipping distill.data.video_split, verbatim rule."""
    h = int(hashlib.sha1(video_path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def content_fingerprint(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(16 << 20)
    return f"{hashlib.sha1(head).hexdigest()[:16]}-{path.stat().st_size}"


def to_pixel_values(crops_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR (N, 256, 192, 3) -> ImageNet-normalized fp16 (N, 3, 256, 192), the same
    rescale + normalize VitPoseImageProcessor applies."""
    rgb = crops_bgr[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    # astype() keeps the flipped/transposed strides: without this the array is not
    # C-contiguous, and a TensorRT binding reads its raw buffer as if it were -- every crop
    # arrived scrambled (48 px error) while PyTorch, which honours strides, was fine.
    return np.ascontiguousarray(((rgb - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float16))


def boxes_xywh(center: np.ndarray, scale_px: np.ndarray) -> np.ndarray:
    """Invert the crop transform to the box HF's post_process_pose_estimation expects: it
    re-derives exactly this (center, scale_px) from the box, since scale_px is already
    aspect-matched to 192:256."""
    wh = scale_px / PADDING
    return np.concatenate([center - wh / 2, wh], axis=1)


def build(distill_dir: Path) -> None:
    rows = [json.loads(line) for line in (distill_dir / "manifest.jsonl").read_text().splitlines() if line]
    by_video = collections.defaultdict(int)
    for r in rows:
        by_video[r["video"]] += 1

    fingerprints = {}
    for v in by_video:
        if not Path(v).is_file():
            raise SystemExit(f"[real_corpus] REFUSING to build: source video {v} (from "
                             f"{distill_dir}/manifest.jsonl) is not on disk, so the train/val "
                             f"content-disjointness check can't run.")
        fingerprints[v] = content_fingerprint(Path(v))
    split_of = {v: video_split(v) for v in by_video}
    train_fp = {fingerprints[v] for v in by_video if split_of[v] == "train"}
    leaked = sorted(v for v in by_video if split_of[v] == "val" and fingerprints[v] in train_fp)
    if leaked:
        raise SystemExit(f"[real_corpus] REFUSING to build: val video(s) {leaked} have the same "
                         f"content fingerprint as a train video (a re-upload under another job "
                         f"id) -- the eval set would not be held out.")

    train = [r for r in rows if split_of[r["video"]] == "train"]
    val = [r for r in rows if split_of[r["video"]] == "val"]
    rng = np.random.default_rng(SEED)
    calib = [train[i] for i in sorted(rng.choice(len(train), N_CALIB, replace=False))]
    evals = val if len(val) <= N_EVAL else [val[i] for i in sorted(rng.choice(len(val), N_EVAL, replace=False))]

    shards = {}
    def gather(sel):
        out = {"crops": [], "center": [], "scale_px": []}
        for r in sel:
            if r["shard"] not in shards:
                # Materialize each shard's arrays ONCE: indexing an NpzFile member
                # re-reads (and decompresses) the whole member on every access.
                with np.load(distill_dir / r["shard"]) as npz:
                    shards[r["shard"]] = {k: npz[k] for k in ("crops", "center", "scale_px")}
            z = shards[r["shard"]]
            out["crops"].append(z["crops"][r["row"]])
            out["center"].append(z["center"][r["row"]])
            out["scale_px"].append(z["scale_px"][r["row"]])
        return {k: np.stack(v) for k, v in out.items()}

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, sel in (("calib", calib), ("eval", evals)):
        arrays = gather(sel)
        arrays["job_id"] = np.array([Path(r["video"]).parent.name for r in sel])
        arrays["frame"] = np.array([r["frame"] for r in sel], dtype=np.int64)
        if name == "eval":
            arrays["ref_heatmaps"] = reference_heatmaps(to_pixel_values(arrays["crops"]))
        path = CORPUS_DIR / f"{name}.npz"
        np.savez(path, **arrays)
        splits[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "num_crops": len(sel),
            "per_job_counts": dict(sorted(collections.Counter(arrays["job_id"].tolist()).items())),
        }
        print(f"[real_corpus] {name}: {len(sel)} crops from {len(splits[name]['per_job_counts'])} "
              f"video(s) -> {path}")

    manifest = {
        "purpose": "Stage 6 INT8 calibration (calib) and held-out accuracy evaluation (eval) on real "
                   "cricket net footage -- see calibration/real_corpus.py",
        "content_source": "real_cricket_net_footage",
        "content_source_note": "Person crops from AutoClipping production jobs: the production person "
                               "detector's boxes, warped with AutoClipping's deployed crop warp "
                               "(pose_codec, antialiased, from the source frame) by its "
                               "distill/build_dataset.py. Private footage: arrays are gitignored.",
        "dataset_index": 0,
        "dataset_index_note": "COCO expert only -- the expert production runs. INT8 scales and "
                              "accuracy are validated for expert 0 and nothing else.",
        "resolution": [256, 192],
        "split_rule": f"by video: AutoClipping distill.data.video_split (sha1(path) < {VAL_FRAC} -> "
                      f"val), plus a content-fingerprint disjointness check",
        "sampling": f"numpy default_rng({SEED}): {N_CALIB} crops without replacement from train rows, "
                    f"then {N_EVAL} from val rows, each kept in manifest order",
        "videos": {Path(v).parent.name: {"split": split_of[v], "fingerprint": fingerprints[v],
                                         "crops_in_source_dataset": n}
                   for v, n in sorted(by_video.items())},
        "splits": splits,
        "eval_reference": "ref_heatmaps in eval.npz: PyTorch FP16 ViTPose++-L (baseline.load_pose_model, "
                          "cuDNN deterministic), dataset_index 0, batches of 16",
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[real_corpus] wrote {MANIFEST_PATH}")


def reference_heatmaps(pixel_values: np.ndarray) -> np.ndarray:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _, model = load_pose_model(resolve_checkpoint(None), "cuda", torch.float16)
    out = []
    with torch.inference_mode():
        for k in range(0, len(pixel_values), 16):
            x = torch.from_numpy(pixel_values[k:k + 16]).cuda()
            di = torch.zeros(len(x), dtype=torch.int64, device="cuda")
            out.append(model(pixel_values=x, dataset_index=di).heatmaps.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


class RealCropSet:
    """One split of the real corpus, sha256-verified against calibration/real_manifest.json."""

    def __init__(self, split: str, manifest_path: Path = MANIFEST_PATH):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"\n\nMissing {manifest_path}. Build the real corpus first:\n\n"
                                    f"    python -m calibration.real_corpus build --distill-dir <dir>\n")
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_sha256 = sha256_file(manifest_path)
        entry = self.manifest["splits"][split]
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"[real_corpus] {manifest_path} describes {path}, which is not on disk (the arrays are "
                f"gitignored private footage -- rebuild locally: python -m calibration.real_corpus "
                f"build --distill-dir <dir>).")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise SystemExit(f"[real_corpus] REFUSING to load {path}: sha256 {actual} != manifest "
                             f"{entry['sha256']}. Rebuild the corpus; never trust a mismatched file.")
        z = np.load(path)
        self.crops_bgr = z["crops"]
        self.center = z["center"]
        self.scale_px = z["scale_px"]
        self.job_id = z["job_id"]
        self.ref_heatmaps = z["ref_heatmaps"] if "ref_heatmaps" in z.files else None
        if len(self.crops_bgr) != entry["num_crops"]:
            raise SystemExit(f"[real_corpus] REFUSING to load {path}: {len(self.crops_bgr)} crops, "
                             f"manifest says {entry['num_crops']}.")
        self.pixel_values = to_pixel_values(self.crops_bgr)
        self.boxes_xywh = boxes_xywh(self.center, self.scale_px)

    def __len__(self) -> int:
        return len(self.crops_bgr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--distill-dir", type=Path, required=True,
                   help="An AutoClipping distill/build_dataset.py output dir (manifest.jsonl + shards).")
    sub.add_parser("check")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.distill_dir)
    for split in ("calib", "eval"):
        s = RealCropSet(split)
        print(f"[real_corpus] {split}: {len(s)} crops verified (sha256 matches manifest), "
              f"per job: {s.manifest['splits'][split]['per_job_counts']}")


if __name__ == "__main__":
    main()
