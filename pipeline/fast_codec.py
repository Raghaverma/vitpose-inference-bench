"""Stage 11: HF's crop warp and pose decode, re-implemented without HF's per-crop machinery.

Stages 7-9 run the crop warp and the pose decode through HF's VitPoseImageProcessor, which is
the numerical reference but costly: every call converts the whole frame to a tensor and back,
warps each crop with scipy's affine_transform one colour channel at a time, and decodes every
heatmap with ~17 scipy gaussian_filter calls in a Python loop. Stage 8 spreads that over 8
worker processes. This module computes the same numbers in a few batched tensor operations, so
they can run on the GPU (or vectorized on a CPU core):

- `GpuCropper.warp`: the crop warp for a whole micro-batch at once, straight from device-resident
  frames, in one Triton kernel (warp_reference states the same arithmetic as torch ops). It reproduces scipy's order-1 affine_transform exactly as HF calls it: sample
  coordinates from the same float64 inverse matrix (np.linalg.inv, bit-identical to HF's
  scipy.linalg.inv), bilinear taps accumulated in float64 in scipy's order, 0 for any sample
  outside [0, n-1] (scipy's mode='constant'), rounded half-up to uint8, then HF's own fused
  rescale + normalize in float32 and the pipeline's fp16 cast. The result is bit-identical to
  HF's on real frames (tests/test_fast_codec.py).
- `decode`: argmax + DARK + the inverse crop transform, batched in torch on the heatmaps' own
  device. The gaussian modulation reproduces scipy's gaussian_filter (symmetric 'reflect'
  padding, float64 accumulation in its symmetric-kernel order, float32 between the two passes).
  The log and the 2x2 solve use the device's own float math, so keypoints agree with HF's to
  ~1e-4 px rather than bit for bit.
- `warp_cv2`: cv2.warpAffine (INTER_LINEAR, 0 border) per crop, the CPU alternative. cv2's
  bilinear is fixed-point (1/32 px positions, 15-bit weights), so its crops differ from HF's
  by a grey level here and there; tests measure how much that moves a keypoint.

Only the crop warp and the pose decode change. The detector, the person selection, the model
and the box -> (center, scale) rule are the ones Stages 7-9 use.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.cpu_stages import HEATMAP_SHAPE, POSE_INPUT

IN_H, IN_W = POSE_INPUT[1], POSE_INPUT[2]
HM_H, HM_W = HEATMAP_SHAPE[1], HEATMAP_SHAPE[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)       # HF VitPoseImageProcessor's image_mean / image_std
IMAGENET_STD = (0.229, 0.224, 0.225)
RESCALE = 1 / 255
PADDING = 1.25
NORMALIZE_FACTOR = 200.0
DARK_KERNEL = 11                            # post_process_pose_estimation's default kernel_size
DARK_SIGMA = 0.8


# ---- box -> crop geometry ---------------------------------------------------------------------

def center_scale(boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized HF box_to_center_and_scale for (N, 4) xywh boxes: (center (N, 2), scale (N, 2))
    in HF's units (px / 200), float32, bit-identical to the per-box function."""
    b = np.asarray(boxes_xywh, np.float32).reshape(-1, 4)
    aspect = IN_W / IN_H
    center = np.stack([b[:, 0] + b[:, 2] * np.float32(0.5), b[:, 1] + b[:, 3] * np.float32(0.5)], axis=1)
    w, h = b[:, 2].copy(), b[:, 3].copy()
    wide, tall = w > aspect * h, w < aspect * h
    h[wide] = w[wide] * np.float32(1.0) / np.float32(aspect)
    w[tall] = h[tall] * np.float32(aspect)
    scale = np.stack([w / np.float32(NORMALIZE_FACTOR), h / np.float32(NORMALIZE_FACTOR)], axis=1)
    return center.astype(np.float32), (scale * np.float32(PADDING)).astype(np.float32)


def crop_height_px(boxes_xywh: np.ndarray) -> np.ndarray:
    """Vectorized pipeline.cpu_stages.crop_height_px (same float32 values)."""
    return center_scale(boxes_xywh)[1][:, 1] * np.float32(NORMALIZE_FACTOR)


def sample_maps(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(N, 4) float64 (row_step, row_offset, col_step, col_offset): output pixel (i, j) of crop n
    samples the source at row = row_offset + row_step * i, col = col_offset + col_step * j.

    HF builds a float32 forward matrix (get_warp_matrix, theta 0), inverts it in float64 with
    scipy.linalg.inv and hands scipy the inverse with rows and columns swapped. Built the same
    way here and inverted with np.linalg.inv, which gives bit-identical results."""
    n = len(center)
    size_input = center * np.float32(2.0)                       # get_warp_matrix(..., center * 2.0, ...)
    size_dst = np.array((IN_W, IN_H), np.float64) - 1.0
    size_target = scale * np.float32(NORMALIZE_FACTOR)
    sx = size_dst[0] / size_target[:, 0].astype(np.float64)
    sy = size_dst[1] / size_target[:, 1].astype(np.float64)
    cos, sin = np.float32(1.0), np.float32(0.0)                 # math.cos(0), math.sin(0)
    half = np.float32(0.5)
    m = np.zeros((n, 3, 3), np.float64)
    fwd = np.zeros((n, 2, 3), np.float32)                       # the float32 matrix HF builds
    fwd[:, 0, 0] = 1.0 * sx
    fwd[:, 0, 1] = -0.0 * sx
    # HF's bracketed terms are numpy float32 scalars times Python floats, so they are summed
    # in float32 and only then multiplied by the float64 scale.
    fwd[:, 0, 2] = sx * (-half * size_input[:, 0] * cos + half * size_input[:, 1] * sin
                         + half * size_target[:, 0]).astype(np.float64)
    fwd[:, 1, 0] = 0.0 * sy
    fwd[:, 1, 1] = 1.0 * sy
    fwd[:, 1, 2] = sy * (-half * size_input[:, 0] * sin - half * size_input[:, 1] * cos
                         + half * size_target[:, 1]).astype(np.float64)
    m[:, :2] = fwd
    m[:, 2, 2] = 1.0
    inv = np.linalg.inv(m)
    # scipy indexes (row, col): HF swaps the inverse's x/y entries. Theta is 0, so the
    # off-diagonal terms are exactly 0 and each axis is independent.
    if np.any(inv[:, 0, 1] != 0) or np.any(inv[:, 1, 0] != 0):
        raise AssertionError("expected an axis-aligned crop transform")
    return np.stack([inv[:, 1, 1], inv[:, 1, 2], inv[:, 0, 0], inv[:, 0, 2]], axis=1)


# ---- the crop warp on the GPU -------------------------------------------------------------------

class GpuCropper:
    """Crop warp + normalization for a micro-batch, from frames already on the device."""

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        # HF's fused rescale + normalize (TorchvisionBackend._fuse_mean_std_and_rescale_factor):
        # mean and std scaled by 1 / rescale_factor, in float32, then tvF.normalize.
        self.mean_flat = torch.tensor(IMAGENET_MEAN, device=self.device) * (1.0 / RESCALE)
        self.std_flat = torch.tensor(IMAGENET_STD, device=self.device) * (1.0 / RESCALE)
        self.mean, self.std = self.mean_flat.view(1, 3, 1, 1), self.std_flat.view(1, 3, 1, 1)
        self.i = torch.arange(IN_H, dtype=torch.float64, device=self.device)
        self.j = torch.arange(IN_W, dtype=torch.float64, device=self.device)

    def warp(self, frames: torch.Tensor, slots: torch.Tensor, maps: torch.Tensor,
             out: torch.Tensor | None = None) -> torch.Tensor:
        """frames: (S, H, W, 3) uint8 BGR on the device; slots: (N,) int64, the frame of each
        crop; maps: (N, 4) float64 from sample_maps(). Returns (N, 3, 256, 192) fp16 RGB,
        normalized, written into `out` if given. Enqueued on the current stream. One Triton
        kernel (warp_triton) when it can run, else warp_reference(); the two agree bit for bit."""
        if triton is not None and frames.is_contiguous() and maps.is_contiguous() and \
                (out is None or out.is_contiguous()):
            if out is None:
                out = torch.empty((len(maps), 3, IN_H, IN_W), dtype=torch.float16, device=frames.device)
            return warp_triton(self, frames, slots.contiguous(), maps, out)
        return self.warp_reference(frames, slots, maps, out)

    def warp_reference(self, frames: torch.Tensor, slots: torch.Tensor, maps: torch.Tensor,
                       out: torch.Tensor | None = None) -> torch.Tensor:
        """The crop warp as plain torch ops: the readable statement of the arithmetic, and what
        the tests check warp_triton against."""
        h, w = frames.shape[1], frames.shape[2]
        # Source coordinates, as scipy computes them: offset + step * index, in float64.
        r = maps[:, 1:2] + maps[:, 0:1] * self.i                # (N, 256)
        c = maps[:, 3:4] + maps[:, 2:3] * self.j                # (N, 192)
        r_ok = (r >= 0) & (r <= h - 1)
        c_ok = (c >= 0) & (c <= w - 1)
        r0f, c0f = torch.floor(r), torch.floor(c)
        wr, wc = r - r0f, c - c0f                               # weight of the +1 tap
        r0 = r0f.long().clamp(0, h - 1)
        c0 = c0f.long().clamp(0, w - 1)
        r1 = (r0 + 1).clamp(max=h - 1)                          # only reached with weight 0 at n-1
        c1 = (c0 + 1).clamp(max=w - 1)
        s = slots.view(-1, 1, 1)
        rr0, rr1 = r0[:, :, None], r1[:, :, None]
        cc0, cc1 = c0[:, None, :], c1[:, None, :]
        wr1, wc1 = wr[:, :, None, None], wc[:, None, :, None]
        wr0, wc0 = 1.0 - wr1, 1.0 - wc1
        # scipy's order-1 spline loop: sum over the 2x2 taps of (value * w_row) * w_col,
        # row-major tap order, in float64.
        acc = (frames[s, rr0, cc0].double() * wr0) * wc0
        acc = acc + (frames[s, rr0, cc1].double() * wr0) * wc1
        acc = acc + (frames[s, rr1, cc0].double() * wr1) * wc0
        acc = acc + (frames[s, rr1, cc1].double() * wr1) * wc1
        acc = torch.floor(acc + 0.5)                            # uint8 output: round half up
        acc = acc * (r_ok[:, :, None, None] & c_ok[:, None, :, None])
        x = acc.float().flip(-1).permute(0, 3, 1, 2)            # BGR -> RGB, NHWC -> NCHW
        x = (x - self.mean) / self.std
        if out is None:
            return x.half().contiguous()
        out.copy_(x)
        return out


# ---- the pose decode ------------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}


def _gaussian_weights(device) -> torch.Tensor:
    """scipy.ndimage._gaussian_kernel1d(0.8, 0, 5), float64."""
    key = str(device)
    if key not in _KERNEL_CACHE:
        radius = (DARK_KERNEL - 1) // 2
        x = np.arange(-radius, radius + 1)
        phi = np.exp(-0.5 / (DARK_SIGMA * DARK_SIGMA) * x ** 2)
        _KERNEL_CACHE[key] = torch.from_numpy(phi / phi.sum()).to(device)
    return _KERNEL_CACHE[key]


def _pad_reflect(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """scipy 'reflect' extension (d c b a | a b c d | d c b a), unlike torch's 'reflect'."""
    n = x.shape[dim]
    return torch.cat([x.narrow(dim, 0, r).flip(dim), x, x.narrow(dim, n - r, r).flip(dim)], dim=dim)


def _correlate1d(x: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    """scipy correlate1d with a symmetric kernel, float32 in and out: center * w0, then
    (left_j + right_j) * w_j for j = r .. 1, accumulated in float64 (ni_filters.c)."""
    r = (len(weights) - 1) // 2
    n = x.shape[dim]
    xp = _pad_reflect(x, r, dim).double()
    tap = lambda k: xp.narrow(dim, r + k, n)
    acc = tap(0) * weights[r]
    for j in range(r, 0, -1):
        acc = acc + (tap(-j) + tap(j)) * weights[r - j]
    return acc.float()


def decode(heatmaps: torch.Tensor, center, scale) -> tuple[torch.Tensor, torch.Tensor]:
    """HF post_process_pose_estimation (DARK, kernel 11) for (N, 17, 64, 48) heatmaps on any
    device: keypoints (N, 17, 2) in source px and scores (N, 17), float32, on that device.
    center, scale: (N, 2) float32 from center_scale(), as numpy arrays or tensors on that device."""
    hm = heatmaps.float()
    n, k, h, w = hm.shape
    dev = hm.device
    flat = hm.reshape(n, k, h * w)
    scores, idx = flat.max(dim=2)                               # first maximum, like np.argmax
    px = (idx % w).float()
    py = torch.div(idx, w, rounding_mode="floor").float()
    valid = scores > 0.0
    px = torch.where(valid, px, torch.full_like(px, -1.0))
    py = torch.where(valid, py, torch.full_like(py, -1.0))

    wts = _gaussian_weights(dev)
    blurred = _correlate1d(_correlate1d(hm, wts, 2), wts, 3)   # axis 0 (rows), then axis 1
    lg = torch.log(blurred.clamp(0.001, 50.0))
    lg = F.pad(lg.reshape(n * k, 1, h, w), (1, 1, 1, 1), mode="replicate").reshape(n, k, (h + 2) * (w + 2))
    stride = w + 2
    # HF's flat index into the edge-padded map; a (-1, -1) keypoint (score <= 0) lands on the
    # padded corner, whose -1 neighbours HF reads from the previous map -- clamp instead.
    base = (px + 1).long() + (py + 1).long() * stride

    def at(off: int) -> torch.Tensor:
        return lg.gather(2, (base + off).clamp(0, lg.shape[2] - 1).unsqueeze(-1)).squeeze(-1)

    i_ = at(0)
    ix1, ix1_ = at(1), at(-1)
    iy1, iy1_ = at(stride), at(-stride)
    ix1y1, ix1_y1_ = at(stride + 1), at(-stride - 1)
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    # hessian + float32 eps * I, inverted in float64 (HF: np.linalg.inv on a float64 array).
    eps = float(np.finfo(np.float32).eps)
    a, b, c, d = dxx.double() + eps, dxy.double(), dxy.double(), dyy.double() + eps
    det = a * d - b * c
    ddx, ddy = dx.double(), dy.double()
    off_x = (d * ddx - b * ddy) / det
    off_y = (a * ddy - c * ddx) / det
    rx = (px - off_x).float()
    ry = (py - off_y).float()

    cen = center if isinstance(center, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(center, np.float32)).to(dev)
    sc = (scale if isinstance(scale, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(scale, np.float32)).to(dev))
    sc = sc * NORMALIZE_FACTOR
    x = rx * (sc[:, 0:1] / (w - 1.0)) + cen[:, 0:1] - sc[:, 0:1] * 0.5
    y = ry * (sc[:, 1:2] / (h - 1.0)) + cen[:, 1:2] - sc[:, 1:2] * 0.5
    return torch.stack([x, y], dim=-1), scores


# ---- the CPU alternative: cv2 ---------------------------------------------------------------------

_MEAN_255 = (np.array(IMAGENET_MEAN, np.float32) * np.float32(1.0 / RESCALE)).reshape(1, 3, 1, 1)
_STD_255 = (np.array(IMAGENET_STD, np.float32) * np.float32(1.0 / RESCALE)).reshape(1, 3, 1, 1)


def warp_cv2(frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
    """The crop warp with cv2.warpAffine: (n, 3, 256, 192) fp16, normalized RGB, like
    cpu_stages.preprocess(). Samples outside [0, n-1] are zeroed as scipy's constant mode does
    (cv2's constant border would blend the edge pixel with 0 in the half-pixel strip)."""
    center, scale = center_scale(boxes_xywh)
    maps = sample_maps(center, scale)
    out = np.empty((len(maps), IN_H, IN_W, 3), np.uint8)
    h, w = frame_bgr.shape[:2]
    for n, (rs, ro, cs, co) in enumerate(maps):
        # Forward matrix (source -> crop) of the inverse maps above.
        fwd = np.array([[1.0 / cs, 0.0, -co / cs], [0.0, 1.0 / rs, -ro / rs]], np.float64)
        cv2.warpAffine(frame_bgr, fwd, (IN_W, IN_H), dst=out[n], flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rows = ro + rs * np.arange(IN_H)
        cols = co + cs * np.arange(IN_W)
        out[n][(rows < 0) | (rows > h - 1)] = 0
        out[n][:, (cols < 0) | (cols > w - 1)] = 0
    x = out[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)
    return np.ascontiguousarray(((x - _MEAN_255) / _STD_255).astype(np.float16))


# ---- the crop warp as one Triton kernel -----------------------------------------------------------
# GpuCropper.warp is ~40 torch kernels whose float64 intermediates are full (N, 256, 192, 3)
# tensors; this does the same arithmetic in one pass, one program per (crop, output row). It stays
# bit-identical to HF because every float op is the same IEEE op in the same order: FMA contraction
# is disabled at compile time, the division is div_rn (correctly rounded, like torch's), and the
# fp16 cast rounds to nearest even.

try:
    import triton
    import triton.language as tl
except ImportError:          # pragma: no cover - triton ships with CUDA torch wheels
    triton = None

if triton is not None:
    @triton.jit
    def _warp_kernel(frames_ptr, slots_ptr, maps_ptr, mean_ptr, std_ptr, out_ptr,
                     H, W, stride_slot, OUT_H: tl.constexpr, OUT_W: tl.constexpr, BLOCK_W: tl.constexpr):
        n = tl.program_id(0)
        i = tl.program_id(1)
        slot = tl.load(slots_ptr + n)
        row_step = tl.load(maps_ptr + n * 4 + 0)
        row_off = tl.load(maps_ptr + n * 4 + 1)
        col_step = tl.load(maps_ptr + n * 4 + 2)
        col_off = tl.load(maps_ptr + n * 4 + 3)
        r = row_off + row_step * i.to(tl.float64)
        j = tl.arange(0, BLOCK_W)
        jmask = j < OUT_W
        c = col_off + col_step * j.to(tl.float64)
        r_ok = (r >= 0.0) & (r <= (H - 1).to(tl.float64))
        c_ok = (c >= 0.0) & (c <= (W - 1).to(tl.float64))
        r0f = tl.floor(r)
        c0f = tl.floor(c)
        wr1 = r - r0f
        wc1 = c - c0f
        wr0 = 1.0 - wr1
        wc0 = 1.0 - wc1
        r0 = tl.minimum(tl.maximum(r0f.to(tl.int64), 0), H - 1)
        c0 = tl.minimum(tl.maximum(c0f.to(tl.int64), 0), W - 1)
        r1 = tl.minimum(r0 + 1, H - 1)
        c1 = tl.minimum(c0 + 1, W - 1)
        base = frames_ptr + slot.to(tl.int64) * stride_slot
        keep = (r_ok & c_ok).to(tl.float64)
        for k in tl.static_range(3):
            ch = 2 - k                                   # output RGB channel k <- source BGR channel 2-k
            v00 = tl.load(base + (r0 * W + c0) * 3 + ch, mask=jmask, other=0).to(tl.float64)
            v01 = tl.load(base + (r0 * W + c1) * 3 + ch, mask=jmask, other=0).to(tl.float64)
            v10 = tl.load(base + (r1 * W + c0) * 3 + ch, mask=jmask, other=0).to(tl.float64)
            v11 = tl.load(base + (r1 * W + c1) * 3 + ch, mask=jmask, other=0).to(tl.float64)
            acc = (v00 * wr0) * wc0
            acc = acc + (v01 * wr0) * wc1
            acc = acc + (v10 * wr1) * wc0
            acc = acc + (v11 * wr1) * wc1
            px = (tl.floor(acc + 0.5) * keep).to(tl.float32)
            x = tl.math.div_rn(px - tl.load(mean_ptr + k), tl.load(std_ptr + k))
            tl.store(out_ptr + ((n * 3 + k) * OUT_H + i) * OUT_W + j, x.to(tl.float16), mask=jmask)


def warp_triton(cropper: GpuCropper, frames: torch.Tensor, slots: torch.Tensor, maps: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """GpuCropper.warp in one Triton kernel, same results bit for bit. frames (S, H, W, 3) uint8
    contiguous; out (N, 3, 256, 192) fp16 contiguous."""
    n = maps.shape[0]
    _, h, w, _ = frames.shape
    if not (frames.is_contiguous() and out.is_contiguous() and maps.is_contiguous()):
        raise ValueError("warp_triton needs contiguous tensors")
    _warp_kernel[(n, IN_H)](frames, slots, maps, cropper.mean_flat, cropper.std_flat, out, h, w, h * w * 3,
                           OUT_H=IN_H, OUT_W=IN_W, BLOCK_W=256, enable_fp_fusion=False)
    return out
