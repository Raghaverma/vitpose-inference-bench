#!/usr/bin/env python3
"""Regenerate every figure in docs/images/ from the repo's actual result
JSON files.

    python visualizations/generate_all.py

Runs each script's generate() in sequence and lets any exception propagate
uncaught -- a missing/incomplete/implausible input JSON aborts the whole run
with that script's specific, actionable error message (see visualizations/
_data.py) rather than silently shipping a partial set of figures. If this
exits 0, every PNG under docs/images/ was just freshly regenerated from
whatever result files are currently on disk.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualizations import (
    architecture,
    batch_matrix,
    batch_scaling,
    latency,
    numerical_equivalence,
    pipeline,
    stage_breakdown,
    throughput,
    vram_scaling,
)

GENERATORS = [
    ("architecture", architecture.generate),
    ("stage_breakdown + methodology", stage_breakdown.generate),
    ("latency", latency.generate),
    ("throughput", throughput.generate),
    ("batch_scaling (latency + throughput)", batch_scaling.generate),
    ("numerical_equivalence", numerical_equivalence.generate),
    ("batch_matrix (Stage 4: throughput/latency vs batch)", batch_matrix.generate),
    ("vram_scaling (Stage 4: TensorRT VRAM breakdown)", vram_scaling.generate),
    ("pipeline (Stages 7-9: video pipelines, production configuration)", pipeline.generate),
]


def main() -> None:
    written: list[Path] = []
    t0 = time.perf_counter()
    for name, generate_fn in GENERATORS:
        print(f"[generate_all] {name} ...")
        result = generate_fn()
        paths = result if isinstance(result, list) else [result]
        written.extend(paths)
        for p in paths:
            print(f"[generate_all]   -> {p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p}")

    elapsed = time.perf_counter() - t0
    print(f"\n[generate_all] wrote {len(written)} figures in {elapsed:.1f}s -- "
          f"every number in them was just read fresh from results/*.json.")


if __name__ == "__main__":
    main()
