"""
Benchmark GPU preprocessing + TRT inference latency on the Jetson.

Simulates the real-time pipeline from apply_polar_to_cart_mapper_cupy:

    [cupy complex GPU array]
         ↓ abs          ← timed (GPU, cupy)
         ↓ log1p        ← timed (GPU, cupy)
         ↓ normalize    ← timed (GPU, cupy)
         ↓ cupy→numpy (unified mem flush) → TRT EP   ← timed
         ↓ (hm, reg)

Frame files must be named  {frame_id:05d}.npy  (float32 magnitude).
Loading npy → cupy is NOT timed.  We cast to complex64 so that cp.abs()
is a real operation (matching what apply_polar_to_cart_mapper_cupy gives).

Usage:
    python benchmark_trt.py \\
        --onnx    model.onnx \\
        --cache-dir ./trt_cache \\
        --folder  /path/to/DZYX_npy_f16_all2p5m \\
        --frames  200

Requires: cupy  onnxruntime-gpu
"""

import argparse
import os
import sys
import time

import cupy as cp
import numpy as np

from trt_utils import TrtSession, preprocess_cupy


# ── data loading ─────────────────────────────────────────────────────────────

def collect_paths(folder: str, n_frames: int) -> list[str]:
    paths, missing = [], []
    for i in range(n_frames):
        p = os.path.join(folder, f"{i:05d}.npy")
        (paths if os.path.exists(p) else missing).append(p)
    if missing:
        print(f"[bench]  WARNING: {len(missing)} file(s) missing "
              f"(e.g. {os.path.basename(missing[0])})")
    if not paths:
        sys.exit(f"[bench]  ERROR: no npy files found in {folder}")
    return paths


def load_as_cupy_complex(paths: list[str]) -> list:
    """
    Load float32 magnitude npy files onto GPU as complex64.
    Casting to complex64 means cp.abs() is a genuine operation,
    matching the real-time pipeline output of apply_polar_to_cart_mapper_cupy.
    Loading time is NOT part of the benchmark.
    """
    arrays = []
    for p in paths:
        arr_np = np.load(p).astype(np.complex64)   # treat magnitude as real part
        arrays.append(cp.asarray(arr_np))           # move to GPU
    return arrays


# ── stats ─────────────────────────────────────────────────────────────────────

def print_stats(label: str, values_ms: np.ndarray) -> None:
    print(f"\n  {label}")
    print(f"    mean   : {values_ms.mean():.3f} ms")
    print(f"    median : {np.percentile(values_ms, 50):.3f} ms")
    print(f"    p95    : {np.percentile(values_ms, 95):.3f} ms")
    print(f"    p99    : {np.percentile(values_ms, 99):.3f} ms")
    print(f"    max    : {values_ms.max():.3f} ms")


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(
    sess: TrtSession,
    gpu_arrays: list,
    norm_scale: float,
    warmup: int,
) -> None:

    # Warmup (TRT kernels + cupy JIT)
    for arr in gpu_arrays[:warmup]:
        prepped = preprocess_cupy(arr, norm_scale)
        sess.run_cupy(prepped)
    cp.cuda.Stream.null.synchronize()
    print(f"[bench]  warmed up ({warmup} passes)")

    preproc_ms = []
    infer_ms   = []

    for arr in gpu_arrays:
        # ── preprocessing (GPU, cupy) ──────────────────────────────────────
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        prepped = preprocess_cupy(arr, norm_scale)

        cp.cuda.Stream.null.synchronize()   # wait for GPU work to finish
        preproc_ms.append((time.perf_counter() - t0) * 1e3)

        # ── inference (cupy→numpy cache flush + TRT EP) ───────────────────
        # On Jetson unified memory, arr.get() is a coherence flush, not a copy.
        t1 = time.perf_counter()

        sess.run_cupy(prepped)

        infer_ms.append((time.perf_counter() - t1) * 1e3)

    pre  = np.array(preproc_ms)
    inf  = np.array(infer_ms)
    tot  = pre + inf

    print(f"\n[bench]  frames : {len(gpu_arrays)}")
    print_stats("GPU preprocessing  (abs + log1p + normalize)", pre)
    print_stats("TRT inference      (zero-copy cupy → ort EP)", inf)
    print_stats("Total per-frame    (preproc + inference)",      tot)
    print(f"\n  throughput : {1000.0 / tot.mean():.1f} FPS  "
          f"(based on mean total latency)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark GPU preproc + TRT inference on radar npy frames")
    p.add_argument("--onnx",       required=True, help="model.onnx")
    p.add_argument("--cache-dir",  required=True, help="TRT engine cache dir (from compile_trt.py)")
    p.add_argument("--folder",     required=True, help="folder with 00000.npy … frames")
    p.add_argument("--frames",     type=int, default=200,  help="number of frames (default 200)")
    p.add_argument("--warmup",     type=int, default=5,    help="warmup passes (default 5)")
    p.add_argument("--norm-scale", type=float, default=16.0, help="log1p normalisation scale (default 16.0)")
    p.add_argument("--no-fp16",    action="store_true")
    return p.parse_args()


def main():
    args  = parse_args()
    paths = collect_paths(args.folder, args.frames)

    print(f"[bench]  onnx      : {args.onnx}")
    print(f"[bench]  cache-dir : {args.cache_dir}")
    print(f"[bench]  folder    : {args.folder}")
    print(f"[bench]  frames    : {len(paths)}")

    print("[bench]  loading frames → GPU (not timed) …")
    gpu_arrays = load_as_cupy_complex(paths)
    ram_gpu_mb = sum(a.nbytes for a in gpu_arrays) / 1024 / 1024
    print(f"[bench]  {len(gpu_arrays)} frames on GPU  ({ram_gpu_mb:.0f} MB)")

    sess = TrtSession(args.onnx, args.cache_dir, fp16=not args.no_fp16)

    run_benchmark(sess, gpu_arrays, norm_scale=args.norm_scale, warmup=args.warmup)


if __name__ == "__main__":
    main()
