# TensorRT sm87 Trilinear Resize Bug — Root Cause & Fix

> **Platform**: Jetson Orin AGX · **TRT version**: 10.3.0 · **ORT version**: 1.23.0  
> **Model**: RT-Pose / HRNet3D  
> **Status**: ✅ Fixed and verified at **~52 FPS (FP16)**

---

## Table of Contents

1. [The Symptom](#1-the-symptom)
2. [Root Cause](#2-root-cause)
3. [Why the Naive Fixes Fail](#3-why-the-naive-fixes-fail)
4. [What Actually Works — The Probe Split](#4-what-actually-works--the-probe-split)
5. [Step-by-Step Fix](#5-step-by-step-fix)
6. [Performance Summary](#6-performance-summary)
7. [Production Usage](#7-production-usage)
8. [Environment Variable Gotcha](#8-environment-variable-gotcha)
9. [Script Reference](#9-script-reference)

---

## 1. The Symptom

Running RT-Pose inference through TensorRT EP produces catastrophically wrong results.
CUDA EP and CPU EP both pass; only TRT EP fails.

```
[check]  hm  max-abs-diff  = 1.666e+04   (tol 2e+00)  WARN   ← 8000× too large
[check]  hm  peak voxel    = DIFFERENT  <- FAIL
[check]  batch[0] ref=(6,3,36)  device=(0,0,0)           ← peak always at corner
```

The heatmap peak is always at voxel `(z, 0, 0)` — the H and W dimensions are
stuck at zero regardless of the input. The error is the same at TRT optimization
levels 0, 1, 2, and 3, and is independent of FP16 vs FP32.

---

## 2. Root Cause

**TRT 10.x has a bug in its internal 5-D trilinear Resize kernel on Jetson Orin (sm87).**

HRNet3D uses 5-D trilinear upsampling (`F.interpolate(..., mode='trilinear')`),
which exports to ONNX as `Resize` nodes with a 5-D input `[N, C, D, H, W]`.

When such a Resize node is *interior* to a single large TRT engine subgraph,
TRT's kernel appears to mis-stride the H and W dimensions — it effectively treats
the 5-D tensor as `[N, C, D, H*W]` and performs only 1-D interpolation along the
flattened spatial axis. The result is that every output voxel has `h_out=0, w_out=0`,
producing the characteristic corner spike.

**The bug fires only when Resize is interior to a TRT engine.**
When Resize sits at a *subgraph boundary* — where ORT's CUDA EP handles the
cross-engine tensor transfer — CUDA EP's own trilinear kernel runs instead,
which is correct.

### Architecture context

HRNet3D has 13 trilinear Resize nodes in its fuse layers (upsampling branches of
different resolutions back to the highest resolution). These are scattered throughout
Stages 2, 3, and 4 of the backbone.

```
  Branch j (low-res) → [GroupNorm → Conv3D] → Resize(trilinear) → add to Branch i
                                                      ↑
                                               sm87 bug fires here
                                           when inside a TRT engine
```

---

## 3. Why the Naive Fixes Fail

We systematically ruled out every obvious fix before finding the real solution.

### ❌ Lower TRT optimization level

`trt_builder_optimization_level = 0` disables all fusion — but the bug is in
the *primitive Resize kernel* itself, not in any fused pattern. Levels 0–3 all
produce the same `1.666e+04` hm error.

### ❌ Change coordinate transform mode (align_corners / half_pixel)

Patching every Resize node's coordinate transform mode from `align_corners=True`
to `half_pixel` or `asymmetric` does fix the z-axis peak coordinate, but the
`(y=0, x=0)` corner bug remains. This confirms the bug is in the H/W dimension
interpolation, not in coordinate alignment.

### ❌ Mark only Resize outputs as graph outputs (15 total)

`create_trt_boundary_model.py` adds the 13 Resize output tensors as explicit ONNX
graph outputs, hoping to force ORT to route those tensors through CUDA EP.
**This does not work.**  TRT EP builds *one big engine* that outputs all 15 tensors
from within a single TRT subgraph. TRT can emit intermediate tensors from inside an
engine; graph output status alone does not force a CUDA EP fallback.

### ❌ Resize sandwich — mark both Resize inputs and outputs (28 total)

`create_resize_sandwich_model.py` adds the input *and* output tensor of every
Resize node as graph outputs (26 probes + 2 originals = 28 total). TRT still
builds one big engine and the results are identical to the original bug.

### ❌ `trt_nodes_to_exclude`

The ORT build installed on this Jetson (`/opt/onnxruntime/`) does not support
the `trt_nodes_to_exclude` option — it returns:

```
EP Error: Invalid TensorRT EP option: trt_nodes_to_exclude
```

---

## 4. What Actually Works — The Probe Split

### Key discovery

`diagnose_trt_layers.py` adds **all** InstanceNormalization and Resize node outputs
as graph outputs — 61 probe outputs in total, making 63 graph outputs. With this
model, TRT produces **correct results**:

```
[check]  hm  peak voxel  = same  <- PASS
[check]  batch[0] ref=(6,3,36)  device=(6,3,36)
```

### Why it works

With 63 scattered graph outputs, TRT EP hits an internal engine complexity/size
limit and **must partition the model into multiple smaller TRT subgraphs**.
The Resize nodes end up at subgraph boundaries. ORT's CUDA EP handles the
cross-engine data transfer at each boundary, running its own (correct) trilinear
kernel. The small TRT engines on either side of each Resize node compute everything
else correctly.

With only 15 or 28 graph outputs, TRT stays under its limit and keeps everything
in one engine — the bug fires. The splitting threshold appears to be somewhere
between 28 and 63 graph outputs.

### The runtime trick: request only `["hm", "reg"]`

ORT's TRT graph partition is determined at **session creation time** from the ONNX
model structure. At `sess.run()` time, the pre-compiled TRT engine binaries are
called; ORT cannot restructure them.

This means:
- **Session creation**: use the 63-output split model → TRT partitions correctly
- **Runtime inference**: call `sess.run(["hm", "reg"], ...)` → only 2 Python numpy
  arrays are allocated per call; the 61 probe boundary tensors flow between engines
  as ORT-internal GPU buffers and are never exposed to Python

Requesting all 63 outputs vs only 2 at runtime:

| `sess.run(...)` call | Latency (FP16) | Correctness |
|---|---|---|
| `sess.run(None, ...)` — all 63 outputs | 67 ms | ✅ |
| `sess.run(["hm","reg"], ...)` — 2 outputs | **19 ms** | ✅ |

The ~45 ms difference comes from allocating 61 numpy array views into Jetson's
unified memory on every call. The TRT compute is identical in both cases.

---

## 5. Step-by-Step Fix

### Prerequisites

```bash
conda activate py10_nvme      # or your ORT environment
pip install onnx              # only needed for model surgery scripts
```

### Step 1 — Create the split model (run once, on any machine with `onnx`)

```bash
cd RT-POSE/tools

python create_full_probe_model.py \
    --onnx rtpose_dzyx_y8/rtpose_dzyx_y8.onnx \
    --out  rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx
```

Expected output:
```
[probe] source : rtpose_dzyx_y8/rtpose_dzyx_y8.onnx
[probe] added 61 probe outputs  (total graph outputs: 63)
[probe] saved  : rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx
```

The `_trt_split.onnx` file is the only model you need on the Jetson for deployment.
Copy the `_trt_split.onnx` alongside the original if you also want CUDA EP reference.

### Step 2 — Verify correctness on the Jetson (TRT FP32)

```bash
rm -rf ./trt_cache_split

python verify_on_device.py \
    --onnx   rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
    --ref    rtpose_dzyx_y8/rtpose_dzyx_y8.ref.npz \
    --provider TensorRT \
    --cache-dir ./trt_cache_split \
    --warmup 3 --runs 20
```

Expected:
```
[check]  hm  peak voxel  = same  <- PASS
[result] PASS — peak voxel matches reference.
[run]    avg latency = 19.xx ms  (~52 FPS)
```

### Step 3 — Verify correctness (TRT FP16, for production)

```bash
rm -rf ./trt_cache_split_fp16

python verify_on_device.py \
    --onnx   rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
    --ref    rtpose_dzyx_y8/rtpose_dzyx_y8.ref.npz \
    --provider TensorRT \
    --cache-dir ./trt_cache_split_fp16 \
    --fp16 \
    --warmup 3 --runs 30
```

Expected:
```
[check]  hm  max-abs-diff  = 3.2e+01  (tol 2e+00)  WARN  ← expected FP16 drift
[check]  hm  peak voxel    = same  <- PASS                ← detection is correct
[result] PASS — peak voxel matches reference.
[run]    avg latency = 19.xx ms  (~52 FPS)
```

The numeric WARN is normal — FP16 introduces absolute value drift of ~30 units,
but the argmax (peak voxel location) is unaffected. RT-Pose's detection pipeline
uses argmax, not the raw heatmap values, so FP16 is correct for this task.

### Step 4 — Run the visualizer

```bash
# Set NVIDIA_TF32_OVERRIDE before the first run (see §8 below)
python visualize_session_rt.py \
    --onnx  rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
    --session-dir /path/to/synchronized/session_dir \
    --skeleton-only \
    --out-dir /tmp/viz_output

# Assemble into a video
ffmpeg -framerate 10 -i /tmp/viz_output/%05d.png \
    -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## 6. Performance Summary

All numbers measured on **Jetson Orin AGX**, input shape `[1, 128, 16, 8, 64]`.
`sess.run(["hm","reg"], ...)` is used at runtime in all ORT-based configs.

| Configuration | Latency | FPS | Peak voxel |
|---|---|---|---|
| TRT FP16, original model | 16 ms | 61.8 | ❌ FAIL `(0,0,0)` — sm87 bug |
| CUDA EP FP32 | 120 ms | 8.4 | ✅ PASS |
| TRT FP32, split model | 19 ms | ~52 | ✅ PASS |
| **TRT FP16, split model** | **19 ms** | **~52** | ✅ **PASS** |

The split model has essentially no latency overhead compared to the original
(wrong) TRT FP16 model because the bottleneck is GPU compute, not engine count.

---

## 7. Production Usage

### Python (ORT)

```python
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"   # must be before any onnxruntime import
import onnxruntime as ort

trt_opts = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_cache_prod",
    "trt_timing_cache_enable": True,
    "trt_builder_optimization_level": 2,   # 3 may break GroupNorm→Conv3D
}
sess = ort.InferenceSession(
    "rtpose_dzyx_y8_trt_split.onnx",
    providers=[("TensorrtExecutionProvider", trt_opts),
               "CUDAExecutionProvider",
               "CPUExecutionProvider"]
)

# At inference time — request ONLY hm and reg (not None / all outputs)
hm, reg = sess.run(["hm", "reg"], {"rdr_tensor": input_np})
```

### C++ (ORT C API)

The C++ class `RtPoseOnnx` in
`src/ros-recording/radar_processing_cpp/src/rt_pose_onnx.cpp`
has a dedicated constructor for TRT EP:

```cpp
RtPoseOnnx::TrtConfig trt_cfg;
trt_cfg.fp16       = true;
trt_cfg.opt_level  = 2;
trt_cfg.cache_path = "/tmp/trt_cache_rt_pose";

auto pose_infer = std::make_unique<RtPoseOnnx>(
    "rtpose_dzyx_y8_trt_split.onnx", trt_cfg, /*device_id=*/0);
```

The `output_names_` array is already `{"hm", "reg"}`, so the C++ inference
path automatically uses the correct 2-output runtime call.

### ROS 2 launch file parameters

```python
# In your launch .py, pass to the radar_4d_monitor node:
parameters=[{
    "onnx_model":          "/path/to/rtpose_dzyx_y8_trt_split.onnx",
    "onnx_trt_enable":     True,
    "onnx_trt_fp16":       True,
    "onnx_trt_opt_level":  2,
    "onnx_trt_cache_path": "/tmp/trt_cache_rt_pose",
}]
```

---

## 8. Environment Variable Gotcha

TRT caches the value of `NVIDIA_TF32_OVERRIDE` at engine build time.
If the variable differs at load time, TRT raises:

```
ICudaEngine::createExecutionContext: Error Code 1: Myelin
  [header.cpp:operator():94] Inconsistent setting of NVIDIA_TF32_OVERRIDE
  env var at build 0 and at execution -1
```

**Fix**: always set `NVIDIA_TF32_OVERRIDE=0` **before** importing `onnxruntime`,
at the very top of any Python script:

```python
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"   # ← line 1, before everything else

import onnxruntime as ort
# ...
```

If you hit this error with an existing cache:

```bash
# Option A: simply re-run — the env var is now set, engines reload fine
# (build=0 and execution=0 are consistent)

# Option B: if engines are truly corrupted, delete and rebuild:
rm -rf ./trt_cache_split_fp16
python verify_on_device.py --onnx ... --fp16 --cache-dir ./trt_cache_split_fp16
```

---

## 9. Script Reference

| Script | Purpose |
|---|---|
| `create_full_probe_model.py` | **The fix**: adds 61 InstanceNorm + Resize outputs to the ONNX graph, producing the 63-output split model. Run once on any machine with `pip install onnx`. |
| `verify_on_device.py` | Verify any ONNX model on Jetson with CPU / CUDA / TRT EP. Detects probe models automatically and uses `sess.run(["hm","reg"])` at runtime. |
| `visualize_session_rt.py` | Per-frame 3D skeleton visualization using the split ONNX model with ORT TRT EP. |
| `diagnose_trt_layers.py` | **Diagnostic only**: the script that first proved the probe split works. Saves probe model to `/tmp/rtpose_probe.onnx`. |
| `direct_compare.py` | Compares ORT-CUDA vs TRT-EP directly on the same model, bypassing the PyTorch reference. Useful for isolating provider disagreements. |
| `create_trt_boundary_model.py` | *(Insufficient)* Adds only 13 Resize outputs as graph outputs. TRT still builds one engine. Kept for reference. |
| `create_resize_sandwich_model.py` | *(Insufficient)* Adds Resize input + output tensors as graph outputs (28 total). TRT still builds one engine. Kept for reference. |
| `exclude_resize_from_trt.py` | Extracts Resize node names for use with `trt_nodes_to_exclude` — not supported in this ORT build. |
| `debug_trt_ops.py` | Sweeps TRT optimization levels 0–3 and reports per-level hm diff. Ruled out fusion as the root cause. |
| `fix_resize_onnx.py` | Patches Resize coordinate transform modes (halfpix / asymmetric). Fixed z-peak but not the (y=0,x=0) bug. |

---

## Appendix — Why `sess.run(None)` is Slow

When you call `sess.run(None, ...)`, ORT allocates a Python numpy array for
**every one of the 63 graph outputs**. On Jetson's unified memory architecture,
each allocation involves `cudaMallocManaged` bookkeeping. This overhead is
proportional to the number of output arrays, not their sizes:

- `sess.run(None, ...)` → 63 allocations → ~67 ms total
- `sess.run(["hm","reg"], ...)` → 2 allocations → ~19 ms total

The 61 probe boundary tensors still exist as GPU buffers — they must, because
they are cross-engine data that flows from one TRT subgraph to the next. ORT
just does not expose them as Python objects when they are not requested.

This is why `verify_on_device.py` and `visualize_session_rt.py` both use
`sess.run(["hm","reg"], ...)` even for the 63-output probe model.
