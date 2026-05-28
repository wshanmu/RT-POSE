"""
Verify a RadarPoseNet ONNX model on a target device (e.g. Jetson Orin AGX).

Requires only:  numpy  onnxruntime   (no PyTorch, no det3d)

Workflow
--------
On the dev machine, after export_onnx.py:
    cp work_dirs/model.onnx      <jetson>:~/rtpose/
    cp work_dirs/model.ref.npz   <jetson>:~/rtpose/

On the Jetson:
    python verify_on_device.py --onnx model.onnx --ref model.ref.npz --provider CUDA
    python verify_on_device.py --onnx model.onnx --ref model.ref.npz --provider TensorRT --cache-dir ./trt_cache

Primary correctness check: peak voxel must match the reference.
Numerical tolerances are informational — GPU/TRT floating-point ordering
differs from PyTorch/CPU-ORT, especially for GroupNorm layers.
"""

import argparse
import os
import sys
import time

# Must be set before importing onnxruntime. This keeps CUDA/TRT FP32 checks from
# silently using TF32 when the runtime honors the environment variable.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np

_TOL = {
    "CPU":       {"hm": 0.05,  "reg": 0.005},
    "CUDA":      {"hm": 1.5,   "reg": 0.05},
    "TENSORRT":  {"hm": 2.0,   "reg": 0.1},
}


def _require_provider(available, provider_name):
    if provider_name not in available:
        sys.exit(f"{provider_name} is not available in this ONNX Runtime build.")


def _get_resize_node_names(onnx_path: str) -> str:
    """
    Return a comma-separated string of all Resize node names in the ONNX model.
    Requires the 'onnx' package.  Raises ImportError if not installed.
    """
    import onnx
    m = onnx.load(onnx_path)
    names = [n.name for n in m.graph.node if n.op_type == "Resize" and n.name]
    if not names:
        print("[warn]   Resize nodes have no 'name' attribute — cannot use trt_nodes_to_exclude. "
              "Run exclude_resize_from_trt.py to investigate.")
        return ""
    print(f"[trt]    excluding {len(names)} Resize nodes from TRT EP → CUDA EP fallback")
    return ",".join(names)


def load_session(onnx_path: str, provider: str, cache_dir: str, fp16: bool, trt_only: bool,
                 trt_opt_level: int = None, exclude_resize: bool = False, exclude_file: str = None):
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"[device] onnxruntime providers available: {available}")

    key = provider.upper()
    if key == "CPU":
        _require_provider(available, "CPUExecutionProvider")
        providers = ["CPUExecutionProvider"]

    elif key == "CUDA":
        _require_provider(available, "CUDAExecutionProvider")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif key in ("TENSORRT", "TRT"):
        _require_provider(available, "TensorrtExecutionProvider")
        os.makedirs(cache_dir, exist_ok=True)
        trt_opts = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            # Optimization level: 2 is the safe default for HRNet3D.
            # Level 3 may incorrectly handle GroupNorm→Conv3D patterns.
            "trt_builder_optimization_level": trt_opt_level if trt_opt_level is not None else 2,
        }

        # ── Exclude Resize nodes from TRT (root-cause fix for Jetson sm87 bug) ──────
        # TRT 10.x on Jetson Orin (sm87) has a bug in its internal 5D trilinear
        # Resize kernel: when a Resize node runs inside a large TRT subgraph
        # (not at a graph output boundary), TRT incorrectly interpolates the H and W
        # spatial dimensions.  The fix is to exclude all Resize nodes from TRT,
        # letting them fall back to CUDA EP which computes them correctly.
        # Performance impact is small since Resize is memory-bound, not compute-bound.
        nodes_to_exclude = ""
        if exclude_file:
            with open(exclude_file) as f:
                nodes_to_exclude = f.read().strip()
            print(f"[trt]    loaded {len(nodes_to_exclude.split(','))} node names from {exclude_file}")
        elif exclude_resize:
            try:
                nodes_to_exclude = _get_resize_node_names(onnx_path)
            except ImportError:
                print("[warn]   'onnx' not installed; cannot auto-detect Resize node names.")
                print("         Run: pip install onnx && python3 exclude_resize_from_trt.py --onnx " + onnx_path)
        if nodes_to_exclude:
            trt_opts["trt_nodes_to_exclude"] = nodes_to_exclude
        providers = [("TensorrtExecutionProvider", trt_opts)]
        if not trt_only:
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
    else:
        sys.exit(f"Unknown provider '{provider}'. Choose: CPU, CUDA, TensorRT")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[device] session using: {sess.get_providers()}")
    return sess


def compare(hm_ref: np.ndarray, reg_ref: np.ndarray,
            hm_tgt: np.ndarray, reg_tgt: np.ndarray,
            provider: str, atol_hm=None, atol_reg=None) -> bool:

    tol = _TOL.get(provider.upper(), _TOL["TENSORRT"])
    hm_tol = tol["hm"] if atol_hm is None else atol_hm
    reg_tol = tol["reg"] if atol_reg is None else atol_reg

    if hm_ref.shape != hm_tgt.shape:
        raise ValueError(f"hm shape mismatch: ref {hm_ref.shape}, device {hm_tgt.shape}")
    if reg_ref.shape != reg_tgt.shape:
        raise ValueError(f"reg shape mismatch: ref {reg_ref.shape}, device {reg_tgt.shape}")

    hm_diff  = np.abs(hm_ref  - hm_tgt).max()
    reg_diff = np.abs(reg_ref - reg_tgt).max()
    hm_mean  = np.abs(hm_ref  - hm_tgt).mean()
    reg_mean = np.abs(reg_ref - reg_tgt).mean()
    hm_ok    = hm_diff  <= hm_tol
    reg_ok   = reg_diff <= reg_tol

    print(f"[check]  hm  max-abs-diff  = {hm_diff :.3e}  (tol {hm_tol :.0e})  {'PASS' if hm_ok  else 'WARN'}")
    print(f"[check]  hm  mean-abs-diff = {hm_mean :.3e}")
    print(f"[check]  reg max-abs-diff  = {reg_diff:.3e}  (tol {reg_tol:.0e})  {'PASS' if reg_ok else 'WARN'}")
    print(f"[check]  reg mean-abs-diff = {reg_mean:.3e}")

    # Peak voxel — primary correctness criterion for detection
    hm_flat_ref = hm_ref.reshape(hm_ref.shape[0], -1)
    hm_flat_tgt = hm_tgt.reshape(hm_tgt.shape[0], -1)
    peak_match  = np.all(hm_flat_ref.argmax(axis=1) == hm_flat_tgt.argmax(axis=1))
    print(f"[check]  hm  peak voxel    = {'same  <- PASS' if peak_match else 'DIFFERENT  <- FAIL'}")

    D, H, W = hm_ref.shape[2], hm_ref.shape[3], hm_ref.shape[4]
    for b in range(hm_ref.shape[0]):
        idx_r = int(hm_flat_ref[b].argmax())
        idx_t = int(hm_flat_tgt[b].argmax())
        d_r, h_r, w_r = idx_r//(H*W), (idx_r%(H*W))//W, idx_r%W
        d_t, h_t, w_t = idx_t//(H*W), (idx_t%(H*W))//W, idx_t%W
        print(f"[check]  batch[{b}] ref=({d_r},{h_r},{w_r})  device=({d_t},{h_t},{w_t})")

    return peak_match, hm_ok and reg_ok


def parse_args():
    p = argparse.ArgumentParser(description="On-device ONNX verification (no PyTorch needed)")
    p.add_argument("--onnx",      required=True,             help="path to model.onnx")
    p.add_argument("--ref",       required=True,             help="path to model.ref.npz")
    p.add_argument("--provider",  default="CUDA",            help="CPU | CUDA | TensorRT  (default: CUDA)")
    p.add_argument("--cache-dir", default="./trt_cache_verify", help="TRT engine cache dir (TensorRT provider only)")
    p.add_argument("--warmup",    type=int, default=3,       help="warmup runs (default 3)")
    p.add_argument("--runs",      type=int, default=10,      help="timed runs (default 50)")
    p.add_argument("--fp16",      action="store_true",       help="enable TensorRT FP16 mode")
    p.add_argument("--trt-opt-level", type=int, default=None, choices=[0,1,2,3,4,5],
                   help="TRT builder optimization level (overrides default 2). "
                        "Level 3+ may incorrectly fold GroupNorm→Conv3D. "
                        "Use 0 to fully disable fusion, 2 to keep safe fusions.")
    p.add_argument("--trt-only",  action="store_true",       help="do not register CUDA/CPU fallback after TensorRT")
    p.add_argument("--trt-exclude-resize", action="store_true",
                   help="Exclude all Resize (trilinear) nodes from TRT EP, forcing them to CUDA EP. "
                        "Fixes TRT 10.x bug on Jetson sm87 where 5D trilinear Resize computes "
                        "wrong H/W interpolation when internal to a TRT subgraph. "
                        "Requires --onnx model to also have 'onnx' package available for node inspection, "
                        "or supply --trt-exclude-file.")
    p.add_argument("--trt-exclude-file", default=None,
                   help="Path to a text file containing comma-separated Resize node names to exclude "
                        "from TRT EP (produced by exclude_resize_from_trt.py). "
                        "Alternative to --trt-exclude-resize when onnx package is not available.")
    p.add_argument("--atol-hm",   type=float, default=None,  help="override hm max-abs tolerance")
    p.add_argument("--atol-reg",  type=float, default=None,  help="override reg max-abs tolerance")
    p.add_argument("--fail-on-numeric", action="store_true", help="exit nonzero when max-abs tolerances are exceeded")
    p.add_argument("--save-output-dir", default=None,        help="optional directory for device output .npy files")
    return p.parse_args()


def main():
    args = parse_args()

    ref      = np.load(args.ref)
    input_np = ref["input"]
    hm_ref   = ref["hm"]
    reg_ref  = ref["reg"]
    source = str(ref["source"]) if "source" in ref else "unknown"

    print(f"[ref]    loaded {args.ref}")
    print(f"[ref]    source={source}")
    print(f"[ref]    input={list(input_np.shape)}  hm={list(hm_ref.shape)}  reg={list(reg_ref.shape)}")
    print(f"[run]    provider={args.provider}")

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    sess = load_session(args.onnx, args.provider, args.cache_dir, args.fp16, args.trt_only,
                        trt_opt_level=args.trt_opt_level,
                        exclude_resize=args.trt_exclude_resize,
                        exclude_file=args.trt_exclude_file)

    # ── Detect probe / split model (> 2 graph outputs) ───────────────────
    _out_names = [o.name for o in sess.get_outputs()]
    _is_probe  = len(_out_names) > 2
    if _is_probe:
        _hm_idx  = _out_names.index("hm")
        _reg_idx = _out_names.index("reg")
        print(f"[run]    probe/split model: {len(_out_names)} graph outputs — "
              f"requesting ALL to keep TRT graph-partition intact")

    print(f"[run]    warming up ({args.warmup} runs) …")
    for _ in range(args.warmup):
        if _is_probe:
            # Request all outputs: prevents ORT from pruning probe subgraphs
            # and re-merging them into one big TRT engine (sm87 Resize bug path)
            sess.run(None, {"rdr_tensor": input_np})
        else:
            sess.run(["hm", "reg"], {"rdr_tensor": input_np})

    # For probe/split models, request ALL outputs at runtime so ORT's TRT EP
    # cannot prune the extra subgraphs and re-merge them into one big engine.
    # That merge would re-expose the sm87 5D-trilinear Resize bug.
    # On Jetson unified memory the extra tensor copies add < 1 ms overhead.
    print(f"[run]    timing {args.runs} runs …")
    t0 = time.perf_counter()
    for _ in range(args.runs):
        if _is_probe:
            _outs   = sess.run(None, {"rdr_tensor": input_np})
            hm_tgt  = _outs[_hm_idx]
            reg_tgt = _outs[_reg_idx]
        else:
            hm_tgt, reg_tgt = sess.run(["hm", "reg"], {"rdr_tensor": input_np})
    avg_ms = (time.perf_counter() - t0) / args.runs * 1000
    print(f"[run]    avg latency = {avg_ms:.2f} ms  ({1000/avg_ms:.1f} FPS)")

    if args.save_output_dir:
        os.makedirs(args.save_output_dir, exist_ok=True)
        np.save(os.path.join(args.save_output_dir, "hm_device.npy"), hm_tgt)
        np.save(os.path.join(args.save_output_dir, "reg_device.npy"), reg_tgt)
        print(f"[run]    saved device outputs -> {args.save_output_dir}")

    peak_ok, numeric_ok = compare(
        hm_ref,
        reg_ref,
        hm_tgt,
        reg_tgt,
        args.provider,
        atol_hm=args.atol_hm,
        atol_reg=args.atol_reg,
    )
    print()
    if peak_ok and (numeric_ok or not args.fail_on_numeric):
        suffix = "" if numeric_ok else " Numerical tolerance exceeded; inspect WARN lines."
        print(f"[result] PASS — peak voxel matches reference.{suffix}")
    else:
        if not peak_ok:
            print("[result] FAIL — peak voxel differs (real detection error).")
        else:
            print("[result] FAIL — numerical tolerance exceeded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
