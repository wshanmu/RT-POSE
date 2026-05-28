"""
Targeted per-layer diagnosis for TRT divergence.

Strategy: add a curated set of intermediate tensors as ONNX outputs — specifically
the outputs of InstanceNormalization and Resize nodes (the two high-risk ops for TRT
on 5D / 3D radar inputs). Compare ORT-CUDA vs ORT-TRT-EP on the reference input.

No PyTorch needed; requires:  pip install onnx  onnxruntime-gpu

Run:
    python3 diagnose_trt_layers.py \
        --onnx  rtpose_dzyx_y8/rtpose_dzyx_y8.onnx \
        --ref   rtpose_dzyx_y8/rtpose_dzyx_y8.ref.npz \
        --atol  0.01
"""

import argparse
import os
import sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np

TARGET_OPS = {"InstanceNormalization", "Resize", "GroupNormalization"}


def build_probe_model(onnx_path: str, tmp_path: str) -> list:
    """
    Clone the ONNX model and add the output of every InstanceNorm / Resize node
    as an extra graph output.  Returns list of (op_type, output_name).
    """
    import onnx
    from onnx import helper, TensorProto

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    # Build a shape map from value_info
    shape_map = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        try:
            if vi.type.tensor_type.HasField("shape"):
                shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        except Exception:
            pass

    existing_output_names = {o.name for o in m.graph.output}
    probes = []

    for node in m.graph.node:
        if node.op_type not in TARGET_OPS:
            continue
        for out_name in node.output:
            if not out_name or out_name in existing_output_names:
                continue
            # Make a value_info (shape unknown is fine for ORT)
            vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            m.graph.output.append(vi)
            existing_output_names.add(out_name)
            probes.append((node.op_type, out_name))

    os.makedirs(os.path.dirname(os.path.abspath(tmp_path)), exist_ok=True)
    onnx.save(m, tmp_path)
    print(f"[probe] added {len(probes)} probe outputs → {tmp_path}")
    return probes


def run_ort(model_path, input_np, providers, label):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=providers)
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[{label}] using {sess.get_providers()}")
    results = sess.run(None, {"rdr_tensor": input_np})
    return dict(zip(out_names, results))


def compare_probes(ref_out, trt_out, probes, atol):
    print(f"\n{'='*80}")
    print(f"{'Op type':<22} {'Tensor (truncated)':<35} {'shape':<25} {'max_diff'}")
    print(f"{'='*80}")

    first_fail = None
    for op_type, name in probes:
        if name not in ref_out or name not in trt_out:
            continue
        r = ref_out[name]
        t = trt_out[name]
        if r.shape != t.shape or not np.issubdtype(r.dtype, np.floating):
            continue
        diff = float(np.abs(r - t).max())
        mean_diff = float(np.abs(r - t).mean())
        status = "FAIL" if diff > atol else "ok"
        if status == "FAIL" and first_fail is None:
            first_fail = (op_type, name, diff, r.shape)
            status = "<<< FIRST FAIL >>>"
        print(f"  {op_type:<20} {name[:33]:<35} {str(r.shape):<25} {diff:.3e}  [{status}]")

    # Always print final outputs
    for out_name in ("hm", "reg"):
        if out_name in ref_out and out_name in trt_out:
            r = ref_out[out_name]
            t = trt_out[out_name]
            diff = float(np.abs(r - t).max())
            r_flat = r.reshape(r.shape[0], -1)
            t_flat = t.reshape(t.shape[0], -1)
            peak_ok = np.all(r_flat.argmax(axis=1) == t_flat.argmax(axis=1))
            print(f"  {'FINAL OUTPUT':<20} {out_name:<35} {str(r.shape):<25} {diff:.3e}  "
                  f"peak={'match' if peak_ok else 'DIFFER'}")

    print()
    if first_fail:
        op_type, name, diff, shape = first_fail
        print(f"[diag] Root cause candidate: {op_type}")
        print(f"       Tensor: {name}")
        print(f"       Shape:  {shape}")
        print(f"       max_diff = {diff:.3e}")
    else:
        print("[diag] All probe layers are within tolerance — error originates elsewhere.")
    return first_fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",      required=True)
    p.add_argument("--ref",       required=True)
    p.add_argument("--atol",      type=float, default=0.01)
    p.add_argument("--tmp-model", default="/tmp/rtpose_probe.onnx")
    p.add_argument("--no-trt",    action="store_true", help="compare ORT-CUDA vs ORT-CPU only")
    args = p.parse_args()

    try:
        import onnx   # noqa: F401
    except ImportError:
        sys.exit("[error] onnx not installed: pip install onnx")

    ref = np.load(args.ref)
    input_np = ref["input"]
    hm_ref   = ref["hm"]
    print(f"[ref]  hm scale: min={hm_ref.min():.4f}, max={hm_ref.max():.4f}, "
          f"mean={hm_ref.mean():.4f}")
    print(f"[ref]  input sparsity: {(input_np == 0).mean():.1%} zeros")
    print(f"[ref]  input range:    [{input_np.min():.4f}, {input_np.max():.4f}]")

    probes = build_probe_model(args.onnx, args.tmp_model)

    print("\n[run] ORT-CUDA (ground truth)…")
    cuda_out = run_ort(args.tmp_model, input_np,
                       ["CUDAExecutionProvider", "CPUExecutionProvider"], "ORT-CUDA")

    if args.no_trt:
        print("\n[run] ORT-CPU…")
        trt_out = run_ort(args.tmp_model, input_np, ["CPUExecutionProvider"], "ORT-CPU")
    else:
        print("\n[run] ORT-TRT-EP (builder opt_level=0)…")
        trt_opts = {
            "trt_engine_cache_enable": False,
            "trt_builder_optimization_level": 0,
        }
        trt_out = run_ort(
            args.tmp_model,
            input_np,
            [("TensorrtExecutionProvider", trt_opts),
             "CUDAExecutionProvider", "CPUExecutionProvider"],
            "ORT-TRT",
        )

    compare_probes(cuda_out, trt_out, probes, atol=args.atol)


if __name__ == "__main__":
    main()
