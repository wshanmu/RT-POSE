"""
Create a permanent "TRT-split" ONNX model for deployment on Jetson Orin.

Problem
-------
TRT 10.x on Jetson sm87 has a buggy internal 5D trilinear Resize kernel.
The bug only fires when the Resize node is *interior* to a large TRT engine
subgraph. When Resize is at a TRT subgraph *boundary*, CUDA EP handles
the cross-engine copy and produces correct values.

ORT's TRT EP does NOT create subgraph boundaries just because a tensor is
marked as a graph output — TRT builds one big engine that outputs everything.

What DOES force graph splitting: adding enough scattered intermediate outputs
that TRT's engine hits its complexity/size limit and must partition.  We
observed that adding 61 InstanceNorm + Resize probe outputs (in
diagnose_trt_layers.py) reliably forces this split and gives correct results.

This script creates that probe model permanently, so it can be used for
deployment without re-running the diagnostic tool.

Usage
-----
    python3 create_full_probe_model.py \
        --onnx rtpose_dzyx_y8/rtpose_dzyx_y8.onnx \
        --out  rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx

    # verify with TRT:
    python3 verify_on_device.py \
        --onnx rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
        --ref  rtpose_dzyx_y8/rtpose_dzyx_y8.ref.npz \
        --provider TensorRT --cache-dir ./trt_cache_split
"""

import argparse
import os
import sys

# Ops whose outputs, when added as graph outputs, force TRT to split the graph.
# InstanceNormalization = GroupNorm decomposition in ONNX.
# Resize               = trilinear upsampling (the buggy one).
SPLIT_OPS = {"InstanceNormalization", "Resize", "GroupNormalization"}


def create_split_model(src: str, dst: str) -> int:
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        sys.exit("pip install onnx")

    m = onnx.load(src)
    onnx.checker.check_model(m)

    # Build shape map
    shape_map = {}
    for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        try:
            if vi.type.tensor_type.HasField("shape"):
                shape_map[vi.name] = vi
        except Exception:
            pass

    existing_outputs = {o.name for o in m.graph.output}
    n_added = 0

    for node in m.graph.node:
        if node.op_type not in SPLIT_OPS:
            continue
        for out_name in node.output:
            if not out_name or out_name in existing_outputs:
                continue
            if out_name in shape_map:
                src_vi = shape_map[out_name]
                new_vi = helper.make_tensor_value_info(
                    out_name,
                    src_vi.type.tensor_type.elem_type,
                    [d.dim_value or None for d in src_vi.type.tensor_type.shape.dim],
                )
            else:
                new_vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            m.graph.output.append(new_vi)
            existing_outputs.add(out_name)
            n_added += 1

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    onnx.save(m, dst)
    return n_added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out",  required=True)
    args = p.parse_args()

    print(f"[probe] source : {args.onnx}")
    n = create_split_model(args.onnx, args.out)
    total_out = n + 2  # hm + reg
    print(f"[probe] added {n} probe outputs  (total graph outputs: {total_out})")
    print(f"[probe] saved  : {args.out}")
    print(f"""
Key: TRT EP splits its engine at ~{total_out} scattered outputs, placing
the Resize nodes at engine boundaries where CUDA EP (correctly) handles them.

verify_on_device.py already requests all outputs at runtime
(sess.run(None, ...)) and extracts hm/reg by name, so the extra
{n} probe tensors are returned but immediately discarded.
On Jetson unified memory this adds <1 ms overhead.

Next:
  rm -rf ./trt_cache_split
  python3 verify_on_device.py \\
      --onnx {args.out} \\
      --ref  <path-to>.ref.npz \\
      --provider TensorRT --cache-dir ./trt_cache_split
""")


if __name__ == "__main__":
    main()
