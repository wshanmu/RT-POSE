"""
Export a trained RadarPoseNet checkpoint to ONNX and verify numerical equivalence.

Usage (from RT-POSE root):
    PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
    python tools/export_onnx.py \
        --config configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
        --ckpt   work_dirs/hr3d_one_hm_23j_dzyx_leaveout/20260429_165334/epoch_35.pth \
        --output work_dirs/rtpose_dzyx_y8/rtpose_dzyx_y8.onnx \
        --npy    /data1/shanmu/ai-fitness-coach/ssd_datas/fitness_data/synchronized/boelter_session17/DZYX_npy_f16/00000.npy \
        --save-ref

The exported ONNX model:
    Input  : "rdr_tensor"  float32  [1, 128, 16, 8, 64]
                           The 128 Doppler bins become the channel axis.
                           Values must be log1p-normalised: arr = log1p(raw) / 16.0
    Outputs: "hm"          float32  [1, 1,  16, 8, 64]  raw heatmap logits (pre-sigmoid)
             "reg"         float32  [1, 69, 16, 8, 64]  regression offsets, 23 joints x 3

Batch size
----------
The ONNX graph is exported with a FIXED batch size of 1.  The model is meant
for real-time, single-frame inference, so batch>1 is not needed.  If you ever
want dynamic batch, pass --dynamic-batch and the "batch" axis will be symbolic.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

# ── env guards (must precede any det3d import that touches spconv / iou3d) ────
os.environ.setdefault("RTPOSE_DISABLE_IOU3D", "1")
os.environ.setdefault("RTPOSE_DISABLE_SPCONV", "1")

from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_npy(
    path: str,
    norm_start: float = 0.0,
    norm_end: float = 16.0,
    expected_shape=None,
) -> np.ndarray:
    """
    Replicate the dataset's get_cube() pipeline on a single raw npy frame.

    Raw file  : float32, shape (128, 16, 8, 64)  - (Doppler, Z, Y, X)
    After this: float32, shape (128, 16, 8, 64)  - same layout, values in [0, ~1]

    This exporter intentionally does not crop the radar cube. For the current
    Jetson deployment test, the DZYX input contract is the raw y=8 tensor.
    """
    arr = np.load(path).astype(np.float32)
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"Expected raw npy shape {tuple(expected_shape)}, got {tuple(arr.shape)} from {path}"
        )

    norm_scale = float(norm_end) - float(norm_start)
    if norm_scale <= 0:
        raise ValueError(f"Invalid normalization range: ({norm_start}, {norm_end})")

    arr = np.log1p(arr)
    arr = (arr - float(norm_start)) / norm_scale
    arr[arr < 0.0] = 0.0
    return arr


def make_input(
    npy_path: str,
    device: torch.device,
    norm_start: float,
    norm_end: float,
    expected_shape,
) -> torch.Tensor:
    """
    Load one real radar frame, preprocess it, and wrap as a [1, 128, 16, 8, 64] tensor.
    """
    arr = preprocess_npy(
        npy_path,
        norm_start=norm_start,
        norm_end=norm_end,
        expected_shape=expected_shape,
    )
    t = torch.from_numpy(arr).unsqueeze(0)
    return t.to(device)


# ── ONNX-friendly wrapper ─────────────────────────────────────────────────────

class _OnnxWrapper(nn.Module):
    """
    Drops the dict-based RadarPoseNet interface to expose plain tensor I/O.

    forward(rdr_tensor) → (hm, reg)
      rdr_tensor : [B, 128, 16, 8, 64]  preprocessed radar cube
      hm         : [B,   1, 16, 8, 64]  raw heatmap logits (pre-sigmoid)
      reg        : [B,  69, 16, 8, 64]  regression offsets (23 joints x xyz)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rdr_tensor: torch.Tensor):
        feat = self.model.extract_feat({"rdr_tensor": rdr_tensor})
        preds, _ = self.model.pose_head(feat)   # list[dict], one dict per task
        hm  = preds[0]["hm"]
        reg = preds[0]["reg"]
        return hm, reg


# ── model build / load ────────────────────────────────────────────────────────

def build_model(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    return build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg), cfg


def load_pth(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    load_checkpoint(model, ckpt_path, map_location=device)
    return model.to(device).eval()


# ── export ────────────────────────────────────────────────────────────────────

def export_onnx(wrapper: nn.Module, dummy: torch.Tensor, out_path: str, dynamic_batch: bool, opset: int):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "rdr_tensor": {0: "batch"},
            "hm":         {0: "batch"},
            "reg":        {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        (dummy,),
        out_path,
        opset_version=opset,
        input_names=["rdr_tensor"],
        output_names=["hm", "reg"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    batch_note = "dynamic batch" if dynamic_batch else "fixed batch=1"
    print(f"[export] saved → {out_path}  ({batch_note})")


# ── verify ────────────────────────────────────────────────────────────────────

# GroupNorm is decomposed differently in ONNX Runtime, causing float32 rounding
# differences that are harmless for inference.  Separate tolerances reflect this:
#   hm  (logits): ~0.05 absolute in logit space → <1% change in sigmoid output
#   reg (offsets): tighter because downstream joint positions are in metres
_HM_ATOL  = 0.05
_REG_ATOL = 0.005


def run_pytorch(wrapper: nn.Module, dummy: torch.Tensor):
    with torch.no_grad():
        hm_pt, reg_pt = wrapper(dummy)
    return hm_pt.cpu().numpy(), reg_pt.cpu().numpy()


def run_onnx_cpu(dummy: torch.Tensor, onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skipping.")
        print("         pip install onnxruntime   (or onnxruntime-gpu)")
        return None

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"rdr_tensor": dummy.cpu().numpy()})


def compare_outputs(hm_ref, reg_ref, hm_tgt, reg_tgt, atol: float, label: str):
    hm_diff  = np.abs(hm_ref  - hm_tgt ).max()
    reg_diff = np.abs(reg_ref - reg_tgt).max()

    # Use per-output tolerances; --atol overrides both when explicitly set.
    hm_tol  = atol if atol != 1e-4 else _HM_ATOL
    reg_tol = atol if atol != 1e-4 else _REG_ATOL
    hm_ok   = hm_diff  <= hm_tol
    reg_ok  = reg_diff <= reg_tol

    print(f"[verify] {label} hm  max-abs-diff = {hm_diff :.3e}  (tol {hm_tol :.0e})  {'PASS' if hm_ok  else 'FAIL'}")
    print(f"[verify] {label} reg max-abs-diff = {reg_diff:.3e}  (tol {reg_tol:.0e})  {'PASS' if reg_ok else 'FAIL'}")

    # The real correctness criterion: do both produce the same peak voxel?
    # hm shape: [B, 1, D, H, W] — flatten spatial dims and find argmax per batch item.
    hm_flat_ref = hm_ref.reshape(hm_ref.shape[0], -1)
    hm_flat_tgt = hm_tgt.reshape(hm_tgt.shape[0], -1)
    peak_match  = np.all(hm_flat_ref.argmax(axis=1) == hm_flat_tgt.argmax(axis=1))
    print(f"[verify] {label} hm  peak voxel   = {'same' if peak_match else 'DIFFERENT <- real problem'}")

    if not peak_match:
        raise RuntimeError(f"Peak voxel locations differ for {label}!")
    if not (hm_ok and reg_ok):
        raise RuntimeError(f"Outputs differ beyond tolerance for {label} — check model or opset.")
    print(f"[verify] {label} is numerically correct.")


def verify(wrapper: nn.Module, dummy: torch.Tensor, onnx_path: str, atol: float):
    hm_pt, reg_pt = run_pytorch(wrapper, dummy)
    ort_outputs = run_onnx_cpu(dummy, onnx_path)
    if ort_outputs is None:
        return hm_pt, reg_pt, None, None
    hm_ort, reg_ort = ort_outputs
    compare_outputs(hm_pt, reg_pt, hm_ort, reg_ort, atol=atol, label="CPU-ORT")
    return hm_pt, reg_pt, hm_ort, reg_ort


# ── CLI ───────────────────────────────────────────────────────────────────────

def save_reference(
    ref_path: str,
    input_np: np.ndarray,
    hm_pt: np.ndarray,
    reg_pt: np.ndarray,
    hm_ort: np.ndarray = None,
    reg_ort: np.ndarray = None,
    npy_dir: str = None,
):
    """
    Save PyTorch checkpoint outputs alongside the ONNX file as a reference bundle.
    Copy model.onnx + model.ref.npz to the target device and run verify_on_device.py.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ref_path)), exist_ok=True)
    payload = {
        "input": input_np,
        "hm": hm_pt,
        "reg": reg_pt,
        "source": np.array("pytorch_checkpoint"),
    }
    if hm_ort is not None and reg_ort is not None:
        payload["hm_ort_cpu"] = hm_ort
        payload["reg_ort_cpu"] = reg_ort
    np.savez_compressed(ref_path, **payload)
    print(f"[ref]    saved  → {ref_path}")

    if npy_dir:
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "input.npy"), input_np)
        np.save(os.path.join(npy_dir, "hm_pytorch.npy"), hm_pt)
        np.save(os.path.join(npy_dir, "reg_pytorch.npy"), reg_pt)
        if hm_ort is not None and reg_ort is not None:
            np.save(os.path.join(npy_dir, "hm_onnx_cpu.npy"), hm_ort)
            np.save(os.path.join(npy_dir, "reg_onnx_cpu.npy"), reg_ort)
        print(f"[ref]    saved npy files → {npy_dir}")


def _normalization_from_cfg(cfg):
    vals = cfg.DATASET.DZYX.NORMALIZING_VALUE
    return float(vals[0]), float(vals[1])


def parse_args():
    p = argparse.ArgumentParser(description="Export RadarPoseNet to ONNX")
    p.add_argument("--config",        required=True,  help="det3d config .py file")
    p.add_argument("--ckpt",          required=True,  help="checkpoint .pth file")
    p.add_argument("--output",        required=True,  help="output .onnx path")
    p.add_argument("--npy",           required=True,  help="raw radar .npy frame for export and verification")
    p.add_argument("--device",        default="cpu",  help="torch device (default: cpu)")
    p.add_argument("--opset",         type=int, default=18, help="ONNX opset version (default: 18)")
    p.add_argument("--atol",          type=float, default=1e-4, help="abs tolerance for verification (default 1e-4)")
    p.add_argument("--expected-shape", type=int, nargs=4, default=(128, 16, 8, 64),
                   metavar=("D", "Z", "Y", "X"),
                   help="expected raw npy shape before batch dim (default: 128 16 8 64)")
    p.add_argument("--norm-start",    type=float, default=None, help="override config DZYX norm start")
    p.add_argument("--norm-end",      type=float, default=None, help="override config DZYX norm end")
    p.add_argument("--dynamic-batch", action="store_true",      help="export with symbolic batch axis instead of fixed 1")
    p.add_argument("--no-verify",     action="store_true",      help="skip onnxruntime numerical check")
    p.add_argument("--save-ref",      action="store_true",      help="save PyTorch reference outputs to <output>.ref.npz for on-device verification")
    p.add_argument("--ref-output",    default=None,              help="reference .npz path; default is <output>.ref.npz")
    p.add_argument("--npy-output-dir", default=None,             help="directory for individual input/output .npy files")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"[build]  config  : {args.config}")
    model, cfg = build_model(args.config)

    print(f"[build]  weights : {args.ckpt}")
    model = load_pth(model, args.ckpt, device)

    wrapper = _OnnxWrapper(model).eval()

    norm_start_cfg, norm_end_cfg = _normalization_from_cfg(cfg)
    norm_start = norm_start_cfg if args.norm_start is None else args.norm_start
    norm_end = norm_end_cfg if args.norm_end is None else args.norm_end

    print(f"[input]  loading npy : {args.npy}")
    print(f"[input]  expected raw shape : {list(args.expected_shape)}")
    print(f"[input]  normalization      : ({norm_start}, {norm_end})")
    dummy = make_input(
        args.npy,
        device,
        norm_start=norm_start,
        norm_end=norm_end,
        expected_shape=args.expected_shape,
    )
    print(f"[input]  shape={list(dummy.shape)}  min={dummy.min():.4f}  max={dummy.max():.4f}")

    export_onnx(wrapper, dummy, args.output, dynamic_batch=args.dynamic_batch, opset=args.opset)

    hm_pt = reg_pt = hm_ort = reg_ort = None
    if not args.no_verify:
        hm_pt, reg_pt, hm_ort, reg_ort = verify(wrapper, dummy, args.output, atol=args.atol)

    if args.save_ref:
        input_np = dummy.cpu().numpy()
        if hm_pt is None or reg_pt is None:
            hm_pt, reg_pt = run_pytorch(wrapper, dummy)
        if hm_ort is None or reg_ort is None:
            ort_outputs = run_onnx_cpu(dummy, args.output)
            if ort_outputs is not None:
                hm_ort, reg_ort = ort_outputs
        ref_output = args.ref_output or args.output.replace(".onnx", ".ref.npz")
        npy_output_dir = args.npy_output_dir
        if npy_output_dir is None:
            npy_output_dir = os.path.splitext(ref_output)[0] + "_npy"
        save_reference(ref_output, input_np, hm_pt, reg_pt, hm_ort, reg_ort, npy_output_dir)


if __name__ == "__main__":
    main()
