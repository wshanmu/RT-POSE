"""
Export a TemporalRadarPoseNet checkpoint as two ONNX graphs:

1. backbone ONNX:
       rdr_tensor -> backbone_feature
       [1, D, Z, Y, X] -> [1, C, Zf, Yf, Xf]

2. temporal head ONNX:
       feature_window -> hm, reg
       [1, T, C, Zf, Yf, Xf] -> [1, 1, Zf, Yf, Xf], [1, 69, Zf, Yf, Xf]

This matches a low-latency Jetson deployment pattern:
    run backbone once for each new radar frame,
    push the feature into a ring buffer,
    run temporal_head on the cached feature window.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("RTPOSE_DISABLE_IOU3D", "1")
os.environ.setdefault("RTPOSE_DISABLE_SPCONV", "1")

from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint


class _BackboneOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.reader = model.reader
        self.backbone = model.backbone

    def forward(self, rdr_tensor):
        x = self.reader(rdr_tensor)
        return self.backbone(x)


class _TemporalHeadOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.temporal_neck = model.temporal_neck
        self.pose_head = model.pose_head

    def forward(self, feature_window):
        feat = self.temporal_neck(feature_window)
        preds, _ = self.pose_head(feat)
        hm = preds[0]["hm"]
        reg = preds[0]["reg"]
        return hm, reg


def _frame_sort_key(path):
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _normalizing_values(cfg):
    vals = cfg.DATASET.DZYX.NORMALIZING_VALUE
    return float(vals[0]), float(vals[1])


def preprocess_npy(path, cfg):
    norm_start, norm_end = _normalizing_values(cfg)
    arr = np.load(path).astype(np.float32)
    arr = np.log1p(arr)
    arr = (arr - norm_start) / (norm_end - norm_start)
    arr[arr < 0.0] = 0.0
    return arr


def _radar_dir_from_cfg(cfg):
    return cfg.DATASET.DIR.get("RADAR_NPY_DIR", "DZYX_npy_f16")


def _resolve_window_paths(args, cfg):
    if args.npy:
        npy = Path(args.npy)
        return [npy for _ in range(args.window_size)]

    session_dir = Path(args.session_dir)
    radar_dir = args.radar_dir or _radar_dir_from_cfg(cfg)
    npy_dir = session_dir / radar_dir
    if args.radar_dir is None and not npy_dir.exists():
        for fallback_name in ["DZYX_npy_f16_all2p5m", "DZYX_npy_f16"]:
            fallback = session_dir / fallback_name
            if fallback.exists():
                print(f"[input]  {npy_dir} not found; using fallback {fallback}")
                npy_dir = fallback
                break

    frame_paths = sorted(npy_dir.glob("*.npy"), key=_frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No npy files found in {npy_dir}")

    idx = args.frame_index
    if args.frame_id is not None:
        matches = [i for i, path in enumerate(frame_paths) if path.stem == args.frame_id]
        if not matches:
            raise FileNotFoundError(f"Frame id {args.frame_id!r} not found in {npy_dir}")
        idx = matches[0]
    if idx < 0 or idx >= len(frame_paths):
        raise IndexError(f"frame index {idx} out of range for {len(frame_paths)} frames")

    window = []
    for offset in range(args.window_size - 1, -1, -1):
        src_idx = max(0, idx - offset)
        window.append(frame_paths[src_idx])
    return window


def build_model(config_path, ckpt_path, device):
    cfg = Config.fromfile(config_path)
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, ckpt_path, map_location=device)
    model.to(device).eval()
    return model, cfg


def _export_backbone(wrapper, dummy, output, dynamic_batch):
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "rdr_tensor": {0: "batch"},
            "backbone_feature": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        (dummy,),
        output,
        opset_version=18,
        input_names=["rdr_tensor"],
        output_names=["backbone_feature"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"[export] backbone      -> {output}")


def _export_temporal_head(wrapper, dummy, output):
    torch.onnx.export(
        wrapper,
        (dummy,),
        output,
        opset_version=18,
        input_names=["feature_window"],
        output_names=["hm", "reg"],
        do_constant_folding=True,
    )
    print(f"[export] temporal head -> {output}")


def _run_onnx_check(backbone_onnx, head_onnx, input_np, feature_window_np, feature_ref, hm_ref, reg_ref):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed; skipping CPU ONNX check.")
        return

    sess_backbone = ort.InferenceSession(backbone_onnx, providers=["CPUExecutionProvider"])
    sess_head = ort.InferenceSession(head_onnx, providers=["CPUExecutionProvider"])
    feature_ort = sess_backbone.run(None, {"rdr_tensor": input_np})[0]
    hm_ort, reg_ort = sess_head.run(None, {"feature_window": feature_window_np})

    print(f"[verify] backbone feature max-abs-diff = {np.abs(feature_ref - feature_ort).max():.3e}")
    print(f"[verify] hm       max-abs-diff = {np.abs(hm_ref - hm_ort).max():.3e}")
    print(f"[verify] reg      max-abs-diff = {np.abs(reg_ref - reg_ort).max():.3e}")

    peak_ref = hm_ref.reshape(hm_ref.shape[0], -1).argmax(axis=1)
    peak_ort = hm_ort.reshape(hm_ort.shape[0], -1).argmax(axis=1)
    if not np.all(peak_ref == peak_ort):
        raise RuntimeError("ONNX CPU check failed: heatmap peak voxel differs")
    print("[verify] hm peak voxel = same")


def _save_ref(path, input_np, feature_window_np, feature_ref, hm_ref, reg_ref, window_paths):
    np.savez_compressed(
        path,
        input=input_np,
        feature_window=feature_window_np,
        backbone_feature=feature_ref,
        hm=hm_ref,
        reg=reg_ref,
        window_paths=np.array([str(p) for p in window_paths]),
    )
    print(f"[ref]    saved         -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Export TemporalRadarPoseNet split ONNX graphs")
    p.add_argument("--config", required=True, help="temporal det3d config .py")
    p.add_argument("--ckpt", required=True, help="temporal checkpoint .pth")
    p.add_argument("--backbone-output", required=True, help="output backbone .onnx")
    p.add_argument("--head-output", required=True, help="output temporal head .onnx")
    p.add_argument("--ref-output", default=None, help="reference .npz path; default next to head ONNX")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--npy", help="single raw radar npy; repeated to fill temporal window")
    group.add_argument("--session-dir", help="session folder containing radar npy files")
    p.add_argument("--radar-dir", default=None, help="override radar npy folder name")
    p.add_argument("--frame-index", type=int, default=0, help="current frame index for session mode")
    p.add_argument("--frame-id", default=None, help="current frame id/stem for session mode")
    p.add_argument("--window-size", type=int, default=None, help="temporal window size")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dynamic-batch", action="store_true", help="dynamic batch only for backbone ONNX")
    p.add_argument("--no-verify", action="store_true", help="skip CPU onnxruntime check")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, cfg = build_model(args.config, args.ckpt, device)

    if args.window_size is None:
        temporal_cfg = cfg.DATASET.get("TEMPORAL", {})
        args.window_size = int(temporal_cfg.get("WINDOW_SIZE", 1))
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")

    window_paths = _resolve_window_paths(args, cfg)
    window_np = np.stack([preprocess_npy(path, cfg) for path in window_paths], axis=0)
    current_np = window_np[-1][None, ...]
    current = torch.from_numpy(current_np).to(device)

    backbone = _BackboneOnnxWrapper(model).eval()
    head = _TemporalHeadOnnxWrapper(model).eval()

    with torch.no_grad():
        features = []
        for i in range(window_np.shape[0]):
            frame = torch.from_numpy(window_np[i][None, ...]).to(device)
            features.append(backbone(frame))
        feature_window = torch.stack(features, dim=1)
        feature_ref = features[-1].cpu().numpy()
        hm_ref, reg_ref = head(feature_window)

    feature_window_np = feature_window.cpu().numpy()
    hm_ref_np = hm_ref.cpu().numpy()
    reg_ref_np = reg_ref.cpu().numpy()

    print(f"[input]  current frame  : {window_paths[-1]}")
    print(f"[input]  window size    : {args.window_size}")
    print(f"[input]  rdr_tensor     : {list(current_np.shape)}")
    print(f"[input]  feature_window : {list(feature_window_np.shape)}")
    print(f"[input]  hm/reg         : {list(hm_ref_np.shape)} / {list(reg_ref_np.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.backbone_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.head_output)), exist_ok=True)
    _export_backbone(backbone, current, args.backbone_output, args.dynamic_batch)
    _export_temporal_head(head, feature_window, args.head_output)

    if not args.no_verify:
        _run_onnx_check(
            args.backbone_output,
            args.head_output,
            current_np,
            feature_window_np,
            feature_ref,
            hm_ref_np,
            reg_ref_np,
        )

    ref_output = args.ref_output
    if ref_output is None:
        ref_output = args.head_output.replace(".onnx", ".ref.npz")
    _save_ref(ref_output, current_np, feature_window_np, feature_ref, hm_ref_np, reg_ref_np, window_paths)


if __name__ == "__main__":
    main()
