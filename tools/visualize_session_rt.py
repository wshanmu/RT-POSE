"""
Per-frame 3D skeleton visualization using the TRT-split ONNX model on Jetson.

Replaces the old .engine-based backend with ORT TensorRT EP + CUDA fallback,
using the 63-output split model that works around the sm87 5-D Resize bug.

Usage — whole session:
    python tools/visualize_session_rt.py \
        --onnx  RT-POSE/tools/rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
        --config configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
        --session-dir /path/to/synchronized/session1 \
        --out-dir /tmp/viz_session1

Usage — single npy frame:
    python tools/visualize_session_rt.py \
        --onnx  RT-POSE/tools/rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx \
        --config configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
        --npy-file /path/to/DZYX_npy_f16/00042.npy \
        --out-dir /tmp/viz_frame

Output PNGs can be assembled into a video with ffmpeg:
    ffmpeg -framerate 10 -i /tmp/viz_session1/%05d.png -c:v libx264 out.mp4

TensorRT note
-------------
The split model (rtpose_*_trt_split.onnx) has 63 graph outputs which force
TRT EP to partition into small subgraphs, placing the 5-D trilinear Resize
nodes at boundaries where CUDA EP handles them correctly (Jetson sm87 fix).
At runtime we call sess.run(["hm","reg"]) so only 2 output tensors are ever
allocated — the 61 probe boundary tensors are ORT-internal GPU buffers.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set BEFORE onnxruntime (and therefore before any import that pulls it
# in transitively).  TRT engines cache this value at build time; if it differs
# at runtime TRT raises "Inconsistent setting of NVIDIA_TF32_OVERRIDE env var".
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rtpose_trt_runtime import (
    DEFAULT_CONFIG,
    decode,
    format_runtime_spec,
    load_runtime_spec,
    preprocess,
)


# ---------------------------------------------------------------------------
# ORT TensorRT EP inference  (replaces the old .engine + PyTorch backend)
# ---------------------------------------------------------------------------

class OrtTrtInference:
    """ORT inference using TensorRT EP + CUDA EP fallback.

    Key design:
    • The ONNX model must be the TRT-split variant (63 graph outputs) produced
      by create_full_probe_model.py.  Those extra outputs force TRT EP to split
      its engine at InstanceNorm/Resize boundaries, putting the sm87-buggy
      Resize kernel at subgraph edges where CUDA EP handles it correctly.
    • We call sess.run(["hm","reg"], ...) at every inference — not sess.run(None).
      The probe boundary tensors still flow between TRT engines as ORT-internal
      GPU buffers; we just don't allocate 61 Python numpy arrays per call.
      This drops latency from ~67 ms to ~20 ms with identical correctness.
    • Works from a cold TRT cache: engine partition is fixed at session creation
      from the model's graph structure, not from the run() output list.
    """

    def __init__(
        self,
        onnx_path: str,
        trt_cache_dir: str = "./trt_cache_viz",
        fp16: bool = True,
        opt_level: int = 2,
    ):
        os.makedirs(trt_cache_dir, exist_ok=True)
        import onnxruntime as ort

        trt_opts = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_dir,
            "trt_timing_cache_enable": True,
            # 2 = safe default for HRNet3D; 3 may break GroupNorm→Conv3D fusion
            "trt_builder_optimization_level": opt_level,
        }
        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        print(f"[ort-trt] loading session from: {onnx_path}")
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        n = len(self.sess.get_outputs())
        mode = "probe/split model — TRT will split correctly" if n > 2 else "standard 2-output model"
        print(f"[ort-trt] active providers : {self.sess.get_providers()}")
        print(f"[ort-trt] graph outputs    : {n}  ({mode})")
        print(f"[ort-trt] fp16={fp16}  opt_level={opt_level}  cache={trt_cache_dir}")

    def infer(self, rdr_tensor: np.ndarray) -> dict:
        """Run inference on a preprocessed float32 array [1, D, Z, Y, X].

        Returns dict with keys 'hm' and 'reg' as float32 numpy arrays.
        """
        rdr_tensor = np.ascontiguousarray(rdr_tensor, dtype=np.float32)
        hm, reg = self.sess.run(["hm", "reg"], {"rdr_tensor": rdr_tensor})
        return {"hm": hm, "reg": reg}


# ---------------------------------------------------------------------------
# Keypoint / skeleton definitions
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "Nose",       # 0
    "LEye",       # 1
    "REye",       # 2
    "LEar",       # 3
    "REar",       # 4
    "LShoulder",  # 5
    "RShoulder",  # 6
    "LElbow",     # 7
    "RElbow",     # 8
    "LWrist",     # 9
    "RWrist",     # 10
    "LHip",       # 11
    "RHip",       # 12
    "LKnee",      # 13
    "RKnee",      # 14
    "LAnkle",     # 15
    "RAnkle",     # 16
    "LBigToe",    # 17
    "LSmallToe",  # 18
    "LHeel",      # 19
    "RBigToe",    # 20
    "RSmallToe",  # 21
    "RHeel",      # 22
]

SKELETON_EDGES = [
    # head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # left arm
    (5, 7), (7, 9),
    # right arm
    (6, 8), (8, 10),
    # torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # left leg
    (11, 13), (13, 15),
    # right leg
    (12, 14), (14, 16),
    # left foot
    (15, 17), (17, 18), (15, 19),
    # right foot
    (16, 20), (20, 21), (16, 22),
]

ROI = {"x": (1.0, 3.0), "y": (-1.25, 1.25), "z": (-1.15, 1.40)}
_NUM_KP = 23


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _infer_single_npy(npy_path, model, spec):
    """Run inference on one npy file.

    Returns:
        arr       : preprocessed numpy array (D, Z, Y, X)
        pred_kps  : list of (joint_id, x, y, z)
        score     : float confidence of the peak voxel
    """
    arr = preprocess(np.load(npy_path), spec)
    inp = arr[np.newaxis]               # [1, D, Z, Y, X]
    outputs = model.infer(inp)
    missing = [name for name in ("hm", "reg") if name not in outputs]
    if missing:
        raise KeyError(f"Model outputs {sorted(outputs)}; missing required {missing}")
    result = decode(outputs["hm"], outputs["reg"], spec)

    kp = result["keypoints"]            # [K, 3]
    pred_kps = [(k, float(kp[k, 0]), float(kp[k, 1]), float(kp[k, 2]))
                for k in range(spec.num_keypoints)]
    return arr, pred_kps, result["score"]


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def _mpjpe_cm(pred_kps, gt_pose):
    if not pred_kps or gt_pose is None:
        return None
    gt = np.array(gt_pose)
    pred_by_id = {int(kp[0]): kp[1:4] for kp in pred_kps}
    errors = [np.linalg.norm(np.array(pred_by_id[j]) - gt[j])
              for j in range(gt.shape[0]) if j in pred_by_id]
    return float(np.mean(errors) * 100) if errors else None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_L_COLOR = "#2979FF"
_R_COLOR = "#FF3D00"
_C_COLOR = "#37474F"

_LEFT_JOINTS  = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19}
_RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22}


def _lr_edge_color(i, j):
    li, ri = i in _LEFT_JOINTS, i in _RIGHT_JOINTS
    lj, rj = j in _LEFT_JOINTS, j in _RIGHT_JOINTS
    if (li or lj) and not (ri or rj):
        return _L_COLOR
    if (ri or rj) and not (li or lj):
        return _R_COLOR
    return _C_COLOR


def _lr_joint_color(idx):
    if idx in _LEFT_JOINTS:
        return _L_COLOR
    if idx in _RIGHT_JOINTS:
        return _R_COLOR
    return _C_COLOR


def _radar_projections(arr):
    """Project (D, Z, Y, X) tensor into three 2D max-projection planes."""
    vol = arr.max(axis=0)
    log_scale = lambda v: np.log1p(v)
    return (log_scale(vol.max(axis=0)),   # (Y, X)  bird's-eye
            log_scale(vol.max(axis=1)),   # (Z, X)  side
            log_scale(vol.max(axis=2)))   # (Z, Y)  front


def _draw_skeleton_2d(ax, joints_xyz, plane, color, alpha=0.85, lw=1.5):
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    n   = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n:
            continue
        ax.plot([joints_xyz[i][idx[0]], joints_xyz[j][idx[0]]],
                [joints_xyz[i][idx[1]], joints_xyz[j][idx[1]]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(joints_xyz[:, idx[0]], joints_xyz[:, idx[1]],
               s=10, c=color, zorder=5, alpha=alpha)


def _draw_skeleton_3d(ax, joints_xyz, color, alpha=0.85, lw=1.5):
    n = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n:
            continue
        ax.plot([joints_xyz[i][0], joints_xyz[j][0]],
                [joints_xyz[i][1], joints_xyz[j][1]],
                [joints_xyz[i][2], joints_xyz[j][2]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2],
               s=15, c=color, zorder=5, alpha=alpha)


def _draw_3d_skel(ax, joints_xyz, alpha=0.9, lw=2.0, ls="-", dot_size=30):
    n = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n:
            continue
        ax.plot([joints_xyz[i][0], joints_xyz[j][0]],
                [joints_xyz[i][1], joints_xyz[j][1]],
                [joints_xyz[i][2], joints_xyz[j][2]],
                color=_lr_edge_color(i, j), lw=lw, alpha=alpha, linestyle=ls)
    jcolors = [_lr_joint_color(k) for k in range(n)]
    ax.scatter(joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2],
               c=jcolors, s=dot_size, zorder=6, alpha=alpha, depthshade=False)


def _style_3d_ax(ax, title, elev, azim):
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.tick_params(colors="#666666", labelsize=6)
    ax.set_xlabel("x depth (m)",   color="#555555", fontsize=7, labelpad=1)
    ax.set_ylabel("y lateral (m)", color="#555555", fontsize=7, labelpad=1)
    ax.set_zlabel("z height (m)",  color="#555555", fontsize=7, labelpad=1)
    ax.set_title(title, fontsize=9, color="#222222", pad=4)
    ax.set_xlim(*ROI["x"])
    ax.set_ylim(*ROI["y"])
    ax.set_zlim(*ROI["z"])
    ax.view_init(elev=elev, azim=azim)


def _kps_to_xyz(pred_kps):
    """Convert list of (joint_id, x, y, z) → (N, 3) numpy array."""
    if not pred_kps:
        return None
    n = max(k[0] for k in pred_kps) + 1
    arr = np.zeros((n, 3))
    for k in pred_kps:
        arr[k[0]] = [k[1], k[2], k[3]]
    return arr


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def make_skeleton_frame(pred_kps, gt_pose, frame_id, mpjpe_cm_val, score):
    """Three 3D-axis views on a white background (compact skeleton-only figure)."""
    pred_xyz = _kps_to_xyz(pred_kps)
    gt_xyz   = np.array(gt_pose) if gt_pose is not None and len(gt_pose) == _NUM_KP else None

    fig = plt.figure(figsize=(15, 5), facecolor="white")

    title_parts = [f"Frame {frame_id}", f"score={score:.3f}"]
    if mpjpe_cm_val is not None:
        title_parts.append(f"Abs-MPJPE {mpjpe_cm_val:.1f} cm")
    fig.suptitle("   |   ".join(title_parts), fontsize=11, color="#222222", y=1.01)

    views = [
        (25,  -60, "3D Perspective"),
        ( 0,  -90, "Front  (Y–Z)"),
        ( 0,    0, "Side   (X–Z)"),
    ]
    for col, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        _style_3d_ax(ax, title, elev, azim)
        if gt_xyz is not None:
            _draw_3d_skel(ax, gt_xyz,   alpha=0.35, lw=1.5, ls="--", dot_size=12)
        if pred_xyz is not None:
            _draw_3d_skel(ax, pred_xyz, alpha=0.95, lw=2.0, ls="-",  dot_size=30)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_L_COLOR, lw=2,   label="Left side"),
        Line2D([0], [0], color=_R_COLOR, lw=2,   label="Right side"),
        Line2D([0], [0], color=_C_COLOR, lw=2,   label="Midline"),
        Line2D([0], [0], color="black",  lw=1.5, ls="--", label="GT"),
        Line2D([0], [0], color="black",  lw=2,   ls="-",  label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               framealpha=0.9, edgecolor="#cccccc", fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return fig


def make_frame(arr, pred_kps, gt_pose, frame_id, mpjpe_cm_val, score):
    """Full figure: radar projections (top row) + three 3D views (bottom row)."""
    xy_img, xz_img, yz_img = _radar_projections(arr)

    pred_xyz = _kps_to_xyz(pred_kps)
    gt_xyz   = np.array(gt_pose) if gt_pose is not None and len(gt_pose) == _NUM_KP else None

    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a1a")

    title_parts = [f"Frame {frame_id}", f"score={score:.3f}"]
    if mpjpe_cm_val is not None:
        title_parts.append(f"Abs-MPJPE {mpjpe_cm_val:.1f} cm")
    fig.suptitle("   |   ".join(title_parts), color="white", fontsize=12, y=0.98)

    cmap = "inferno"

    def _format_ax(ax, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="gray", labelsize=7)
        ax.set_xlabel(xlabel, color="gray", fontsize=8)
        ax.set_ylabel(ylabel, color="gray", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # --- Panel 1: bird's-eye ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(xy_img, origin="lower", cmap=cmap, aspect="auto")
    ax1.set_title("Bird's-eye (X-Y)", color="white", fontsize=9)
    _format_ax(ax1, "x  depth (m)", "y  lateral (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax1, pred_xyz, "xy", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax1, gt_xyz,   "xy", color="yellow", lw=1.2, alpha=0.6)

    # --- Panel 2: side ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(xz_img, origin="lower", cmap=cmap, aspect="auto")
    ax2.set_title("Side (X-Z)", color="white", fontsize=9)
    _format_ax(ax2, "x  depth (m)", "z  height (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax2, pred_xyz, "xz", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax2, gt_xyz,   "xz", color="yellow", lw=1.2, alpha=0.6)

    # --- Panel 3: front ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(yz_img, origin="lower", cmap=cmap, aspect="auto")
    ax3.set_title("Front (Y-Z)", color="white", fontsize=9)
    _format_ax(ax3, "y  lateral (m)", "z  height (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax3, pred_xyz, "yz", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax3, gt_xyz,   "yz", color="yellow", lw=1.2, alpha=0.6)

    # --- Panels 4-6: 3D views ---
    for col, (elev, azim, label) in enumerate([
        (30, -60, "3D"),
        (90, -90, "Top-down"),
        ( 0, -90, "Side"),
    ]):
        ax3d = fig.add_subplot(2, 3, 4 + col, projection="3d")
        ax3d.set_facecolor("#1a1a1a")
        ax3d.set_xlabel("x depth",   color="gray", fontsize=7, labelpad=2)
        ax3d.set_ylabel("y lateral", color="gray", fontsize=7, labelpad=2)
        ax3d.set_zlabel("z height",  color="gray", fontsize=7, labelpad=2)
        ax3d.tick_params(colors="gray", labelsize=6)
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_title(label, color="white", fontsize=9)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        if pred_xyz is not None:
            _draw_skeleton_3d(ax3d, pred_xyz, color="lime")
        if gt_xyz is not None:
            _draw_skeleton_3d(ax3d, gt_xyz,   color="yellow", alpha=0.5, lw=1.2)

    handles = [
        plt.Line2D([0], [0], color="lime",   lw=2, label="Prediction"),
        plt.Line2D([0], [0], color="yellow", lw=2, label="GT"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RT-Pose predictions (ORT TensorRT EP, split ONNX model)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir",
        help="Session folder containing DZYX_npy_f16/ and (optionally) Train.json")
    group.add_argument("--npy-file", help="Single radar npy frame")

    parser.add_argument("--onnx", required=True,
        help="Path to the TRT-split ONNX model "
             "(e.g. rtpose_dzyx_y8/rtpose_dzyx_y8_trt_split.onnx). "
             "Created by create_full_probe_model.py.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
        help="Training config used for preprocessing/decode")
    parser.add_argument("--trt-cache-dir", default="./trt_cache_viz",
        help="TRT engine cache directory (default: ./trt_cache_viz)")
    parser.add_argument("--no-fp16", action="store_true",
        help="Disable FP16 (use FP32; ~4× slower, slightly more accurate numerics)")
    parser.add_argument("--trt-opt-level", type=int, default=2, choices=[0, 1, 2, 3],
        help="TRT builder optimization level (default 2; 3 may break GroupNorm→Conv3D)")
    parser.add_argument("--out-dir", default=None,
        help="Output directory for PNGs (default: <session_dir>/visualizations_rt/)")
    parser.add_argument("--no-gt", action="store_true",
        help="Skip ground-truth overlay")
    parser.add_argument("--max-frames", type=int, default=None,
        help="Limit frames to process (quick sanity check)")
    parser.add_argument("--label-file", default="Train.json",
        help="Label filename inside session dir (default: Train.json)")
    parser.add_argument("--skeleton-only", action="store_true",
        help="Save compact 3-view skeleton-only image instead of full radar figure")
    return parser.parse_args()


def main():
    args = parse_args()

    spec = load_runtime_spec(args.config)
    global ROI, _NUM_KP
    ROI = spec.roi
    _NUM_KP = spec.num_keypoints
    print("Runtime settings:")
    print(format_runtime_spec(spec))
    print()

    fp16 = not args.no_fp16
    model = OrtTrtInference(
        onnx_path=args.onnx,
        trt_cache_dir=args.trt_cache_dir,
        fp16=fp16,
        opt_level=args.trt_opt_level,
    )
    print("Model ready.\n")

    # --- Collect frames ---
    frames = []   # list of (frame_id, npy_path, gt_pose_or_None)

    if args.npy_file:
        npy_path = Path(args.npy_file)
        frames.append((npy_path.stem, str(npy_path), None))
        out_dir = Path(args.out_dir) if args.out_dir else npy_path.parent.parent / "visualizations_rt"

    else:
        session_dir = Path(args.session_dir).resolve()

        # Auto-detect: user may have passed the DZYX_npy_f16/ directory itself
        # (e.g. --session-dir .../session6/DZYX_npy_f16).  We handle both forms.
        if (session_dir / "DZYX_npy_f16").exists():
            npy_dir = session_dir / "DZYX_npy_f16"
        elif list(session_dir.glob("*.npy")):
            # session_dir already IS the npy directory
            npy_dir     = session_dir
            session_dir = session_dir.parent
            print(f"[info]   --session-dir points to the npy folder directly; "
                  f"using parent {session_dir} as session root")
        else:
            # Fall back to default and let the error below explain clearly
            npy_dir = session_dir / "DZYX_npy_f16"

        out_dir = Path(args.out_dir) if args.out_dir else session_dir / "visualizations_rt"

        gt_lookup = {}
        if not args.no_gt:
            label_path = session_dir / args.label_file
            if label_path.exists():
                with open(label_path) as f:
                    gt_data = json.load(f)
                for _, seq_frames in gt_data.items():
                    for fid, objs in seq_frames.items():
                        pose = objs[0].get("pose", []) if objs else []
                        gt_lookup[fid] = pose if len(pose) == _NUM_KP else None
                print(f"Loaded GT from {label_path}  ({len(gt_lookup)} frames)")
            else:
                print(f"No {args.label_file} found in {session_dir}; skipping GT")

        npy_files = sorted(npy_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No npy files found in {npy_dir}\n"
                f"  Tip: --session-dir should point to the session folder that "
                f"*contains* DZYX_npy_f16/, not to DZYX_npy_f16/ itself.\n"
                f"  Both forms are now accepted automatically."
            )

        for npy_path in npy_files:
            fid = npy_path.stem
            frames.append((fid, str(npy_path),
                           gt_lookup.get(fid) if not args.no_gt else None))

    if args.max_frames:
        frames = frames[:args.max_frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(frames)} frames → {out_dir}")

    # Warm up the TRT engines before benchmarking (builds engine on first call
    # if cache is empty; subsequent calls are fast)
    print("Warming up TRT engine (first call may take a moment to compile) …")
    dummy_path = frames[0][1]
    _infer_single_npy(dummy_path, model, spec)
    print("Warmup done.\n")

    mpjpe_all = []
    latencies = []

    for idx, (frame_id, npy_path, gt_pose) in enumerate(frames):
        t0  = time.perf_counter()
        arr, pred_kps, score = _infer_single_npy(npy_path, model, spec)
        latencies.append((time.perf_counter() - t0) * 1000)

        err = _mpjpe_cm(pred_kps, gt_pose)
        if err is not None:
            mpjpe_all.append(err)

        if args.skeleton_only:
            fig = make_skeleton_frame(pred_kps, gt_pose, frame_id, err, score)
        else:
            fig = make_frame(arr, pred_kps, gt_pose, frame_id, err, score)

        out_path = out_dir / f"{idx:05d}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        if (idx + 1) % 50 == 0 or idx == 0:
            mean_err = f"{np.mean(mpjpe_all):.1f} cm" if mpjpe_all else "n/a"
            msg = (f"  [{idx+1}/{len(frames)}]  frame {frame_id}"
                   f"  score={score:.3f}"
                   f"  infer={latencies[-1]:.1f} ms")
            if err is not None:
                msg += f"  err={err:.1f} cm  running_mean={mean_err}"
            print(msg)

    print(f"\nDone. {len(frames)} frames saved to {out_dir}")
    if latencies:
        print(f"Inference  mean={np.mean(latencies):.1f} ms  "
              f"p99={np.percentile(latencies, 99):.1f} ms  "
              f"max={np.max(latencies):.1f} ms")
    if mpjpe_all:
        print(f"Mean Abs-MPJPE over {len(mpjpe_all)} frames: {np.mean(mpjpe_all):.2f} cm")
        print(f"  best: {np.min(mpjpe_all):.2f} cm   worst: {np.max(mpjpe_all):.2f} cm")

    print("\nTo convert to video:")
    print(f"  ffmpeg -framerate 10 -i {out_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    main()
