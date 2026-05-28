"""
Per-frame 3D skeleton visualization using an ONNX model + CUDA execution provider.

Mirrors visualize_session_rt.py but replaces the TensorRT engine with an ONNX
model run through onnxruntime CUDAExecutionProvider (no TensorRT required).

Usage — whole session:
    python tools/visualize_session_CUDA.py \
        --onnx work_dirs/model.onnx \
        --config configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
        --session-dir /path/to/synchronized/boelter_closer_session1 \
        --out-dir /tmp/viz_session1_cuda

Usage — single npy frame:
    python tools/visualize_session_CUDA.py \
        --onnx work_dirs/model.onnx \
        --config configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
        --npy-file /path/to/DZYX_npy_f16/00042.npy \
        --out-dir /tmp/viz_frame_cuda

python ./visualize_session_CUDA.py --session-dir ../../ssd_datas/fitness_data/synchronized/boelter_closer_session1/ --onnx ./model_opset18.onnx --max-frames 100 --skeleton-only 
--out-dir ./vis

Output PNGs can be assembled into a video with ffmpeg:
    ffmpeg -framerate 10 -i /tmp/viz_session1_cuda/%05d.png -c:v libx264 out.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rtpose_trt_runtime import (
    DEFAULT_CONFIG,
    decode as decode_onnx,
    format_runtime_spec,
    load_runtime_spec,
    preprocess as preprocess_onnx,
)

# ---------------------------------------------------------------------------
# Keypoint / skeleton definitions  (identical to visualize_session_rt.py)
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
# ONNX Runtime CUDA inference wrapper
# ---------------------------------------------------------------------------

class ONNXInference:
    """ONNX Runtime inference using CUDAExecutionProvider."""

    def __init__(self, onnx_path: str, provider: str = "CUDA"):
        import onnxruntime as ort

        available = [p.split("ExecutionProvider")[0] for p in ort.get_available_providers()]
        print(f"[onnx] available providers: {available}")

        key = provider.upper()
        if key == "CPU":
            providers = ["CPUExecutionProvider"]
        elif key == "CUDA":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Choose: CPU | CUDA")

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[onnx] session using: {self.sess.get_providers()}")

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def infer(self, rdr_tensor: np.ndarray) -> dict:
        """Run inference on a preprocessed [1, D, Z, Y, X] float32 array."""
        rdr_tensor = np.ascontiguousarray(rdr_tensor, dtype=np.float32)
        outputs = self.sess.run(self.output_names, {self.input_name: rdr_tensor})
        return {name: out for name, out in zip(self.output_names, outputs)}


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _infer_single_npy(npy_path, onnx_model, spec):
    """Run ONNX inference on one npy file.

    Returns:
        arr         : preprocessed numpy array (D, Z, Y, X)
        pred_kps    : list of (joint_id, x, y, z) — same format as other visualize scripts
        score       : float confidence of the peak voxel
        timing_ms   : dict with keys load_ms, preprocess_ms, infer_ms
    """
    t_load = time.perf_counter()
    raw = np.load(npy_path)
    t_pre = time.perf_counter()
    arr = preprocess_onnx(raw, spec)
    t_inf = time.perf_counter()
    inp = arr[np.newaxis]                          # [1, D, Z, Y, X]
    outputs = onnx_model.infer(inp)
    t_done = time.perf_counter()

    missing = [name for name in ("hm", "reg") if name not in outputs]
    if missing:
        raise KeyError(
            f"ONNX model outputs {sorted(outputs)}; missing required {missing}"
        )

    result = decode_onnx(outputs["hm"], outputs["reg"], spec)
    kp = result["keypoints"]                       # [K, 3]
    pred_kps = [(k, float(kp[k, 0]), float(kp[k, 1]), float(kp[k, 2]))
                for k in range(spec.num_keypoints)]

    timing_ms = {
        "load_ms":       (t_pre  - t_load) * 1000,
        "preprocess_ms": (t_inf  - t_pre)  * 1000,
        "infer_ms":      (t_done - t_inf)  * 1000,
    }
    return arr, pred_kps, result["score"], timing_ms


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
# Plotting helpers  (identical to visualize_session_rt.py)
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
# Figure builders  (mirrors visualize_session_rt.py)
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
        description="Visualize RT-Pose ONNX model predictions via CUDA onnxruntime")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir",
        help="Session folder containing DZYX_npy_f16/ and (optionally) Train.json")
    group.add_argument("--npy-file", help="Single radar npy frame")
    parser.add_argument("--onnx", required=True,
        help="ONNX model file (exported by export_onnx.py)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
        help="Training config used for preprocessing/decode (default: %(default)s)")
    parser.add_argument("--provider", default="CUDA", choices=["CUDA", "CPU"],
        help="Execution provider: CUDA (default) or CPU")
    parser.add_argument("--out-dir", default=None,
        help="Output directory for PNGs (default: <session_dir>/visualizations_cuda/)")
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

    print(f"Loading ONNX model: {args.onnx}  [provider: {args.provider}]")
    onnx_model = ONNXInference(args.onnx, provider=args.provider)

    # Warm up CUDA kernels before timing begins.
    # ONNX Runtime with CUDA JIT-compiles kernels on first use; the first few
    # runs can be 50-200 ms slower than steady state even on a warmed-up GPU.
    _inp_meta = onnx_model.sess.get_inputs()[0]
    _dummy_shape = tuple(d if isinstance(d, int) and d > 0 else 1
                         for d in _inp_meta.shape)
    _dummy = np.zeros(_dummy_shape, dtype=np.float32)
    print("Warming up (3 runs) …", end=" ", flush=True)
    for _ in range(3):
        onnx_model.infer(_dummy)
    print("done.\n")

    # --- Collect frames ---
    frames = []   # list of (frame_id, npy_path, gt_pose_or_None)

    if args.npy_file:
        npy_path = Path(args.npy_file)
        frames.append((npy_path.stem, str(npy_path), None))
        out_dir = Path(args.out_dir) if args.out_dir else npy_path.parent.parent / "visualizations_cuda"

    else:
        session_dir = Path(args.session_dir)
        npy_dir     = session_dir / "DZYX_npy_f16"
        out_dir     = Path(args.out_dir) if args.out_dir else session_dir / "visualizations_cuda"

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
            raise FileNotFoundError(f"No npy files found in {npy_dir}")

        for npy_path in npy_files:
            fid = npy_path.stem
            frames.append((fid, str(npy_path),
                           gt_lookup.get(fid) if not args.no_gt else None))

    if args.max_frames:
        frames = frames[:args.max_frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(frames)} frames → {out_dir}")

    mpjpe_all = []
    infer_latencies = []   # pure sess.run() time
    total_latencies = []   # load + preprocess + infer

    for idx, (frame_id, npy_path, gt_pose) in enumerate(frames):
        arr, pred_kps, score, timing_ms = _infer_single_npy(npy_path, onnx_model, spec)
        infer_latencies.append(timing_ms["infer_ms"])
        total_latencies.append(timing_ms["load_ms"] + timing_ms["preprocess_ms"] + timing_ms["infer_ms"])

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
                   f"  load={timing_ms['load_ms']:.1f} ms"
                   f"  pre={timing_ms['preprocess_ms']:.1f} ms"
                   f"  infer={timing_ms['infer_ms']:.1f} ms")
            if err is not None:
                msg += f"  err={err:.1f} cm  running_mean={mean_err}"
            print(msg)

    print(f"\nDone. {len(frames)} frames saved to {out_dir}")
    if infer_latencies:
        print(f"Pure inference (sess.run)  "
              f"mean={np.mean(infer_latencies):.1f} ms  "
              f"p99={np.percentile(infer_latencies, 99):.1f} ms  "
              f"max={np.max(infer_latencies):.1f} ms")
        print(f"Total per-frame (load+pre+infer)  "
              f"mean={np.mean(total_latencies):.1f} ms  "
              f"p99={np.percentile(total_latencies, 99):.1f} ms")
    if mpjpe_all:
        print(f"Mean Abs-MPJPE over {len(mpjpe_all)} frames: {np.mean(mpjpe_all):.2f} cm")
        print(f"  best: {np.min(mpjpe_all):.2f} cm   worst: {np.max(mpjpe_all):.2f} cm")

    print("\nTo convert to video:")
    print(f"  ffmpeg -framerate 10 -i {out_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    main()
