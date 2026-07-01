"""
Per-frame 3D skeleton visualization for a trained RT-Pose model.

Usage — whole session:
    python tools/visualize_session_3D.py \
        --session-dir ../ssd_datas/fitness_data/synchronized/boelter_loc1_session_3 \
        --checkpoint ./work_dirs/hr3d_one_hm_18j_dzyx_leaveout/polar4d/epoch_20.pth \
        --config configs/custom_fitness_body18/hr3d_one_hm_18j_dzyx_leaveout.py \
        --out-dir /tmp/viz_body18 --skeleton-only --max-frames 20

Usage — whole session, batched across 4 GPUs:
    python tools/visualize_session_3D.py \
        --session-dir ../ssd_datas/fitness_data/synchronized/boelter_loc1_session_3 \
        --checkpoint ./work_dirs/hr3d_one_hm_18j_dzyx_leaveout/polar4d/epoch_20.pth \
        --out-dir /tmp/viz_body18 --skeleton-only \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 --batch-size 16

Usage — smooth predictions with a zero-phase Butterworth filter:
    python tools/visualize_session_3D.py \
        --session-dir ../ssd_datas/fitness_data/synchronized/boelter_loc2_session_1 \
        --checkpoint ./work_dirs/hr3d_one_hm_18j_dzyx_leaveout/leaveout_boelter_loc2_train_only_on_printer/epoch_45.pth \
        --out-dir /tmp/viz_body18/loc2_session3 --skeleton-only \
        --devices cuda:0,cuda:2,cuda:3 --batch-size 256 \
        --save-predictions --predictions-file ./work_dirs/hr3d_one_hm_18j_dzyx_leaveout/leaveout_boelter_loc2_train_only_on_printer/predictions_boelter_loc2_session1_epoch45.json \
        --smooth-keypoints --fps 30 --filter-cutoff-hz 7 --filter-order 4 --max-frames 20

Usage — export predictions + GT for offline post-processing:
    python tools/visualize_session_3D.py \
        --session-dir ../ssd_datas/fitness_data/synchronized/boelter_loc1_session_3 \
        --checkpoint ./work_dirs/custom_fitness_body18_leaveout/latest.pth \
        --out-dir /tmp/viz_body18 --skeleton-only \
        --save-predictions --predictions-file /tmp/viz_body18/predictions.json

Usage — single npy frame:
    python tools/visualize_session_3D.py \
        --npy-file /path/to/DZYX_npy_f16/00042.npy \
        --checkpoint work_dirs/custom_fitness_body18_leaveout/latest.pth \
        --out-dir /tmp/viz_frame

Output PNGs can be assembled into a video with ffmpeg:
    ffmpeg -framerate 30 -i /tmp/viz_body18/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p /tmp/viz_body18/output.mp4
"""

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("RTPOSE_DISABLE_SPCONV", "1")
os.environ.setdefault("RTPOSE_DISABLE_IOU3D", "1")

RT_POSE_ROOT = Path(__file__).resolve().parents[1]
if str(RT_POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(RT_POSE_ROOT))

import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint

# ---------------------------------------------------------------------------
# BODY_18 keypoints used by skeleton_smoothed_by_timestamp.json and the
# custom_fitness_body18 configs. Coordinates are RT-POSE/radar XYZ:
#   x = depth/forward, y = lateral, z = height/up, all in metres.
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "Nose",           # 0
    "Neck",           # 1
    "RShoulder",      # 2
    "RElbow",         # 3
    "RWrist",         # 4
    "LShoulder",      # 5
    "LElbow",         # 6
    "LWrist",         # 7
    "RHip",           # 8
    "RKnee",          # 9
    "RAnkle",         # 10
    "LHip",           # 11
    "LKnee",          # 12
    "LAnkle",         # 13
    "REye",           # 14
    "LEye",           # 15
    "REar",           # 16
    "LEar",           # 17
]

# (parent, child) index pairs to draw limb lines.
SKELETON_EDGES = [
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17),
]
EXPECTED_NUM_KEYPOINTS = len(JOINT_NAMES)
KEYPOINT_FORMAT = "BODY_18"
COORDINATE_FRAME = {
    "x": "depth_forward_m",
    "y": "lateral_m",
    "z": "height_up_m",
}

# Colour coding: left side blue, right side red, centre green.
def _edge_color(i, j):
    left_joints = {5, 6, 7, 11, 12, 13, 15, 17}
    right_joints = {2, 3, 4, 8, 9, 10, 14, 16}
    if i in left_joints or j in left_joints:
        return "tab:blue"
    if i in right_joints or j in right_joints:
        return "tab:red"
    return "tab:green"


# ---------------------------------------------------------------------------
# Radar axes (from config)
# ---------------------------------------------------------------------------
ROI = {"x": (0.9, 3.5), "y": (-1.25, 1.25), "z": (-1.15, 1.40)}   # depth, lateral, height


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_visualization_spec_from_config(cfg):
    """Update display metadata from the selected config when available."""
    global ROI, JOINT_NAMES, EXPECTED_NUM_KEYPOINTS

    keypoint_names = _cfg_get(cfg, "BODY18_KEYPOINT_NAMES", None)
    if keypoint_names is not None:
        JOINT_NAMES = [str(name) for name in keypoint_names]
        EXPECTED_NUM_KEYPOINTS = len(JOINT_NAMES)
    else:
        config_num_keypoints = _cfg_get(cfg, "NUM_KEYPOINTS", None)
        if config_num_keypoints is not None:
            EXPECTED_NUM_KEYPOINTS = int(config_num_keypoints)

    dataset = _cfg_get(cfg, "DATASET", None)
    label_cfg = _cfg_get(dataset, "LABEL", {})
    roi_type = _cfg_get(label_cfg, "ROI_TYPE", "roi1")
    roi_cfg = _cfg_get(_cfg_get(dataset, "ROI", {}), roi_type, None)
    if roi_cfg is not None:
        ROI = {
            "x": tuple(float(v) for v in _cfg_get(roi_cfg, "x", ROI["x"])),
            "y": tuple(float(v) for v in _cfg_get(roi_cfg, "y", ROI["y"])),
            "z": tuple(float(v) for v in _cfg_get(roi_cfg, "z", ROI["z"])),
        }


def _radar_npy_dir_from_config(cfg):
    dataset = _cfg_get(cfg, "DATASET", None)
    dir_cfg = _cfg_get(dataset, "DIR", {})
    return str(_cfg_get(dir_cfg, "RADAR_NPY_DIR", "DZYX_npy_f16"))


def _load_model(config_path, checkpoint_path, device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    cfg = Config.fromfile(config_path)
    _set_visualization_spec_from_config(cfg)
    # Use -1 threshold so the heatmap peak is always accepted, even when
    # sigmoid underflows to 0.0 (undertrained models with very negative logits).
    cfg.test_cfg.score_threshold = -1.0
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model, cfg


def _preprocess_npy(npy_path, cfg):
    arr = np.load(npy_path).astype(np.float32)   # (D, Z, Y, X)
    # Must match get_cube exactly: log1p first, then linear scale to [0, 1].
    norm_start = float(cfg.DATASET.DZYX.NORMALIZING_VALUE[0])
    norm_scale = float(cfg.DATASET.DZYX.NORMALIZING_VALUE[1]) - norm_start
    arr = np.log1p(arr)
    arr = (arr - norm_start) / norm_scale
    arr[arr < 0.0] = 0.0
    return arr


def _infer_batch_npy(batch_items, model, cfg, device, seq="unknown"):
    """Run inference on a batch of npy files.

    Returns list of (idx, frame_id, arr, keypoints) in the same order as
    batch_items. Each keypoint is (joint_id, x, y, z, score).
    """
    arrays = []
    metas = []
    for _, frame_id, npy_path, _ in batch_items:
        arrays.append(_preprocess_npy(npy_path, cfg))
        metas.append({"seq": seq, "frame": frame_id, "rdr_frame": frame_id})

    rdr_tensor = torch.from_numpy(np.stack(arrays, axis=0)).to(device, non_blocking=True)
    example = {
        "rdr": {"rdr_tensor": rdr_tensor},
        "meta": metas,
    }
    with torch.inference_mode():
        outputs = model(example, return_loss=False)

    if len(outputs) != len(batch_items):
        raise RuntimeError(
            f"Model returned {len(outputs)} outputs for batch size {len(batch_items)}"
        )

    results = []
    for item, arr, output in zip(batch_items, arrays, outputs):
        idx, frame_id, _, _ = item
        results.append((idx, frame_id, arr, output.get("keypoints", [])))
    return results


def _infer_single_npy(npy_path, model, cfg, device, frame_id="00000", seq="unknown"):
    """Run inference on one npy file. Returns list of (joint_id, x, y, z, score)."""
    batch_item = (0, frame_id, npy_path, None)
    _, _, arr, keypoints = _infer_batch_npy([batch_item], model, cfg, device, seq)[0]
    # keypoints are (joint_id, x_depth, y_lateral, z_height, score) in metres
    return arr, keypoints


def _keypoints_to_xyz(kps):
    """Convert model output keypoints to an (N, 3) BODY_18 array."""
    if not kps:
        return None
    joints = np.full((EXPECTED_NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
    for kp in kps:
        jid = int(kp[0])
        if 0 <= jid < EXPECTED_NUM_KEYPOINTS:
            joints[jid] = [float(kp[1]), float(kp[2]), float(kp[3])]
    return joints if np.isfinite(joints).any() else None


def _gt_pose_to_xyz(gt_pose):
    """Return GT pose as (N, 3), only when it matches the active keypoint definition."""
    if gt_pose is None:
        return None
    pose = np.asarray(gt_pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 3:
        return None
    if pose.shape[0] != EXPECTED_NUM_KEYPOINTS:
        return None
    pose = pose[:, :3]
    return pose if np.isfinite(pose).any() else None


def _mpjpe_cm(pred_kps, gt_pose):
    """Absolute mean per-joint position error in centimetres.
    Matches joints by ID; joints missing from pred are skipped."""
    if not pred_kps:
        return None
    gt = _gt_pose_to_xyz(gt_pose)
    if gt is None:
        return None
    pred_by_id = {int(kp[0]): np.asarray(kp[1:4], dtype=np.float64) for kp in pred_kps}
    errors = []
    for jid in range(gt.shape[0]):
        if jid in pred_by_id and np.isfinite(gt[jid]).all():
            errors.append(np.linalg.norm(pred_by_id[jid] - gt[jid]))
    if not errors:
        return None
    return float(np.mean(errors) * 100)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _radar_projections(arr):
    """Project (D, Z, Y, X) tensor into three 2D planes via max-over-Doppler then max-over-axis."""
    vol = arr.max(axis=0)          # (Z, Y, X)  — collapse Doppler
    log_scale = lambda v: np.log1p(v)
    xy = log_scale(vol.max(axis=0))   # (Y, X)  bird's-eye
    xz = log_scale(vol.max(axis=1))   # (Z, X)  side
    yz = log_scale(vol.max(axis=2))   # (Z, Y)  front
    return xy, xz, yz


def _draw_skeleton_2d(ax, joints_xyz, plane, color, alpha=0.85, lw=1.5):
    """Draw skeleton edges on a 2D axis. plane is 'xy', 'xz', or 'yz'."""
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    n = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n or not (np.isfinite(joints_xyz[i]).all() and np.isfinite(joints_xyz[j]).all()):
            continue
        xs = [joints_xyz[i][idx[0]], joints_xyz[j][idx[0]]]
        ys = [joints_xyz[i][idx[1]], joints_xyz[j][idx[1]]]
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)
    valid = np.isfinite(joints_xyz).all(axis=1)
    if np.any(valid):
        ax.scatter(joints_xyz[valid, idx[0]], joints_xyz[valid, idx[1]],
                   s=10, c=color, zorder=5, alpha=alpha)


def _draw_skeleton_3d(ax, joints_xyz, color, alpha=0.85, lw=1.5):
    n = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n or not (np.isfinite(joints_xyz[i]).all() and np.isfinite(joints_xyz[j]).all()):
            continue
        xs = [joints_xyz[i][0], joints_xyz[j][0]]
        ys = [joints_xyz[i][1], joints_xyz[j][1]]
        zs = [joints_xyz[i][2], joints_xyz[j][2]]
        ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha)
    valid = np.isfinite(joints_xyz).all(axis=1)
    if np.any(valid):
        ax.scatter(joints_xyz[valid, 0], joints_xyz[valid, 1], joints_xyz[valid, 2],
                   s=15, c=color, zorder=5, alpha=alpha)


def _metric_to_pixel(val, axis_range, n_pixels):
    """Convert metric coordinate to image pixel index."""
    lo, hi = axis_range
    return (val - lo) / (hi - lo) * (n_pixels - 1)


# ---------------------------------------------------------------------------
# Left / right colour palette for the skeleton-only 3-view figure
# ---------------------------------------------------------------------------
_L_COLOR = "#2979FF"   # vivid blue        — left  side
_R_COLOR = "#FF3D00"   # vivid red-orange  — right side
_C_COLOR = "#37474F"   # dark slate        — midline / bilateral

_LEFT_JOINTS = {5, 6, 7, 11, 12, 13, 15, 17}
_RIGHT_JOINTS = {2, 3, 4, 8, 9, 10, 14, 16}


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


def _draw_3d_skel(ax, joints_xyz, alpha=0.9, lw=2.0, ls="-", dot_size=30):
    """Draw skeleton on a 3D axis with per-limb left/right colour coding."""
    n = len(joints_xyz)
    for i, j in SKELETON_EDGES:
        if i >= n or j >= n or not (np.isfinite(joints_xyz[i]).all() and np.isfinite(joints_xyz[j]).all()):
            continue
        ax.plot(
            [joints_xyz[i][0], joints_xyz[j][0]],
            [joints_xyz[i][1], joints_xyz[j][1]],
            [joints_xyz[i][2], joints_xyz[j][2]],
            color=_lr_edge_color(i, j), lw=lw, alpha=alpha, linestyle=ls,
        )
    valid = np.isfinite(joints_xyz).all(axis=1)
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        jcolors = [_lr_joint_color(int(k)) for k in valid_indices]
        ax.scatter(
            joints_xyz[valid, 0], joints_xyz[valid, 1], joints_xyz[valid, 2],
            c=jcolors, s=dot_size, zorder=6, alpha=alpha, depthshade=False,
        )


def _style_3d_ax(ax, title, elev, azim, flip_y=True):
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
    ax.set_ylim(*(ROI["y"][::-1] if flip_y else ROI["y"]))
    ax.set_zlim(*ROI["z"])
    ax.view_init(elev=elev, azim=azim)


def make_skeleton_frame(pred_kps, gt_pose, frame_id, mpjpe_cm_val, flip_y=True):
    """Three 3D-axis views (perspective / front / side) on a white background.

    Left-body limbs: blue.  Right-body limbs: red-orange.  Midline: dark slate.
    GT drawn as dashed semi-transparent lines; prediction as solid opaque lines.
    """
    pred_xyz = _keypoints_to_xyz(pred_kps)
    gt_xyz = _gt_pose_to_xyz(gt_pose)

    fig = plt.figure(figsize=(5, 5), facecolor="white")

    title_parts = [f"Frame {frame_id}"]
    if mpjpe_cm_val is not None:
        title_parts.append(f"Abs-MPJPE {mpjpe_cm_val:.1f} cm")
    fig.suptitle("   |   ".join(title_parts), fontsize=11, color="#222222", y=0.96)

    # elev, azim, title
    # azim=180 views from low X toward higher X (radar/front side).
    views = [
        (25,  180, "3D Perspective"),
        # ( 0,  180, "Front  (Y-Z)"),
        # ( 0,    0, "Side   (X-Z)"),
    ]

    for col, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, len(views), col + 1, projection="3d")
        _style_3d_ax(ax, title, elev, azim, flip_y=flip_y)
        if gt_xyz is not None:
            _draw_3d_skel(ax, gt_xyz,   alpha=0.35, lw=1.5, ls="--", dot_size=12)
        if pred_xyz is not None:
            _draw_3d_skel(ax, pred_xyz, alpha=0.95, lw=2.0, ls="-",  dot_size=30)

    legend_handles = [
        Line2D([0], [0], color=_L_COLOR, lw=2,   label="Left side"),
        Line2D([0], [0], color=_R_COLOR, lw=2,   label="Right side"),
        Line2D([0], [0], color=_C_COLOR, lw=2,   label="Midline"),
        Line2D([0], [0], color="black",  lw=1.5, ls="--", label="GT"),
        Line2D([0], [0], color="black",  lw=2,   ls="-",  label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               framealpha=0.9, edgecolor="#cccccc", fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.08, 1, 0.91])
    return fig


class FastSkeletonFrameRenderer:
    """Reusable skeleton-only renderer.

    This keeps one Matplotlib 3D canvas alive and updates artists in-place.
    It is much faster than constructing a new figure and calling savefig for
    every frame, while preserving the same BODY_18 left/right/GT styling.
    """

    def __init__(self, width=250, height=250, elev=25.0, azim=210.0, flip_y=True):
        dpi = 150
        self.width = int(width)
        self.height = int(height)
        self.num_keypoints = int(EXPECTED_NUM_KEYPOINTS)
        self.blank_xyz = np.full((self.num_keypoints, 3), np.nan, dtype=np.float64)

        self.fig = plt.figure(
            figsize=(self.width / dpi, self.height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        _style_3d_ax(self.ax, "3D Perspective", elev, azim, flip_y=flip_y)
        self.ax.set_box_aspect((1, 1, 1))
        self.frame_title = self.fig.text(
            0.5,
            0.965,
            "",
            ha="center",
            va="top",
            fontsize=11,
            color="#222222",
        )

        self.pred_lines = []
        self.gt_lines = []
        for i, j in SKELETON_EDGES:
            color = _lr_edge_color(i, j)
            pred_line, = self.ax.plot([], [], [], color=color, lw=2.0, alpha=0.95, linestyle="-")
            gt_line, = self.ax.plot([], [], [], color=color, lw=1.5, alpha=0.35, linestyle="--")
            self.pred_lines.append((i, j, pred_line))
            self.gt_lines.append((i, j, gt_line))

        colors = [_lr_joint_color(idx) for idx in range(self.num_keypoints)]
        empty = np.full(self.num_keypoints, np.nan, dtype=np.float64)
        self.pred_scatter = self.ax.scatter(
            empty,
            empty,
            empty,
            c=colors,
            s=46,
            zorder=6,
            alpha=0.95,
            depthshade=False,
            edgecolors="white",
            linewidths=0.6,
        )
        self.gt_scatter = self.ax.scatter(
            empty,
            empty,
            empty,
            c=colors,
            s=22,
            zorder=5,
            alpha=0.45,
            depthshade=False,
            edgecolors="#111827",
            linewidths=0.25,
        )

        legend_handles = [
            Line2D([0], [0], color=_L_COLOR, lw=2,   label="Left side"),
            Line2D([0], [0], color=_R_COLOR, lw=2,   label="Right side"),
            Line2D([0], [0], color=_C_COLOR, lw=2,   label="Midline"),
            Line2D([0], [0], color="black",  lw=1.5, ls="--", label="GT"),
            Line2D([0], [0], color="black",  lw=2,   ls="-",  label="Prediction"),
        ]
        self.fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            framealpha=0.9,
            edgecolor="#cccccc",
            fontsize=8,
            bbox_to_anchor=(0.5, 0.02),
        )
        self.fig.subplots_adjust(left=0.04, right=0.98, bottom=0.13, top=0.86)

    def _normalize_xyz(self, joints_xyz):
        if joints_xyz is None:
            return self.blank_xyz.copy()
        xyz = np.asarray(joints_xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            return self.blank_xyz.copy()
        out = self.blank_xyz.copy()
        count = min(self.num_keypoints, xyz.shape[0])
        out[:count] = xyz[:count, :3]
        return out

    @staticmethod
    def _set_line(line, xyz, i, j):
        if (
            i < xyz.shape[0]
            and j < xyz.shape[0]
            and np.isfinite(xyz[i]).all()
            and np.isfinite(xyz[j]).all()
        ):
            segment = xyz[[i, j]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        else:
            line.set_data([], [])
            line.set_3d_properties([])

    @staticmethod
    def _set_scatter(scatter, xyz):
        scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def render_bgr(self, pred_kps, gt_pose, frame_id, mpjpe_cm_val):
        pred_xyz = self._normalize_xyz(_keypoints_to_xyz(pred_kps))
        gt_xyz = self._normalize_xyz(_gt_pose_to_xyz(gt_pose))

        for i, j, line in self.pred_lines:
            self._set_line(line, pred_xyz, i, j)
        for i, j, line in self.gt_lines:
            self._set_line(line, gt_xyz, i, j)
        self._set_scatter(self.pred_scatter, pred_xyz)
        self._set_scatter(self.gt_scatter, gt_xyz)

        title_parts = [f"Frame {frame_id}"]
        if mpjpe_cm_val is not None:
            title_parts.append(f"Abs-MPJPE {mpjpe_cm_val:.1f} cm")
        self.frame_title.set_text("   |   ".join(title_parts))

        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def save_png(self, out_path, pred_kps, gt_pose, frame_id, mpjpe_cm_val):
        image = self.render_bgr(pred_kps, gt_pose, frame_id, mpjpe_cm_val)
        if not cv2.imwrite(str(out_path), image):
            raise RuntimeError(f"Could not write visualization frame: {out_path}")

    def close(self):
        plt.close(self.fig)


def _render_result_fast(renderer, idx, frame_id, pred_kps, gt_pose, out_dir):
    err = _mpjpe_cm(pred_kps, gt_pose)
    out_path = out_dir / f"{idx:05d}.png"
    renderer.save_png(out_path, pred_kps, gt_pose, frame_id, err)
    return err


def make_frame(arr, pred_kps, gt_pose, frame_id, mpjpe_cm_val, flip_y=True):
    """Render one figure and return it (caller saves/closes)."""
    xy_img, xz_img, yz_img = _radar_projections(arr)

    x_range = ROI["x"]
    y_range = ROI["y"]
    z_range = ROI["z"]
    nZ, nY, nX = arr.shape[1], arr.shape[2], arr.shape[3]

    # Build (N, 3) arrays for pred and gt.
    pred_xyz = _keypoints_to_xyz(pred_kps)
    gt_xyz = _gt_pose_to_xyz(gt_pose)
    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a1a")
    axes = []

    title_parts = [f"Frame {frame_id}"]
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

    # --- Panel 1: bird's-eye  X (depth) vs Y (lateral) ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(xy_img, origin="lower", cmap=cmap, aspect="auto")
            #    extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax1.set_title("Bird's-eye (X-Y)", color="white", fontsize=9)
    _format_ax(ax1, "x  depth (m)", "y  lateral (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax1, pred_xyz, "xy", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax1, gt_xyz, "xy", color="yellow", lw=1.2, alpha=0.6)

    # --- Panel 2: side  X (depth) vs Z (height) ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(xz_img, origin="lower", cmap=cmap, aspect="auto")
            #    extent=[x_range[0], x_range[1], z_range[0], z_range[1]])
    ax2.set_title("Side (X-Z)", color="white", fontsize=9)
    _format_ax(ax2, "x  depth (m)", "z  height (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax2, pred_xyz, "xz", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax2, gt_xyz, "xz", color="yellow", lw=1.2, alpha=0.6)

    # --- Panel 3: front  Y (lateral) vs Z (height) ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(yz_img, origin="lower", cmap=cmap, aspect="auto")
            #    extent=[y_range[0], y_range[1], z_range[0], z_range[1]])
    ax3.set_title("Front (Y-Z)", color="white", fontsize=9)
    _format_ax(ax3, "y  lateral (m)", "z  height (m)")
    if pred_xyz is not None:
        _draw_skeleton_2d(ax3, pred_xyz, "yz", color="lime")
    if gt_xyz is not None:
        _draw_skeleton_2d(ax3, gt_xyz, "yz", color="yellow", lw=1.2, alpha=0.6)

    # --- Panel 4-6: 3D view from three angles ---
    for col, (elev, azim, label) in enumerate(
        [(30, -60, "3D"), (90, -90, "Top-down"), (0, -90, "Side")]
    ):
        ax3d = fig.add_subplot(2, 3, 4 + col, projection="3d")
        ax3d.set_facecolor("#1a1a1a")
        ax3d.set_xlim(*x_range)
        ax3d.set_ylim(*(y_range[::-1] if flip_y else y_range))
        ax3d.set_zlim(*z_range)
        ax3d.set_xlabel("x depth", color="gray", fontsize=7, labelpad=2)
        ax3d.set_ylabel("y lateral", color="gray", fontsize=7, labelpad=2)
        ax3d.set_zlabel("z height", color="gray", fontsize=7, labelpad=2)
        ax3d.tick_params(colors="gray", labelsize=6)
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_title(label, color="white", fontsize=9)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        if pred_xyz is not None:
            _draw_skeleton_3d(ax3d, pred_xyz, color="lime")
        if gt_xyz is not None:
            _draw_skeleton_3d(ax3d, gt_xyz, color="yellow", alpha=0.5, lw=1.2)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="lime", lw=2, label="Prediction"),
        plt.Line2D([0], [0], color="yellow", lw=2, label="GT"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _iter_batches(items, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _normalize_device_name(device_name):
    device_name = str(device_name).strip()
    if device_name.isdigit():
        return f"cuda:{device_name}"
    return device_name


def _requested_devices(args):
    if args.devices:
        devices_arg = args.devices.strip()
        if devices_arg.lower() == "all":
            if torch.cuda.is_available():
                return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
            return ["cpu"]
        return [_normalize_device_name(dev) for dev in devices_arg.split(",") if dev.strip()]
    return [_normalize_device_name(args.device)]


def _resolve_device(requested_device):
    requested_device = _normalize_device_name(requested_device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _render_result(idx, frame_id, arr, pred_kps, gt_pose, out_dir, skeleton_only, flip_y=True):
    err = _mpjpe_cm(pred_kps, gt_pose)
    if skeleton_only:
        fig = make_skeleton_frame(pred_kps, gt_pose, frame_id, err, flip_y=flip_y)
    else:
        fig = make_frame(arr, pred_kps, gt_pose, frame_id, err, flip_y=flip_y)

    # Save with zero-padded index so frames sort correctly for ffmpeg.
    out_path = out_dir / f"{idx:05d}.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return err


def _process_frame_subset(
    worker_id,
    device_name,
    frame_items,
    config_path,
    checkpoint_path,
    out_dir,
    batch_size,
    skeleton_only,
    seq_name,
    render_backend,
    render_width,
    render_height,
    render_elev,
    render_azim,
    render_flip_y,
):
    device = _resolve_device(device_name)
    out_dir = Path(out_dir)
    model, cfg = _load_model(config_path, checkpoint_path, device)
    total = len(frame_items)
    num_batches = int(math.ceil(total / max(1, batch_size))) if total else 0
    print(
        f"[worker {worker_id}] model loaded from {checkpoint_path} "
        f"[device: {device}, frames: {total}, batch_size: {batch_size}]",
        flush=True,
    )

    fast_renderer = None
    if skeleton_only and render_backend == "fast":
        fast_renderer = FastSkeletonFrameRenderer(
            width=render_width,
            height=render_height,
            elev=render_elev,
            azim=render_azim,
            flip_y=render_flip_y,
        )

    mpjpe_all = []
    processed = 0
    try:
        for batch_idx, batch_items in enumerate(_iter_batches(frame_items, batch_size), start=1):
            batch_results = _infer_batch_npy(batch_items, model, cfg, device, seq_name)
            for batch_item, batch_result in zip(batch_items, batch_results):
                gt_pose = batch_item[3]
                idx, frame_id, arr, pred_kps = batch_result
                if fast_renderer is not None:
                    err = _render_result_fast(fast_renderer, idx, frame_id, pred_kps, gt_pose, out_dir)
                else:
                    err = _render_result(idx, frame_id, arr, pred_kps, gt_pose, out_dir, skeleton_only, flip_y=render_flip_y)
                if err is not None:
                    mpjpe_all.append(err)
                processed += 1

            if processed == total or batch_idx == 1 or processed % 50 == 0:
                mean_err = f"{np.mean(mpjpe_all):.1f} cm" if mpjpe_all else "n/a"
                print(
                    f"[worker {worker_id}] [{processed}/{total}] "
                    f"batch {batch_idx}/{num_batches} running_mean={mean_err}",
                    flush=True,
                )
    finally:
        if fast_renderer is not None:
            fast_renderer.close()

    return {"worker_id": worker_id, "device": str(device), "count": total, "mpjpe": mpjpe_all}


def _infer_frame_subset(
    worker_id,
    device_name,
    frame_items,
    config_path,
    checkpoint_path,
    batch_size,
    seq_name,
):
    device = _resolve_device(device_name)
    model, cfg = _load_model(config_path, checkpoint_path, device)
    total = len(frame_items)
    num_batches = int(math.ceil(total / max(1, batch_size))) if total else 0
    print(
        f"[worker {worker_id}] model loaded from {checkpoint_path} "
        f"[device: {device}, frames: {total}, batch_size: {batch_size}]",
        flush=True,
    )

    predictions = []
    processed = 0
    for batch_idx, batch_items in enumerate(_iter_batches(frame_items, batch_size), start=1):
        batch_results = _infer_batch_npy(batch_items, model, cfg, device, seq_name)
        for idx, frame_id, _, pred_kps in batch_results:
            predictions.append((idx, frame_id, pred_kps))
            processed += 1

        if processed == total or batch_idx == 1 or processed % 50 == 0:
            print(
                f"[worker {worker_id}] inference [{processed}/{total}] "
                f"batch {batch_idx}/{num_batches}",
                flush=True,
            )

    return {
        "worker_id": worker_id,
        "device": str(device),
        "count": total,
        "predictions": predictions,
    }


def _collect_predictions(frame_items, devices, args, seq_name):
    if len(devices) > 1:
        shards = _split_round_robin(frame_items, len(devices))
        jobs = [
            (worker_id, device_name, shard)
            for worker_id, (device_name, shard) in enumerate(zip(devices, shards))
            if shard
        ]
        ctx = mp.get_context("spawn")
        completed = 0
        predictions = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=ctx,
        ) as executor:
            futures = [
                executor.submit(
                    _infer_frame_subset,
                    worker_id,
                    device_name,
                    shard,
                    args.config,
                    args.checkpoint,
                    args.batch_size,
                    seq_name,
                )
                for worker_id, device_name, shard in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += result["count"]
                predictions.extend(result["predictions"])
                print(
                    f"[worker {result['worker_id']}] inference complete on "
                    f"{result['device']} ({completed}/{len(frame_items)} total frames)",
                    flush=True,
                )
    else:
        result = _infer_frame_subset(
            0,
            devices[0],
            frame_items,
            args.config,
            args.checkpoint,
            args.batch_size,
            seq_name,
        )
        predictions = result["predictions"]

    predictions.sort(key=lambda item: item[0])
    if len(predictions) != len(frame_items):
        raise RuntimeError(
            f"Collected {len(predictions)} predictions for {len(frame_items)} frames"
        )
    return predictions


def _sosfiltfilt_padlen(sos):
    ntaps = 2 * len(sos) + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * ntaps


def _smooth_keypoint_predictions(predictions, fps, cutoff_hz, order):
    """Apply a zero-phase Butterworth low-pass filter to each keypoint axis."""
    if not predictions:
        return predictions
    if fps <= 0:
        raise ValueError("--fps must be > 0")
    if cutoff_hz <= 0:
        raise ValueError("--filter-cutoff-hz must be > 0")
    if cutoff_hz >= fps / 2.0:
        raise ValueError(
            f"--filter-cutoff-hz must be below Nyquist ({fps / 2.0:.2f} Hz)"
        )
    if order < 1:
        raise ValueError("--filter-order must be >= 1")

    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --smooth-keypoints; install scipy or run without smoothing"
        ) from exc

    num_frames = len(predictions)
    num_keypoints = len(JOINT_NAMES)
    coords = np.full((num_frames, num_keypoints, 3), np.nan, dtype=np.float64)
    scores = np.full((num_frames, num_keypoints), np.nan, dtype=np.float64)
    present = np.zeros((num_frames, num_keypoints), dtype=bool)

    for frame_idx, (_, _, pred_kps) in enumerate(predictions):
        for kp in pred_kps:
            jid = int(kp[0])
            if jid < 0 or jid >= num_keypoints:
                continue
            coords[frame_idx, jid] = [float(kp[1]), float(kp[2]), float(kp[3])]
            scores[frame_idx, jid] = float(kp[4]) if len(kp) > 4 else 1.0
            present[frame_idx, jid] = True

    sos = butter(order, cutoff_hz, btype="lowpass", fs=fps, output="sos")
    padlen = _sosfiltfilt_padlen(sos)
    if num_frames <= padlen:
        print(
            f"Skipping Butterworth smoothing: need more than {padlen} frames, "
            f"got {num_frames}.",
            flush=True,
        )
        return predictions

    time_idx = np.arange(num_frames)
    filtered = coords.copy()
    for jid in range(num_keypoints):
        for axis in range(3):
            values = coords[:, jid, axis]
            valid = np.isfinite(values)
            if valid.sum() < 2:
                continue
            if valid.sum() < num_frames:
                values = np.interp(time_idx, time_idx[valid], values[valid])
            filtered[:, jid, axis] = sosfiltfilt(sos, values)

    smoothed = []
    for frame_idx, (idx, frame_id, pred_kps) in enumerate(predictions):
        smoothed_kps = []
        original_by_id = {int(kp[0]): kp for kp in pred_kps}
        for jid in sorted(original_by_id):
            if jid < 0 or jid >= num_keypoints or not present[frame_idx, jid]:
                continue
            score = scores[frame_idx, jid]
            if not np.isfinite(score):
                score = original_by_id[jid][4] if len(original_by_id[jid]) > 4 else 1.0
            x, y, z = filtered[frame_idx, jid]
            smoothed_kps.append((jid, float(x), float(y), float(z), float(score)))
        smoothed.append((idx, frame_id, smoothed_kps))

    print(
        f"Applied zero-phase Butterworth smoothing "
        f"[fps={fps:g}, cutoff={cutoff_hz:g} Hz, order={order}]",
        flush=True,
    )
    return smoothed


def _render_prediction_sequence(frame_items, predictions, out_dir, skeleton_only, config_path, args):
    pred_by_idx = {idx: pred_kps for idx, _, pred_kps in predictions}
    cfg = Config.fromfile(config_path)
    _set_visualization_spec_from_config(cfg)
    fast_renderer = None
    if skeleton_only and args.render_backend == "fast":
        fast_renderer = FastSkeletonFrameRenderer(
            width=args.render_width,
            height=args.render_height,
            elev=args.render_elev,
            azim=args.render_azim,
            flip_y=not args.no_flip_y,
        )

    mpjpe_all = []
    try:
        for idx, frame_id, npy_path, gt_pose in frame_items:
            pred_kps = pred_by_idx.get(idx, [])
            if fast_renderer is not None:
                err = _render_result_fast(fast_renderer, idx, frame_id, pred_kps, gt_pose, out_dir)
            else:
                arr = None if skeleton_only else _preprocess_npy(npy_path, cfg)
                err = _render_result(idx, frame_id, arr, pred_kps, gt_pose, out_dir, skeleton_only, flip_y=not args.no_flip_y)
            if err is not None:
                mpjpe_all.append(err)

            if (idx + 1) % 50 == 0 or idx == 0 or idx == len(frame_items) - 1:
                mean_err = f"{np.mean(mpjpe_all):.1f} cm" if mpjpe_all else "n/a"
                if err is not None:
                    print(
                        f"  [{idx+1}/{len(frame_items)}] frame {frame_id} "
                        f"joints={len(pred_kps)} err={err:.1f} cm "
                        f"running_mean={mean_err}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{idx+1}/{len(frame_items)}] frame {frame_id} "
                        f"joints={len(pred_kps)}",
                        flush=True,
                    )
    finally:
        if fast_renderer is not None:
            fast_renderer.close()

    return mpjpe_all


def _keypoints_for_json(keypoints):
    return [
        [int(kp[0]), float(kp[1]), float(kp[2]), float(kp[3]), float(kp[4])]
        if len(kp) > 4 else
        [int(kp[0]), float(kp[1]), float(kp[2]), float(kp[3])]
        for kp in keypoints
    ]


def _pose_for_json(pose):
    if pose is None:
        return None
    return [[float(x), float(y), float(z)] for x, y, z in pose]


def _prediction_export_path(args, out_dir):
    return Path(args.predictions_file) if args.predictions_file else out_dir / "predictions.json"


def _save_prediction_export(
    export_path,
    frame_items,
    raw_predictions,
    smoothed_predictions,
    args,
    seq_name,
):
    raw_by_idx = {idx: pred_kps for idx, _, pred_kps in raw_predictions}
    smooth_by_idx = (
        {idx: pred_kps for idx, _, pred_kps in smoothed_predictions}
        if smoothed_predictions is not None else None
    )

    frames = []
    for idx, frame_id, npy_path, gt_pose in frame_items:
        record = {
            "index": int(idx),
            "frame_id": frame_id,
            "npy_path": npy_path,
            "pred_keypoints": _keypoints_for_json(raw_by_idx.get(idx, [])),
            "gt_pose": _pose_for_json(gt_pose),
        }
        if smooth_by_idx is not None:
            record["smoothed_keypoints"] = _keypoints_for_json(smooth_by_idx.get(idx, []))
        frames.append(record)

    payload = {
        "format_version": 1,
        "sequence": seq_name,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "frame_count": len(frames),
        "keypoint_format": KEYPOINT_FORMAT,
        "keypoint_names": JOINT_NAMES,
        "coordinate_frame": COORDINATE_FRAME,
        "prediction_columns": ["joint_id", "x_depth", "y_lateral", "z_height", "score"],
        "gt_pose_columns": ["x_depth", "y_lateral", "z_height"],
        "smoothing": {
            "enabled": bool(smoothed_predictions is not None),
            "fps": float(args.fps),
            "cutoff_hz": float(args.filter_cutoff_hz),
            "order": int(args.filter_order),
            "method": "scipy.signal.butter(output='sos') + scipy.signal.sosfiltfilt",
        },
        "frames": frames,
    }

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved predictions + GT to {export_path}", flush=True)


def _split_round_robin(items, num_shards):
    shards = [[] for _ in range(num_shards)]
    for pos, item in enumerate(items):
        shards[pos % num_shards].append(item)
    return shards


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RT-Pose model predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir", help="Session folder containing radar npy frames and Train.json")
    group.add_argument("--npy-file", help="Single radar npy frame")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",
        default=str(RT_POSE_ROOT / "configs/custom_fitness_body18/hr3d_one_hm_18j_dzyx_leaveout.py"))
    parser.add_argument("--out-dir", default=None,
        help="Output directory for PNGs. Default: <session_dir>/visualizations/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default=None,
        help="Comma-separated devices for concurrent session processing, e.g. cuda:0,cuda:1,cuda:2,cuda:3. Use 'all' for all visible CUDA GPUs.")
    parser.add_argument("--batch-size", type=int, default=8,
        help="Number of frames per model forward on each device (default: 8).")
    parser.add_argument("--smooth-keypoints", action="store_true",
        help="Apply zero-phase Butterworth smoothing to predicted X/Y/Z keypoint coordinates before rendering")
    parser.add_argument("--fps", type=float, default=30.0,
        help="Frame rate used by --smooth-keypoints (default: 30)")
    parser.add_argument("--filter-cutoff-hz", type=float, default=7.0,
        help="Low-pass cutoff frequency in Hz for --smooth-keypoints (default: 7)")
    parser.add_argument("--filter-order", type=int, default=4,
        help="Butterworth filter order for --smooth-keypoints (default: 4)")
    parser.add_argument("--save-predictions", action="store_true",
        help="Save raw predictions and GT as JSON for offline post-processing")
    parser.add_argument("--predictions-file", default=None,
        help="Prediction JSON path. Default with --save-predictions: <out_dir>/predictions.json")
    parser.add_argument("--no-gt", action="store_true", help="Skip ground-truth overlay")
    parser.add_argument("--max-frames", type=int, default=None,
        help="Limit number of frames to process (useful for quick checks)")
    parser.add_argument("--label-file", default="Train.json",
        help="Label filename inside session dir (default: Train.json)")
    parser.add_argument("--skeleton-only", action="store_true",
        help="Save a compact skeleton-only image instead of the full radar figure")
    parser.add_argument("--render-backend", choices=("fast", "legacy"), default="fast",
        help="For --skeleton-only: 'fast' reuses one Matplotlib canvas and saves with OpenCV; 'legacy' uses the old savefig path")
    parser.add_argument("--render-width", type=int, default=1000,
        help="Fast skeleton renderer output PNG width in pixels (default: 1000)")
    parser.add_argument("--render-height", type=int, default=1000,
        help="Fast skeleton renderer output PNG height in pixels (default: 1000)")
    parser.add_argument("--render-elev", type=float, default=25.0,
        help="Fast skeleton renderer 3D elevation angle (default: 25)")
    parser.add_argument("--render-azim", type=float, default=210.0,
        help="Fast skeleton renderer 3D azimuth; 180 views from low x toward high x")
    parser.add_argument("--no-flip-y", action="store_true",
        help="Keep the original y-axis direction in 3D visualizations instead of flipping left/right")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.render_width < 64 or args.render_height < 64:
        raise ValueError("--render-width and --render-height must be >= 64")

    display_cfg = Config.fromfile(args.config)
    _set_visualization_spec_from_config(display_cfg)
    radar_npy_dir_name = _radar_npy_dir_from_config(display_cfg)

    # --- Collect frames ---
    frames = []   # list of (frame_id, npy_path, gt_pose_or_None)

    if args.npy_file:
        npy_path = Path(args.npy_file)
        frame_id = npy_path.stem
        frames.append((frame_id, str(npy_path), None))
        seq_name = npy_path.parent.parent.name
        out_dir = Path(args.out_dir) if args.out_dir else npy_path.parent.parent / "visualizations"

    else:
        session_dir = Path(args.session_dir)
        npy_dir = session_dir / radar_npy_dir_name
        seq_name = session_dir.name
        out_dir = Path(args.out_dir) if args.out_dir else session_dir / "visualizations"

        # Load GT if available
        gt_lookup = {}
        if not args.no_gt:
            label_path = session_dir / args.label_file
            if label_path.exists():
                with open(label_path) as f:
                    gt_data = json.load(f)
                # GT structure: {session: {frame_id: [{pose: [...]}]}}
                for seq_key, seq_frames in gt_data.items():
                    for fid, objs in seq_frames.items():
                        pose = objs[0].get("pose", []) if objs else []
                        gt_lookup[fid] = pose if _gt_pose_to_xyz(pose) is not None else None
                print(f"Loaded GT from {label_path}  ({len(gt_lookup)} frames)")
            else:
                print(f"No {args.label_file} found in {session_dir}; skipping GT")

        npy_files = sorted(npy_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No npy files found in {npy_dir}")

        for npy_path in npy_files:
            fid = npy_path.stem
            gt_pose = gt_lookup.get(fid) if not args.no_gt else None
            frames.append((fid, str(npy_path), gt_pose))

    if args.max_frames:
        frames = frames[: args.max_frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_items = [
        (idx, frame_id, npy_path, gt_pose)
        for idx, (frame_id, npy_path, gt_pose) in enumerate(frames)
    ]

    devices = _requested_devices(args)
    if not devices:
        devices = [_normalize_device_name(args.device)]
    if args.npy_file and len(devices) > 1:
        print(f"Single-frame mode uses one device; using {devices[0]}.")
        devices = devices[:1]
    if len(frame_items) < len(devices):
        devices = devices[:len(frame_items)]

    device_label = ", ".join(devices)
    render_label = args.render_backend if args.skeleton_only else "legacy-full"
    print(
        f"Processing {len(frames)} frames -> {out_dir} "
        f"[devices: {device_label}, batch_size: {args.batch_size}, render={render_label}]"
    )

    mpjpe_all = []
    should_save_predictions = args.save_predictions or args.predictions_file is not None
    if args.smooth_keypoints or should_save_predictions:
        raw_predictions = _collect_predictions(frame_items, devices, args, seq_name)
        render_predictions = raw_predictions
        smoothed_predictions = None

        if args.smooth_keypoints:
            smoothed_predictions = _smooth_keypoint_predictions(
                raw_predictions,
                fps=args.fps,
                cutoff_hz=args.filter_cutoff_hz,
                order=args.filter_order,
            )
            render_predictions = smoothed_predictions

        if should_save_predictions:
            _save_prediction_export(
                _prediction_export_path(args, out_dir),
                frame_items,
                raw_predictions,
                smoothed_predictions,
                args,
                seq_name,
            )

        render_label = "smoothed" if args.smooth_keypoints else "raw"
        print(
            f"Rendering {len(render_predictions)} {render_label} frames -> {out_dir}",
            flush=True,
        )
        mpjpe_all = _render_prediction_sequence(
            frame_items,
            render_predictions,
            out_dir,
            args.skeleton_only,
            args.config,
            args,
        )
    elif len(devices) > 1:
        shards = _split_round_robin(frame_items, len(devices))
        jobs = [
            (worker_id, device_name, shard)
            for worker_id, (device_name, shard) in enumerate(zip(devices, shards))
            if shard
        ]
        ctx = mp.get_context("spawn")
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=ctx,
        ) as executor:
            futures = [
                executor.submit(
                    _process_frame_subset,
                    worker_id,
                    device_name,
                    shard,
                    args.config,
                    args.checkpoint,
                    str(out_dir),
                    args.batch_size,
                    args.skeleton_only,
                    seq_name,
                    args.render_backend,
                    args.render_width,
                    args.render_height,
                    args.render_elev,
                    args.render_azim,
                    not args.no_flip_y,
                )
                for worker_id, device_name, shard in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += result["count"]
                mpjpe_all.extend(result["mpjpe"])
                print(
                    f"[worker {result['worker_id']}] complete on {result['device']} "
                    f"({completed}/{len(frame_items)} total frames)",
                    flush=True,
                )
    else:
        result = _process_frame_subset(
            0,
            devices[0],
            frame_items,
            args.config,
            args.checkpoint,
            str(out_dir),
            args.batch_size,
            args.skeleton_only,
            seq_name,
            args.render_backend,
            args.render_width,
            args.render_height,
            args.render_elev,
            args.render_azim,
            not args.no_flip_y,
        )
        mpjpe_all.extend(result["mpjpe"])

    print(f"\nDone. {len(frames)} frames saved to {out_dir}")
    if mpjpe_all:
        print(f"Mean Abs-MPJPE over {len(mpjpe_all)} frames: {np.mean(mpjpe_all):.2f} cm")
        print(f"  Per-sequence best: {np.min(mpjpe_all):.2f} cm   worst: {np.max(mpjpe_all):.2f} cm")

    ffmpeg_fps = f"{args.fps:g}"
    input_pattern = out_dir / "%05d.png"
    output_video = out_dir / "output.mp4"
    print("\nTo convert PNG frames to an H.264 video:")
    print(
        f'  ffmpeg -framerate {ffmpeg_fps} -i "{input_pattern}" '
        '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'-c:v libx264 -pix_fmt yuv420p "{output_video}"'
    )


if __name__ == "__main__":
    main()
