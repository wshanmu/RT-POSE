#!/usr/bin/env python
"""Render saved RT-POSE prediction JSON as BODY_18 3D comparison frames/video.

Input is the JSON produced by tools/visualize_session_3D.py with
--save-predictions. The renderer draws prediction as solid lines and GT as
dashed lines, with left and right body parts colored separately.

Example:
    python tools/visualize_prediction_json_3d.py \
        work_dirs/hr3d_one_hm_18j_dzyx_leaveout/predictions_epoch20_boelter_loc1_session3.json \
        --out-dir /tmp/body18_epoch20_viz \
        --smooth \
        --fps 30
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D


BODY18_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]

BODY18_EDGES = [
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17),
]

LEFT_JOINTS = {5, 6, 7, 11, 12, 13, 15, 17}
RIGHT_JOINTS = {2, 3, 4, 8, 9, 10, 14, 16}
LEFT_COLOR = "#2979FF"
RIGHT_COLOR = "#FF3D00"
CENTER_COLOR = "#37474F"
GT_ALPHA = 0.38
PRED_ALPHA = 0.95


def edge_color(start: int, end: int) -> str:
    left = start in LEFT_JOINTS or end in LEFT_JOINTS
    right = start in RIGHT_JOINTS or end in RIGHT_JOINTS
    if left and not right:
        return LEFT_COLOR
    if right and not left:
        return RIGHT_COLOR
    return CENTER_COLOR


def joint_colors(num_keypoints: int) -> list[str]:
    colors = []
    for jid in range(num_keypoints):
        if jid in LEFT_JOINTS:
            colors.append(LEFT_COLOR)
        elif jid in RIGHT_JOINTS:
            colors.append(RIGHT_COLOR)
        else:
            colors.append(CENTER_COLOR)
    return colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render saved RT-POSE BODY_18 predictions and GT into a 30 fps 3D video."
    )
    parser.add_argument("predictions_json", type=Path, help="JSON exported by visualize_session_3D.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for frames, video, and smoothed JSON")
    parser.add_argument("--video-name", default="prediction_gt_3d.mp4", help="Output video filename inside --out-dir")
    parser.add_argument("--frames-dir", default="frames", help="Frame PNG folder inside --out-dir")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video fps")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start", type=int, default=0, help="First frame index in JSON")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end frame index in JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit rendered frame count after --start/--end")
    parser.add_argument("--elev", type=float, default=15.0, help="Matplotlib 3D elevation")
    parser.add_argument("--azim", type=float, default=210.0, help="Matplotlib 3D azimuth; 180 views from low x toward high x")
    parser.add_argument("--prediction-key", default=None, help="Prediction key to render. Default: smoothed_keypoints if --smooth else pred_keypoints")
    parser.add_argument("--gt-key", default="gt_pose", help="GT pose key to render")
    parser.add_argument("--score-threshold", type=float, default=-1.0, help="Hide predicted joints below this score")
    parser.add_argument("--axis-padding", type=float, default=0.18, help="Extra metres around auto axis limits")
    parser.add_argument("--fixed-limits", action="store_true", help="Use radar-style fixed XYZ limits instead of auto limits")
    parser.add_argument("--no-flip-y", action="store_true", help="Keep the original y-axis direction instead of flipping left/right in the 3D view")
    parser.add_argument("--no-save-frames", action="store_true", help="Write video only, without keeping PNG frames")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing frame folder/video/smoothed JSON")

    parser.add_argument("--smooth", action="store_true", help="Write smoothed JSON and render smoothed predictions by default")
    parser.add_argument("--smooth-target", choices=("pred", "gt", "both"), default="pred")
    parser.add_argument("--smooth-output", type=Path, default=None, help="Smoothed JSON path. Default: <out-dir>/<input_stem>_smoothed.json")
    parser.add_argument("--sigma-trans", type=float, default=0.10, help="HMM transition sigma in metres")
    parser.add_argument("--sigma-emit", type=float, default=0.10, help="HMM emission sigma in metres")
    parser.add_argument("--min-smooth-score", type=float, default=1e-3, help="Minimum confidence used by the HMM smoother")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise ValueError(f"Expected prediction JSON with a top-level 'frames' list: {path}")
    return payload


def infer_num_keypoints(payload: dict) -> int:
    names = payload.get("keypoint_names")
    if isinstance(names, list) and names:
        return len(names)
    max_id = -1
    for frame in payload.get("frames", []):
        for key in ("pred_keypoints", "smoothed_keypoints"):
            for kp in frame.get(key, []) or []:
                if kp:
                    max_id = max(max_id, int(kp[0]))
        pose = frame.get("gt_pose")
        if pose:
            max_id = max(max_id, len(pose) - 1)
    return max(max_id + 1, len(BODY18_NAMES))


def pred_keypoints_to_array(keypoints: list, num_keypoints: int, score_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.full((num_keypoints, 3), np.nan, dtype=np.float64)
    scores = np.zeros(num_keypoints, dtype=np.float64)
    for kp in keypoints or []:
        if len(kp) < 4:
            continue
        jid = int(kp[0])
        if jid < 0 or jid >= num_keypoints:
            continue
        score = float(kp[4]) if len(kp) > 4 else 1.0
        if score < score_threshold:
            continue
        xyz[jid] = [float(kp[1]), float(kp[2]), float(kp[3])]
        scores[jid] = score
    return xyz, scores


def gt_pose_to_array(pose: list | None, num_keypoints: int) -> np.ndarray:
    xyz = np.full((num_keypoints, 3), np.nan, dtype=np.float64)
    if not pose:
        return xyz
    try:
        raw = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return xyz
    if raw.ndim != 2 or raw.shape[1] < 3:
        return xyz
    count = min(num_keypoints, raw.shape[0])
    xyz[:count] = raw[:count, :3]
    return xyz


def arrays_from_frames(
    frames: list[dict],
    num_keypoints: int,
    pred_key: str,
    gt_key: str,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.full((len(frames), num_keypoints, 3), np.nan, dtype=np.float64)
    pred_scores = np.zeros((len(frames), num_keypoints), dtype=np.float64)
    gt = np.full((len(frames), num_keypoints, 3), np.nan, dtype=np.float64)

    for idx, frame in enumerate(frames):
        pred[idx], pred_scores[idx] = pred_keypoints_to_array(
            frame.get(pred_key, []), num_keypoints, score_threshold
        )
        gt[idx] = gt_pose_to_array(frame.get(gt_key), num_keypoints)
    return pred, pred_scores, gt


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Fill missing values for smoothing; returns a copy."""
    output = np.asarray(values, dtype=np.float64).copy()
    frame_idx = np.arange(output.shape[0])
    for joint in range(output.shape[1]):
        for axis in range(3):
            track = output[:, joint, axis]
            valid = np.isfinite(track)
            if not np.any(valid):
                output[:, joint, axis] = 0.0
            elif valid.sum() == 1:
                output[:, joint, axis] = track[valid][0]
            elif valid.sum() < len(track):
                output[:, joint, axis] = np.interp(frame_idx, frame_idx[valid], track[valid])
    return output


def hmm_smooth_xyz(
    xyz: np.ndarray,
    scores: np.ndarray | None,
    sigma_trans: float,
    sigma_emit: float,
    min_score: float,
) -> np.ndarray:
    """Confidence-weighted forward/backward HMM smoothing.

    This mirrors the skeleton-preprocessing smoothing idea, but skips temporal
    person association because RT-POSE JSON already has one ordered BODY_18 pose
    per frame.
    """
    if xyz.shape[0] < 2:
        return xyz.copy()
    obs = interpolate_missing(xyz)
    valid = np.isfinite(xyz).all(axis=2)
    if scores is None:
        confidence = valid.astype(np.float64)
    else:
        confidence = np.asarray(scores, dtype=np.float64).copy()
        confidence[~valid] = 0.0
    confidence = np.clip(confidence, min_score, None)

    var_trans = float(sigma_trans) ** 2
    smoothed = np.full_like(obs, np.nan, dtype=np.float64)

    for joint in range(obs.shape[1]):
        emit_var = (float(sigma_emit) / confidence[:, joint]) ** 2

        f_mean = np.zeros((obs.shape[0], 3), dtype=np.float64)
        f_var = np.zeros(obs.shape[0], dtype=np.float64)
        f_mean[0] = obs[0, joint]
        f_var[0] = emit_var[0] + var_trans
        for t in range(1, obs.shape[0]):
            pred_mean = f_mean[t - 1]
            pred_var = f_var[t - 1] + var_trans
            cur_var = emit_var[t]
            denom = pred_var + cur_var
            f_mean[t] = (obs[t, joint] * pred_var + pred_mean * cur_var) / denom
            f_var[t] = (pred_var * cur_var) / denom

        b_mean = np.zeros((obs.shape[0], 3), dtype=np.float64)
        b_var = np.zeros(obs.shape[0], dtype=np.float64)
        b_mean[-1] = obs[-1, joint]
        b_var[-1] = emit_var[-1] + var_trans
        for t in range(obs.shape[0] - 2, -1, -1):
            pred_mean = b_mean[t + 1]
            pred_var = b_var[t + 1] + var_trans
            cur_var = emit_var[t]
            denom = pred_var + cur_var
            b_mean[t] = (obs[t, joint] * pred_var + pred_mean * cur_var) / denom
            b_var[t] = (pred_var * cur_var) / denom

        inv_f = 1.0 / np.maximum(f_var, 1e-12)
        inv_b = 1.0 / np.maximum(b_var, 1e-12)
        smoothed[:, joint] = (
            f_mean * inv_f[:, None] + b_mean * inv_b[:, None]
        ) / (inv_f + inv_b)[:, None]

    smoothed[~valid] = np.nan
    return smoothed


def array_to_pred_keypoints(xyz: np.ndarray, scores: np.ndarray) -> list[list[float]]:
    keypoints = []
    for jid, point in enumerate(xyz):
        if not np.isfinite(point).all():
            continue
        keypoints.append([
            int(jid),
            float(point[0]),
            float(point[1]),
            float(point[2]),
            float(scores[jid]) if jid < len(scores) else 1.0,
        ])
    return keypoints


def array_to_gt_pose(xyz: np.ndarray) -> list[list[float]]:
    pose = []
    for point in xyz:
        if np.isfinite(point).all():
            pose.append([float(point[0]), float(point[1]), float(point[2])])
        else:
            pose.append([None, None, None])
    return pose


def add_smoothed_tracks(
    payload: dict,
    num_keypoints: int,
    args: argparse.Namespace,
) -> dict:
    output = copy.deepcopy(payload)
    frames = output["frames"]
    pred, scores, gt = arrays_from_frames(
        frames,
        num_keypoints=num_keypoints,
        pred_key="pred_keypoints",
        gt_key=args.gt_key,
        score_threshold=-np.inf,
    )

    if args.smooth_target in ("pred", "both"):
        pred_smoothed = hmm_smooth_xyz(
            pred,
            scores=scores,
            sigma_trans=args.sigma_trans,
            sigma_emit=args.sigma_emit,
            min_score=args.min_smooth_score,
        )
        for idx, frame in enumerate(frames):
            frame["smoothed_keypoints"] = array_to_pred_keypoints(pred_smoothed[idx], scores[idx])

    if args.smooth_target in ("gt", "both"):
        gt_scores = np.isfinite(gt).all(axis=2).astype(np.float64)
        gt_smoothed = hmm_smooth_xyz(
            gt,
            scores=gt_scores,
            sigma_trans=args.sigma_trans,
            sigma_emit=args.sigma_emit,
            min_score=args.min_smooth_score,
        )
        for idx, frame in enumerate(frames):
            frame["smoothed_gt_pose"] = array_to_gt_pose(gt_smoothed[idx])

    output["smoothing"] = {
        "enabled": True,
        "method": "confidence_weighted_hmm_forward_backward",
        "source": "skeleton-preprocessing/preprocessing/smooth_skeleton_3d.py",
        "target": args.smooth_target,
        "sigma_trans_m": float(args.sigma_trans),
        "sigma_emit_m": float(args.sigma_emit),
        "min_score": float(args.min_smooth_score),
    }
    return output


def save_smoothed_payload(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def compute_limits(pred: np.ndarray, gt: np.ndarray, padding: float, fixed: bool) -> tuple[tuple[float, float], ...]:
    if fixed:
        return ((0.9, 3.5), (-1.25, 1.25), (-1.15, 1.40))
    points = np.concatenate([
        pred.reshape(-1, 3),
        gt.reshape(-1, 3),
    ], axis=0)
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return ((0.9, 3.5), (-1.25, 1.25), (-1.15, 1.40))
    points = points[valid]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins)) / 2.0 + padding, 0.5)
    return tuple((float(c - half), float(c + half)) for c in center)


def mean_mpjpe_cm(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float | None:
    valid = np.isfinite(pred_xyz).all(axis=1) & np.isfinite(gt_xyz).all(axis=1)
    if not np.any(valid):
        return None
    return float(np.linalg.norm(pred_xyz[valid] - gt_xyz[valid], axis=1).mean() * 100.0)


class FastMatplotlibRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        limits: tuple[tuple[float, float], ...],
        elev: float,
        azim: float,
        num_keypoints: int,
        flip_y: bool = True,
    ) -> None:
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("white")
        self.ax.set_xlim(*limits[0])
        self.ax.set_ylim(*(limits[1][::-1] if flip_y else limits[1]))
        self.ax.set_zlim(*limits[2])
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("x depth (m)")
        self.ax.set_ylabel("y lateral (m)")
        self.ax.set_zlabel("z height (m)")
        self.ax.grid(True, color="#e5e7eb", linewidth=0.5)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#d1d5db")

        self.title = self.ax.set_title("", fontsize=11, pad=16)
        self.pred_lines = []
        self.gt_lines = []
        for start, end in BODY18_EDGES:
            color = edge_color(start, end)
            pred_line, = self.ax.plot([], [], [], color=color, lw=2.8, alpha=PRED_ALPHA, linestyle="-")
            gt_line, = self.ax.plot([], [], [], color=color, lw=2.0, alpha=GT_ALPHA, linestyle="--")
            self.pred_lines.append((start, end, pred_line))
            self.gt_lines.append((start, end, gt_line))

        colors = joint_colors(num_keypoints)
        empty = np.full(num_keypoints, np.nan)
        self.pred_scatter = self.ax.scatter(
            empty,
            empty,
            empty,
            c=colors,
            s=70,
            depthshade=False,
            alpha=PRED_ALPHA,
            edgecolors="white",
            linewidths=0.8,
        )
        self.gt_scatter = self.ax.scatter(
            empty,
            empty,
            empty,
            c=colors,
            s=38,
            depthshade=False,
            alpha=0.55,
            edgecolors="#111827",
            linewidths=0.35,
        )
        self.legend = self.fig.legend(
            handles=[
                Line2D([0], [0], color=LEFT_COLOR, lw=3, label="Left"),
                Line2D([0], [0], color=RIGHT_COLOR, lw=3, label="Right"),
                Line2D([0], [0], color=CENTER_COLOR, lw=3, label="Midline"),
                Line2D([0], [0], color="#111827", lw=2.8, label="Prediction"),
                Line2D([0], [0], color="#111827", lw=2.0, linestyle="--", label="GT"),
            ],
            loc="lower center",
            ncol=5,
            framealpha=0.9,
            fontsize=9,
        )
        self.fig.subplots_adjust(left=0.03, right=0.98, bottom=0.13, top=0.88)

    @staticmethod
    def _set_line(line, xyz: np.ndarray, start: int, end: int) -> None:
        if (
            start < xyz.shape[0]
            and end < xyz.shape[0]
            and np.isfinite(xyz[start]).all()
            and np.isfinite(xyz[end]).all()
        ):
            segment = xyz[[start, end]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        else:
            line.set_data([], [])
            line.set_3d_properties([])

    @staticmethod
    def _set_scatter(scatter, xyz: np.ndarray) -> None:
        scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def render(self, frame_id: str, pred_xyz: np.ndarray, gt_xyz: np.ndarray, frame_index: int, frame_count: int) -> np.ndarray:
        for start, end, line in self.pred_lines:
            self._set_line(line, pred_xyz, start, end)
        for start, end, line in self.gt_lines:
            self._set_line(line, gt_xyz, start, end)
        self._set_scatter(self.pred_scatter, pred_xyz)
        self._set_scatter(self.gt_scatter, gt_xyz)

        err = mean_mpjpe_cm(pred_xyz, gt_xyz)
        title = f"Frame {frame_id}  ({frame_index + 1}/{frame_count})"
        if err is not None:
            title += f"  |  MPJPE {err:.1f} cm"
        self.title.set_text(title)

        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        plt.close(self.fig)


def render_video_and_frames(
    frames: list[dict],
    pred: np.ndarray,
    gt: np.ndarray,
    args: argparse.Namespace,
    video_path: Path,
    frames_dir: Path,
) -> None:
    limits = compute_limits(pred, gt, padding=args.axis_padding, fixed=args.fixed_limits)
    renderer = FastMatplotlibRenderer(
        width=args.width,
        height=args.height,
        limits=limits,
        elev=args.elev,
        azim=args.azim,
        num_keypoints=pred.shape[1],
        flip_y=not args.no_flip_y,
    )

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not open video writer for {video_path}")

    try:
        total = len(frames)
        for idx, frame in enumerate(frames):
            image = renderer.render(str(frame.get("frame_id", idx)), pred[idx], gt[idx], idx, total)
            writer.write(image)
            if not args.no_save_frames:
                frame_path = frames_dir / f"frame_{idx:06d}.png"
                if not cv2.imwrite(str(frame_path), image):
                    raise RuntimeError(f"Could not write frame: {frame_path}")
            if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"Rendered {idx + 1}/{total} frames", flush=True)
    finally:
        writer.release()
        renderer.close()


def selected_frames(payload: dict, args: argparse.Namespace) -> list[dict]:
    frames = payload["frames"]
    end = len(frames) if args.end is None else min(len(frames), args.end)
    if args.start < 0 or args.start >= end:
        raise ValueError(f"Invalid frame range: start={args.start}, end={end}")
    frames = frames[args.start:end]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    return frames


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    payload = load_payload(args.predictions_json)
    num_keypoints = infer_num_keypoints(payload)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.out_dir / args.video_name
    frames_dir = args.out_dir / args.frames_dir
    if args.overwrite:
        if video_path.exists():
            video_path.unlink()
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    if video_path.exists():
        raise FileExistsError(f"Video already exists: {video_path}. Use --overwrite to replace it.")
    if not args.no_save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    if args.smooth:
        smooth_output = args.smooth_output or (args.out_dir / f"{args.predictions_json.stem}_smoothed.json")
        if smooth_output.exists() and not args.overwrite:
            raise FileExistsError(f"Smoothed JSON already exists: {smooth_output}. Use --overwrite to replace it.")
        payload = add_smoothed_tracks(payload, num_keypoints=num_keypoints, args=args)
        save_smoothed_payload(payload, smooth_output)
        print(f"Saved smoothed JSON to {smooth_output}", flush=True)

    pred_key = args.prediction_key
    if pred_key is None:
        pred_key = "smoothed_keypoints" if args.smooth and args.smooth_target in ("pred", "both") else "pred_keypoints"
    gt_key = "smoothed_gt_pose" if args.smooth and args.smooth_target in ("gt", "both") else args.gt_key

    frames = selected_frames(payload, args)
    pred, _scores, gt = arrays_from_frames(
        frames,
        num_keypoints=num_keypoints,
        pred_key=pred_key,
        gt_key=gt_key,
        score_threshold=args.score_threshold,
    )
    render_video_and_frames(frames, pred, gt, args=args, video_path=video_path, frames_dir=frames_dir)
    print(f"Saved video to {video_path}", flush=True)
    if not args.no_save_frames:
        print(f"Saved frames to {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
