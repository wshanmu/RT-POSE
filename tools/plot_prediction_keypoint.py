"""
Plot per-frame keypoint error, velocity, and acceleration from prediction JSON.

Example:
    python tools/plot_prediction_keypoint.py \
        --pred-json ./vis_session6_epoch40/predictions.json \
        --keypoint LAnkle \
        --prediction-key pred_keypoints \
        --out ./lankle_error_velocity.png

Example with GT-tuned velocity/acceleration gates:
    python tools/plot_prediction_keypoint.py \
        --pred-json ./vis_session6_epoch40/predictions.json \
        --keypoint LAnkle \
        --coord-mode 2d \
        --optimize-gates \
        --gated-output-json ./vis_session6_epoch40/predictions_gated.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one keypoint's error, velocity, and acceleration over frames."
    )
    parser.add_argument("--pred-json", required=True, help="Path to predictions.json")
    parser.add_argument(
        "--keypoint",
        required=True,
        help="Keypoint id or name, e.g. 15 or LAnkle",
    )
    parser.add_argument(
        "--prediction-key",
        default="pred_keypoints",
        help="Which prediction field to plot (default: pred_keypoints)",
    )
    parser.add_argument(
        "--coord-mode",
        choices=("3d", "2d"),
        default="3d",
        help="Use full x/y/z coordinates, or 2D y/z only by ignoring x/depth (default: 3d)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate for velocity. Default: JSON smoothing.fps, or 30.",
    )
    parser.add_argument(
        "--unit",
        choices=("cm", "m"),
        default="cm",
        help="Position unit for plotting error, speed, and acceleration (default: cm)",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--optimize-gates",
        action="store_true",
        help="Tune per-joint velocity/acceleration gates against GT and overlay gated curves",
    )
    parser.add_argument(
        "--gated-output-json",
        default=None,
        help="Output JSON with gated_keypoints. Default: <pred-json stem>_gated.json",
    )
    parser.add_argument(
        "--gated-prediction-key",
        default="gated_keypoints",
        help="Field name to write for gated predictions (default: gated_keypoints)",
    )
    parser.add_argument(
        "--gate-grid-size",
        type=int,
        default=18,
        help="Number of velocity/acceleration candidate quantiles per joint (default: 18)",
    )
    parser.add_argument(
        "--gate-min-quantile",
        type=float,
        default=0.50,
        help="Lowest motion quantile considered during gate search (default: 0.50)",
    )
    parser.add_argument(
        "--gate-max-quantile",
        type=float,
        default=0.995,
        help="Highest motion quantile considered during gate search (default: 0.995)",
    )
    parser.add_argument(
        "--min-kept-ratio",
        type=float,
        default=0.60,
        help="Reject threshold candidates that keep less than this ratio of valid frames (default: 0.60)",
    )
    return parser.parse_args()


def _norm_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_keypoint_id(keypoint, keypoint_names):
    try:
        keypoint_id = int(keypoint)
    except ValueError:
        target = _norm_name(keypoint)
        lookup = {_norm_name(name): idx for idx, name in enumerate(keypoint_names)}
        if target not in lookup:
            valid = ", ".join(f"{idx}:{name}" for idx, name in enumerate(keypoint_names))
            raise ValueError(f"Unknown keypoint {keypoint!r}. Valid keypoints: {valid}")
        keypoint_id = lookup[target]

    if keypoint_id < 0 or keypoint_id >= len(keypoint_names):
        raise ValueError(
            f"Keypoint id {keypoint_id} is out of range [0, {len(keypoint_names) - 1}]"
        )
    return keypoint_id


def _prediction_xyz(frame, prediction_key, keypoint_id):
    keypoints = frame.get(prediction_key)
    if keypoints is None:
        return None
    for kp in keypoints:
        if int(kp[0]) == keypoint_id:
            return np.array(kp[1:4], dtype=np.float64)
    return None


def _gt_xyz(frame, keypoint_id):
    pose = frame.get("gt_pose")
    if pose is None or keypoint_id >= len(pose):
        return None
    return np.array(pose[keypoint_id], dtype=np.float64)


def _extract_series(payload, keypoint_id, prediction_key):
    frames = payload.get("frames", [])
    frame_indices = np.array(
        [int(frame.get("index", idx)) for idx, frame in enumerate(frames)],
        dtype=np.int64,
    )
    pred_xyz = np.full((len(frames), 3), np.nan, dtype=np.float64)
    gt_xyz = np.full((len(frames), 3), np.nan, dtype=np.float64)

    saw_prediction_key = False
    for idx, frame in enumerate(frames):
        if prediction_key in frame:
            saw_prediction_key = True
        pred = _prediction_xyz(frame, prediction_key, keypoint_id)
        gt = _gt_xyz(frame, keypoint_id)
        if pred is not None:
            pred_xyz[idx] = pred
        if gt is not None:
            gt_xyz[idx] = gt

    if not saw_prediction_key:
        raise ValueError(
            f"Prediction key {prediction_key!r} was not found in the JSON. "
            "Use --prediction-key pred_keypoints, or export with --smooth-keypoints "
            "to create smoothed_keypoints."
        )

    return frame_indices, pred_xyz, gt_xyz


def _extract_all_series(payload, prediction_key):
    keypoint_names = payload["keypoint_names"]
    frames = payload.get("frames", [])
    num_frames = len(frames)
    num_keypoints = len(keypoint_names)
    frame_indices = np.array(
        [int(frame.get("index", idx)) for idx, frame in enumerate(frames)],
        dtype=np.int64,
    )
    pred_xyz = np.full((num_frames, num_keypoints, 3), np.nan, dtype=np.float64)
    scores = np.full((num_frames, num_keypoints), np.nan, dtype=np.float64)
    gt_xyz = np.full((num_frames, num_keypoints, 3), np.nan, dtype=np.float64)
    saw_prediction_key = False

    for frame_idx, frame in enumerate(frames):
        keypoints = frame.get(prediction_key)
        if keypoints is not None:
            saw_prediction_key = True
            for kp in keypoints:
                jid = int(kp[0])
                if jid < 0 or jid >= num_keypoints:
                    continue
                pred_xyz[frame_idx, jid] = kp[1:4]
                scores[frame_idx, jid] = float(kp[4]) if len(kp) > 4 else 1.0

        pose = frame.get("gt_pose")
        if pose is not None:
            for jid, xyz in enumerate(pose[:num_keypoints]):
                gt_xyz[frame_idx, jid] = xyz

    if not saw_prediction_key:
        raise ValueError(f"Prediction key {prediction_key!r} was not found in the JSON.")
    return frame_indices, pred_xyz, scores, gt_xyz


def _finite_rows(arr):
    return np.isfinite(arr).all(axis=1)


def _coord_indices(coord_mode):
    return [1, 2] if coord_mode == "2d" else [0, 1, 2]


def _coord_label(coord_mode):
    return "2D Y-Z" if coord_mode == "2d" else "3D X-Y-Z"


def _project_coords(xyz, coord_mode):
    return xyz[..., _coord_indices(coord_mode)]


def _merge_projected_coords(source_xyz, filtered_projected, coord_mode):
    if coord_mode == "3d":
        return filtered_projected
    merged = np.array(source_xyz, copy=True)
    merged[..., _coord_indices(coord_mode)] = filtered_projected
    return merged


def _compute_error(pred_xyz, gt_xyz):
    valid = _finite_rows(pred_xyz) & _finite_rows(gt_xyz)
    error = np.full(pred_xyz.shape[0], np.nan, dtype=np.float64)
    error[valid] = np.linalg.norm(pred_xyz[valid] - gt_xyz[valid], axis=1)
    return error


def _compute_speed(pred_xyz, fps):
    speed = np.full(pred_xyz.shape[0], np.nan, dtype=np.float64)
    valid = _finite_rows(pred_xyz)
    for idx in range(1, pred_xyz.shape[0]):
        if valid[idx] and valid[idx - 1]:
            speed[idx] = np.linalg.norm(pred_xyz[idx] - pred_xyz[idx - 1]) * fps
    return speed


def _compute_acceleration(pred_xyz, fps):
    accel = np.full(pred_xyz.shape[0], np.nan, dtype=np.float64)
    valid = _finite_rows(pred_xyz)
    for idx in range(1, pred_xyz.shape[0] - 1):
        if valid[idx - 1] and valid[idx] and valid[idx + 1]:
            accel[idx] = (
                np.linalg.norm(pred_xyz[idx + 1] - 2.0 * pred_xyz[idx] + pred_xyz[idx - 1])
                * fps * fps
            )
    return accel


def _interpolate_xyz(pred_xyz, keep_mask):
    filtered = np.array(pred_xyz, copy=True)
    valid = _finite_rows(filtered) & keep_mask
    if valid.sum() == 0:
        return filtered

    frame_idx = np.arange(filtered.shape[0])
    for axis in range(filtered.shape[1]):
        values = filtered[:, axis]
        axis_valid = valid & np.isfinite(values)
        if axis_valid.sum() == 0:
            continue
        if axis_valid.sum() == 1:
            filtered[:, axis] = values[axis_valid][0]
        else:
            filtered[:, axis] = np.interp(frame_idx, frame_idx[axis_valid], values[axis_valid])
    return filtered


def _velocity_keep_mask(pred_xyz, fps, velocity_threshold):
    keep = _finite_rows(pred_xyz)
    if not np.isfinite(velocity_threshold):
        return keep

    last_kept = None
    for idx in range(pred_xyz.shape[0]):
        if not keep[idx]:
            continue
        if last_kept is None:
            last_kept = idx
            continue
        dt_frames = max(1, idx - last_kept)
        velocity = np.linalg.norm(pred_xyz[idx] - pred_xyz[last_kept]) * fps / dt_frames
        if velocity > velocity_threshold:
            keep[idx] = False
        else:
            last_kept = idx
    return keep


def _acceleration_keep_mask(pred_xyz, fps, acceleration_threshold):
    keep = _finite_rows(pred_xyz)
    if not np.isfinite(acceleration_threshold):
        return keep

    accel = _compute_acceleration(pred_xyz, fps)
    reject = np.isfinite(accel) & (accel > acceleration_threshold)
    keep[reject] = False
    return keep


def _apply_gates_to_joint(pred_xyz, fps, velocity_threshold, acceleration_threshold):
    keep = _finite_rows(pred_xyz)
    keep &= _velocity_keep_mask(pred_xyz, fps, velocity_threshold)
    keep &= _acceleration_keep_mask(pred_xyz, fps, acceleration_threshold)
    filtered = _interpolate_xyz(pred_xyz, keep)
    return filtered, keep


def _mean_error(pred_xyz, gt_xyz):
    error = _compute_error(pred_xyz, gt_xyz)
    if not np.isfinite(error).any():
        return np.inf
    return float(np.nanmean(error))


def _candidate_thresholds(values, grid_size, min_quantile, max_quantile):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [np.inf]
    qs = np.linspace(min_quantile, max_quantile, max(1, grid_size))
    candidates = np.unique(np.quantile(finite, qs))
    candidates = [float(v) for v in candidates if np.isfinite(v) and v >= 0.0]
    candidates.append(np.inf)
    return candidates


def _optimize_joint_gates(
    pred_xyz,
    gt_xyz,
    fps,
    grid_size,
    min_quantile,
    max_quantile,
    min_kept_ratio,
):
    valid_gt = _finite_rows(gt_xyz)
    valid_pred = _finite_rows(pred_xyz)
    valid_eval = valid_gt & valid_pred
    if valid_eval.sum() < 3:
        filtered = np.array(pred_xyz, copy=True)
        return filtered, {
            "velocity_threshold_mps": None,
            "acceleration_threshold_mps2": None,
            "raw_mean_error_m": None,
            "gated_mean_error_m": None,
            "kept_ratio": float(valid_pred.mean()) if valid_pred.size else 0.0,
            "rejected_frames": [],
        }

    speed = _compute_speed(pred_xyz, fps)
    accel = _compute_acceleration(pred_xyz, fps)
    velocity_candidates = _candidate_thresholds(
        speed, grid_size, min_quantile, max_quantile
    )
    acceleration_candidates = _candidate_thresholds(
        accel, grid_size, min_quantile, max_quantile
    )

    raw_error = _mean_error(pred_xyz, gt_xyz)
    best = {
        "error": raw_error,
        "velocity": np.inf,
        "acceleration": np.inf,
        "filtered": np.array(pred_xyz, copy=True),
        "keep": valid_pred,
    }

    min_kept = max(2, int(np.ceil(valid_pred.sum() * min_kept_ratio)))
    for velocity_threshold in velocity_candidates:
        velocity_keep = _velocity_keep_mask(pred_xyz, fps, velocity_threshold)
        for acceleration_threshold in acceleration_candidates:
            keep = velocity_keep & _acceleration_keep_mask(
                pred_xyz, fps, acceleration_threshold
            )
            if keep.sum() < min_kept:
                continue
            filtered = _interpolate_xyz(pred_xyz, keep)
            error = _mean_error(filtered, gt_xyz)
            if error < best["error"]:
                best = {
                    "error": error,
                    "velocity": velocity_threshold,
                    "acceleration": acceleration_threshold,
                    "filtered": filtered,
                    "keep": keep,
                }

    rejected = np.where(valid_pred & ~best["keep"])[0].astype(int).tolist()
    return best["filtered"], {
        "velocity_threshold_mps": (
            None if not np.isfinite(best["velocity"]) else float(best["velocity"])
        ),
        "acceleration_threshold_mps2": (
            None if not np.isfinite(best["acceleration"]) else float(best["acceleration"])
        ),
        "raw_mean_error_m": float(raw_error) if np.isfinite(raw_error) else None,
        "gated_mean_error_m": (
            float(best["error"]) if np.isfinite(best["error"]) else None
        ),
        "kept_ratio": float(best["keep"].sum() / max(1, valid_pred.sum())),
        "rejected_frames": rejected,
    }


def _optimize_all_joints(
    pred_xyz_all,
    scores_all,
    gt_xyz_all,
    fps,
    coord_mode,
    grid_size,
    min_quantile,
    max_quantile,
    min_kept_ratio,
):
    gated_xyz = np.array(pred_xyz_all, copy=True)
    threshold_info = []
    num_keypoints = pred_xyz_all.shape[1]
    for jid in range(num_keypoints):
        pred_eval = _project_coords(pred_xyz_all[:, jid, :], coord_mode)
        gt_eval = _project_coords(gt_xyz_all[:, jid, :], coord_mode)
        filtered_eval, info = _optimize_joint_gates(
            pred_eval,
            gt_eval,
            fps,
            grid_size,
            min_quantile,
            max_quantile,
            min_kept_ratio,
        )
        gated_xyz[:, jid, :] = _merge_projected_coords(
            pred_xyz_all[:, jid, :],
            filtered_eval,
            coord_mode,
        )
        info["joint_id"] = int(jid)
        info["coord_mode"] = coord_mode
        info["coord_indices"] = _coord_indices(coord_mode)
        threshold_info.append(info)

    return gated_xyz, scores_all, threshold_info


def _write_gated_json(
    payload,
    pred_json,
    gated_xyz,
    scores,
    threshold_info,
    output_path,
    gated_prediction_key,
    source_prediction_key,
    fps,
    coord_mode,
):
    output_payload = json.loads(json.dumps(payload))
    keypoint_names = output_payload["keypoint_names"]
    for frame_idx, frame in enumerate(output_payload.get("frames", [])):
        keypoints = []
        for jid in range(len(keypoint_names)):
            xyz = gated_xyz[frame_idx, jid]
            if not np.isfinite(xyz).all():
                continue
            score = scores[frame_idx, jid]
            if not np.isfinite(score):
                score = 1.0
            keypoints.append(
                [int(jid), float(xyz[0]), float(xyz[1]), float(xyz[2]), float(score)]
            )
        frame[gated_prediction_key] = keypoints

    output_payload["outlier_gating"] = {
        "source_prediction_key": source_prediction_key,
        "output_prediction_key": gated_prediction_key,
        "coord_mode": coord_mode,
        "coord_indices": _coord_indices(coord_mode),
        "coord_label": _coord_label(coord_mode),
        "fps": float(fps),
        "method": "velocity gate + centered acceleration gate + linear interpolation",
        "threshold_units": {
            "velocity_threshold_mps": "m/s",
            "acceleration_threshold_mps2": "m/s^2",
        },
        "overall_raw_mean_error_m": _mean_threshold_metric(
            threshold_info, "raw_mean_error_m"
        ),
        "overall_gated_mean_error_m": _mean_threshold_metric(
            threshold_info, "gated_mean_error_m"
        ),
        "per_joint": threshold_info,
    }

    output_path = (
        Path(output_path)
        if output_path
        else Path(pred_json).with_name(f"{Path(pred_json).stem}_gated.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    return output_path


def _mean_threshold_metric(threshold_info, key):
    values = [item.get(key) for item in threshold_info]
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))


def _default_output_path(pred_json, prediction_key, keypoint_id, keypoint_name, coord_mode):
    slug = _norm_name(keypoint_name) or f"kp{keypoint_id}"
    stem = Path(pred_json).stem
    return Path(pred_json).with_name(
        f"{stem}_{prediction_key}_{coord_mode}_kp{keypoint_id}_{slug}.png"
    )


def _nanmean_label(values, unit):
    if np.isfinite(values).any():
        return f"mean={np.nanmean(values):.2f} {unit}"
    return "mean=n/a"


def _axis_top(values, multiplier=2.5):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    top = float(np.percentile(finite, 99.5) * multiplier)
    if top <= 0.0 or not np.isfinite(top):
        return None
    return top


def main():
    args = parse_args()
    pred_json = Path(args.pred_json)
    with open(pred_json) as f:
        payload = json.load(f)

    keypoint_names = payload.get("keypoint_names")
    if not keypoint_names:
        raise ValueError("JSON is missing keypoint_names")

    keypoint_id = _resolve_keypoint_id(args.keypoint, keypoint_names)
    keypoint_name = keypoint_names[keypoint_id]

    fps = args.fps
    if fps is None:
        fps = float(payload.get("smoothing", {}).get("fps", 30.0))
    if fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.gate_grid_size < 1:
        raise ValueError("--gate-grid-size must be >= 1")
    if not 0.0 <= args.gate_min_quantile <= args.gate_max_quantile <= 1.0:
        raise ValueError("--gate-min-quantile and --gate-max-quantile must be in [0, 1]")
    if not 0.0 <= args.min_kept_ratio <= 1.0:
        raise ValueError("--min-kept-ratio must be in [0, 1]")

    gated_xyz = None
    if args.optimize_gates:
        frame_indices, pred_xyz_all, scores_all, gt_xyz_all = _extract_all_series(
            payload, args.prediction_key
        )
        gated_xyz_all, gated_scores, threshold_info = _optimize_all_joints(
            pred_xyz_all,
            scores_all,
            gt_xyz_all,
            fps,
            args.coord_mode,
            args.gate_grid_size,
            args.gate_min_quantile,
            args.gate_max_quantile,
            args.min_kept_ratio,
        )
        output_path = _write_gated_json(
            payload,
            pred_json,
            gated_xyz_all,
            gated_scores,
            threshold_info,
            args.gated_output_json,
            args.gated_prediction_key,
            args.prediction_key,
            fps,
            args.coord_mode,
        )
        joint_info = threshold_info[keypoint_id]
        overall_raw = _mean_threshold_metric(threshold_info, "raw_mean_error_m")
        overall_gated = _mean_threshold_metric(threshold_info, "gated_mean_error_m")
        print(f"Saved gated predictions to {output_path}")
        print(
            f"Overall mean error ({_coord_label(args.coord_mode)}): "
            f"raw={overall_raw} m, gated={overall_gated} m"
        )
        print(
            f"{keypoint_name}: velocity_threshold="
            f"{joint_info['velocity_threshold_mps']} m/s, "
            f"acceleration_threshold={joint_info['acceleration_threshold_mps2']} m/s^2, "
            f"raw_mean_error={joint_info['raw_mean_error_m']} m, "
            f"gated_mean_error={joint_info['gated_mean_error_m']} m"
        )
        pred_xyz = pred_xyz_all[:, keypoint_id, :]
        gt_xyz = gt_xyz_all[:, keypoint_id, :]
        gated_xyz = gated_xyz_all[:, keypoint_id, :]
    else:
        frame_indices, pred_xyz, gt_xyz = _extract_series(
            payload, keypoint_id, args.prediction_key
        )

    pred_eval = _project_coords(pred_xyz, args.coord_mode)
    gt_eval = _project_coords(gt_xyz, args.coord_mode)
    error = _compute_error(pred_eval, gt_eval)
    speed = _compute_speed(pred_eval, fps)
    accel = _compute_acceleration(pred_eval, fps)

    gated_error = gated_speed = gated_accel = None
    if gated_xyz is not None:
        gated_eval = _project_coords(gated_xyz, args.coord_mode)
        gated_error = _compute_error(gated_eval, gt_eval)
        gated_speed = _compute_speed(gated_eval, fps)
        gated_accel = _compute_acceleration(gated_eval, fps)

    scale = 100.0 if args.unit == "cm" else 1.0
    pos_unit = args.unit
    speed_unit = f"{args.unit}/s"
    accel_unit = f"{args.unit}/s^2"
    error_plot = error * scale
    speed_plot = speed * scale
    accel_plot = accel * scale
    gated_error_plot = None if gated_error is None else gated_error * scale
    gated_speed_plot = None if gated_speed is None else gated_speed * scale
    gated_accel_plot = None if gated_accel is None else gated_accel * scale

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(
        frame_indices,
        error_plot,
        color="tab:red",
        linewidth=1.5,
        label=f"{args.prediction_key} {_nanmean_label(error_plot, pos_unit)}",
    )
    if gated_error_plot is not None:
        axes[0].plot(
            frame_indices,
            gated_error_plot,
            color="tab:green",
            linewidth=1.4,
            linestyle="--",
            label=f"{args.gated_prediction_key} {_nanmean_label(gated_error_plot, pos_unit)}",
        )
    axes[0].set_ylabel(f"{_coord_label(args.coord_mode)} error ({pos_unit})")
    axes[0].set_title(
        f"{keypoint_name} (id {keypoint_id}) - {args.prediction_key} - "
        f"{_coord_label(args.coord_mode)}"
    )
    axes[0].legend(loc="upper right")
    # set y-axis limit to show most of the error distribution, but allow outliers to be visible
    error_top = _axis_top(
        np.concatenate(
            [error_plot, gated_error_plot]
            if gated_error_plot is not None else [error_plot]
        )
    )
    if error_top is not None:
        axes[0].set_ylim(bottom=0, top=error_top)

    axes[1].plot(
        frame_indices,
        speed_plot,
        color="tab:blue",
        linewidth=1.5,
        label=f"{args.prediction_key} {_nanmean_label(speed_plot, speed_unit)}",
    )
    if gated_speed_plot is not None:
        axes[1].plot(
            frame_indices,
            gated_speed_plot,
            color="tab:green",
            linewidth=1.4,
            linestyle="--",
            label=f"{args.gated_prediction_key} {_nanmean_label(gated_speed_plot, speed_unit)}",
        )
    axes[1].set_ylabel(f"Speed ({speed_unit})")
    axes[1].legend(loc="upper right")

    axes[2].plot(
        frame_indices,
        accel_plot,
        color="tab:purple",
        linewidth=1.5,
        label=f"{args.prediction_key} {_nanmean_label(accel_plot, accel_unit)}",
    )
    if gated_accel_plot is not None:
        axes[2].plot(
            frame_indices,
            gated_accel_plot,
            color="tab:green",
            linewidth=1.4,
            linestyle="--",
            label=f"{args.gated_prediction_key} {_nanmean_label(gated_accel_plot, accel_unit)}",
        )
    axes[2].set_ylabel(f"Acceleration ({accel_unit})")
    axes[2].set_xlabel("Frame index")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)

    fig.tight_layout()

    out_path = (
        Path(args.out)
        if args.out
        else _default_output_path(
            pred_json,
            args.prediction_key,
            keypoint_id,
            keypoint_name,
            args.coord_mode,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {out_path}")
    print(f"Frames: {len(frame_indices)}")
    print(f"Finite error frames: {np.isfinite(error).sum()}")
    print(f"Finite velocity frames: {np.isfinite(speed).sum()}")
    print(f"Finite acceleration frames: {np.isfinite(accel).sum()}")


if __name__ == "__main__":
    main()
