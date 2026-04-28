import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-session RT-Pose Train.json files into train/eval split JSONs."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Parent folder containing session folders.",
    )
    parser.add_argument(
        "--train-sessions",
        nargs="+",
        required=True,
        help="Session folder names to use for training.",
    )
    parser.add_argument(
        "--eval-sessions",
        nargs="+",
        required=True,
        help="Session folder names to use for evaluation.",
    )
    parser.add_argument(
        "--session-label",
        default="Train.json",
        help="Label JSON filename inside each session folder.",
    )
    parser.add_argument(
        "--out-dir",
        default="splits",
        help="Output directory, relative to --root-dir unless absolute.",
    )
    parser.add_argument("--train-out", default="train_sessions.json")
    parser.add_argument("--eval-out", default="eval_sessions.json")
    return parser.parse_args()


def load_session_label(root_dir, session, session_label):
    label_path = root_dir / session / session_label
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    with label_path.open("r") as f:
        data = json.load(f)

    if session in data:
        return {session: data[session]}
    if len(data) == 1:
        only_key = next(iter(data))
        return {session: data[only_key]}
    raise ValueError(
        f"{label_path} has multiple top-level sessions and none match '{session}'."
    )


def merge_sessions(root_dir, sessions, session_label):
    merged = {}
    for session in sessions:
        merged.update(load_session_label(root_dir, session, session_label))
    return merged


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    num_frames = sum(len(frames) for frames in data.values())
    print(f"Wrote {path} ({len(data)} sessions, {num_frames} frames)")


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root_dir / out_dir

    train_data = merge_sessions(root_dir, args.train_sessions, args.session_label)
    eval_data = merge_sessions(root_dir, args.eval_sessions, args.session_label)

    overlap = set(train_data) & set(eval_data)
    if overlap:
        raise ValueError(f"Train/eval sessions overlap: {sorted(overlap)}")

    write_json(train_data, out_dir / args.train_out)
    write_json(eval_data, out_dir / args.eval_out)


if __name__ == "__main__":
    main()
