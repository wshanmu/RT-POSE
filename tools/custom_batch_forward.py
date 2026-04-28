import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RTPOSE_DISABLE_SPCONV", "1")
os.environ.setdefault("RTPOSE_DISABLE_IOU3D", "1")

import torch
from torch.utils.data import DataLoader


RT_POSE_ROOT = Path(__file__).resolve().parents[1]
if str(RT_POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(RT_POSE_ROOT))

from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one custom RT-Pose batch and run a forward pass."
    )
    parser.add_argument(
        "--config",
        default=str(RT_POSE_ROOT / "configs/custom_fitness/hr3d_one_hm_23j_dzyx.py"),
        help="RT-Pose config file.",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Dataset root. Use the session folder if Train.json and DZYX_npy_f16 are inside it.",
    )
    parser.add_argument("--label-file", default="Train.json")
    parser.add_argument("--radar-subdir", default="DZYX_npy_f16")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def update_dataset_cfg(cfg, args):
    root_dir = os.path.abspath(args.root_dir)
    for split_name in ("train", "val", "test"):
        split_cfg = cfg.data[split_name]
        split_cfg.label_file = args.label_file
        split_cfg.cfg.DATASET.DIR.ROOT_DIR = root_dir
        split_cfg.cfg.DATASET.DIR.RADAR_ROOT_DIR = root_dir
        split_cfg.cfg.DATASET.DIR.RADAR_NPY_DIR = args.radar_subdir
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.num_workers


def move_model_to_device(model, requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)
    return device


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    update_dataset_cfg(cfg, args)

    dataset = build_dataset(cfg.data.train)
    if args.num_workers > 0:
        data_loader = build_dataloader(
            dataset,
            batch_size=args.batch_size,
            workers_per_gpu=args.num_workers,
            dist=False,
            shuffle=False,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )

    data_batch = next(iter(data_loader))
    print(f"Loaded dataset samples: {len(dataset)}")
    print(f"Batch radar tensor shape: {tuple(data_batch['rdr']['rdr_tensor'].shape)}")
    if "hm" in data_batch["rdr"]:
        print(f"Batch heatmap shape: {tuple(data_batch['rdr']['hm'][0].shape)}")
    if "anno_pose" in data_batch["rdr"]:
        print(f"Batch pose target shape: {tuple(data_batch['rdr']['anno_pose'][0].shape)}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = move_model_to_device(model, args.device)
    model.eval()

    example = example_to_device(data_batch, device, non_blocking=False)
    with torch.no_grad():
        outputs = model(example, return_loss=False)

    print(f"Model output samples: {len(outputs)}")
    for sample_idx, output in enumerate(outputs[:2]):
        keypoints = output.get("keypoints", [])
        metadata = output.get("metadata", {})
        print(f"sample {sample_idx}: meta={metadata}, keypoints={len(keypoints)}")
        print(f"sample {sample_idx}: first_keypoints={keypoints[:3]}")


if __name__ == "__main__":
    main()
