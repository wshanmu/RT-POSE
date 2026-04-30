# workaround
import sys
sys.path.append('/home/andy/ipl/CRUW-POSE')
# workaround

import argparse
import json
import os
import sys
from datetime import datetime
import logging
from pathlib import Path
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import subprocess
import torch.distributed as dist

from det3d.datasets import build_dataset, build_dataloader
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import get_root_logger, set_random_seed, train_detector
from det3d.torchie.utils import count_parameters
from det3d.torchie.trainer.hooks import Hook
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer.utils import synchronize

# Hydra compose API — DDP-safe (no CWD change, no output dirs)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig

RT_POSE_ROOT = Path(__file__).resolve().parents[1]
HYDRA_CFG_DIR = str(RT_POSE_ROOT / "configs" / "hydra")


# ---------------------------------------------------------------------------
# Wandb hook — logs train metrics at every log_interval iters (rank 0 only)
# ---------------------------------------------------------------------------
class WandbHook(Hook):
    def __init__(self, wandb_run, log_interval: int, local_rank: int):
        self._run = wandb_run          # None on non-rank-0 or disabled
        self.log_interval = log_interval
        self.local_rank = local_rank

    def after_train_iter(self, trainer):
        if self._run is None:
            return
        should_log = (
            self.every_n_inner_iters(trainer, self.log_interval)
            or self.end_of_epoch(trainer)
        )
        if not should_log:
            return
        buf = trainer.log_buffer
        # Recompute the average if no logger has prepared the buffer yet, or if
        # another logger already cleared it.
        if not buf.ready:
            buf.average(self.log_interval)
        if not buf.output:
            return
        metrics = {}
        for k, v in buf.output.items():
            if k in ("time", "data_time", "transfer_time", "forward_time", "loss_parse_time"):
                continue
            if isinstance(v, (int, float)):
                metrics[f"train/{k}"] = v
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)):
                metrics[f"train/{k}"] = v[0]
        metrics["train/lr"] = trainer.current_lr()[0]
        metrics["train/epoch"] = trainer.epoch + (trainer.inner_iter + 1) / len(trainer.data_loader)
        self._run.log(metrics, step=trainer.iter)


# ---------------------------------------------------------------------------
# Eval hook — runs inference on the test split every N epochs (rank 0 only)
# ---------------------------------------------------------------------------
class PoseEvalHook(Hook):
    def __init__(self, eval_dataset, interval: int, cfg, local_rank: int, wandb_run=None):
        self.dataset = eval_dataset
        self.interval = interval
        self.cfg = cfg
        self.local_rank = local_rank
        self._run = wandb_run

    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return
        if self.local_rank != 0:
            synchronize()
            return

        device = next(trainer.model.parameters()).device
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        raw_model.eval()

        loader = build_dataloader(
            self.dataset,
            batch_size=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )
        detections = {}
        with torch.no_grad():
            for data_batch in loader:
                data_batch = example_to_device(data_batch, device, non_blocking=False)
                outputs = raw_model(data_batch, return_loss=False)
                for output in outputs:
                    meta = output.pop("metadata")
                    key = f"{meta['seq']}/{meta['frame']}/{meta['rdr_frame']}"
                    detections[key] = output

        result_dict, _ = self.dataset.evaluation(detections)
        if result_dict is not None:
            epoch = trainer.epoch + 1
            mpjpe     = result_dict["results"].get("MPJPE",     float("nan"))
            abs_mpjpe = result_dict["results"].get("ABS_MPJPE", float("nan"))
            trainer.logger.info(
                f"[Eval epoch {epoch}]  MPJPE={mpjpe:.2f} mm  ABS_MPJPE={abs_mpjpe:.2f} mm"
            )
            if self._run is not None:
                self._run.log({
                    "val/MPJPE_mm":     mpjpe,
                    "val/ABS_MPJPE_mm": abs_mpjpe,
                    "epoch": epoch,
                }, step=trainer.iter)

        raw_model.train()
        synchronize()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-Pose")
    parser.add_argument("config", help="det3d config file path")
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="Hydra-style overrides, e.g. training.lr_max=0.002 wandb.name=run1",
    )
    parser.add_argument("--work_dir")
    parser.add_argument("--resume_from")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--launcher", choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--autoscale-lr", action="store_true")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Load Hydra config with CLI overrides ---
    with initialize_config_dir(config_dir=HYDRA_CFG_DIR, version_base=None):
        hcfg: DictConfig = compose(config_name="train", overrides=args.overrides)

    # --- Load det3d config and apply Hydra overrides ---
    cfg = Config.fromfile(args.config)
    cfg.total_epochs                 = hcfg.training.epochs
    cfg.data.samples_per_gpu         = hcfg.training.batch_size
    cfg.lr_config.lr_max             = hcfg.training.lr_max
    cfg.lr_config.div_factor         = hcfg.training.lr_div_factor
    cfg.lr_config.pct_start          = hcfg.training.lr_pct_start
    if cfg.get("log_config", None) is not None:
        cfg.log_config.interval = int(hcfg.wandb.log_interval)
        for hook_cfg in cfg.log_config.hooks:
            if "ignore_last" not in hook_cfg:
                hook_cfg["ignore_last"] = False

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        run_name = hcfg.wandb.name or datetime.now().strftime('%Y%m%d_%H%M%S')
        cfg.work_dir = os.path.join(
            "./work_dirs", os.path.basename(args.config[:-3]), run_name
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # --- Distributed setup ---
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            cfg.local_rank = args.local_rank
        elif args.launcher == "slurm":
            proc_id   = int(os.environ["SLURM_PROCID"])
            ntasks    = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus  = torch.cuda.device_count()
            cfg.gpus  = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            os.environ.setdefault("MASTER_PORT", "29501")
            os.environ.setdefault("MASTER_ADDR", addr)
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"]       = str(proc_id)
            dist.init_process_group(backend="nccl")
            cfg.local_rank = int(os.environ["LOCAL_RANK"])
        cfg.gpus = dist.get_world_size()
    else:
        cfg.local_rank = args.local_rank

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # --- Work-dir and file logging (rank 0 only) ---
    if args.local_rank == 0:
        os.makedirs(os.path.join(cfg.work_dir, "det3d"), exist_ok=True)

    logger = get_root_logger(cfg.log_level)
    log_file_handler = None
    if args.local_rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_handler = logging.FileHandler(os.path.join(cfg.work_dir, f"exp_{ts}.log"))
        log_file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(log_file_handler)

    if args.local_rank == 0:
        with open(os.path.join(cfg.work_dir, "exp_config.py"), "w") as f:
            f.write(cfg.text)
        with open(os.path.join(cfg.work_dir, "hydra_overrides.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(hcfg))

    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(hcfg)}")
    logger.info(f"Distributed: {distributed}  |  GPUs: {getattr(cfg, 'gpus', 1)}")

    if args.seed is not None:
        set_random_seed(args.seed)

    # --- wandb (rank 0 only; disabled on other ranks) ---
    import wandb
    wandb_run = None
    if args.local_rank == 0 and hcfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            project=hcfg.wandb.project,
            name=hcfg.wandb.name or None,
            tags=list(hcfg.wandb.tags),
            notes=hcfg.wandb.notes,
            mode=hcfg.wandb.mode,
            config={
                **OmegaConf.to_container(hcfg.training, resolve=True),
                "det3d_config": args.config,
                "gpus": getattr(cfg, "gpus", 1),
            },
            dir=cfg.work_dir,
        )
        logger.info(f"wandb run: {wandb_run.url}")
    else:
        wandb.init(mode="disabled")   # keeps wandb.log() calls safe on non-rank-0

    # --- Build model and datasets ---
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(config=cfg.text, CLASSES=datasets[0].class_names)
    model.CLASSES = datasets[0].class_names

    if log_file_handler is not None:
        logger.removeHandler(log_file_handler)
        log_file_handler.close()

    # --- Register hooks via cfg ---
    eval_dataset = build_dataset(cfg.data.test)
    cfg._eval_hook   = PoseEvalHook(
        eval_dataset,
        interval=hcfg.training.eval_interval,
        cfg=cfg,
        local_rank=cfg.local_rank,
        wandb_run=wandb_run,
    )
    cfg._wandb_hook  = WandbHook(
        wandb_run=wandb_run,
        log_interval=hcfg.wandb.log_interval,
        local_rank=cfg.local_rank,
    )

    train_detector(model, datasets, cfg, distributed=distributed,
                   validate=args.validate, logger=logger)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
