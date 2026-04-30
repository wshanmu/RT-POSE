"""
Export RT-Pose to ONNX for TensorRT deployment on Jetson Orin AGX.

Run on the training machine after training completes:
PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
    python tools/export_onnx.py \
        configs/custom_fitness/hr3d_one_hm_23j_dzyx.py \
        --checkpoint /data1/shanmu/ai-fitness-coach/RT-POSE/work_dirs/hr3d_one_hm_23j_dzyx_leaveout/20260429_165334/epoch_35.pth \
        --sample-npy /data1/shanmu/ai-fitness-coach/ssd_datas/fitness_data/synchronized/boelter_session17/DZYX_npy_f16/00000.npy \
        --output /data1/shanmu/ai-fitness-coach/RT-POSE/rt_pose_epoch35_test.onnx --fp16

The exported model takes a single tensor input (rdr_tensor, already normalized)
and returns two raw (pre-sigmoid) tensors: hm and reg.
The lightweight decode step is handled in Python on Jetson.
"""

import sys
import warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint


# ---------------------------------------------------------------------------
# Thin wrapper: raw rdr_tensor → (hm, reg) — no dict, no meta, no post-proc
# ---------------------------------------------------------------------------
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.reader = model.reader
        self.backbone = model.backbone
        self.neck = model.neck if model.with_neck else None
        self.pose_head_convs = model.pose_head   # we only call forward(), not predict()

    def forward(self, rdr_tensor: torch.Tensor):
        x = self.reader(rdr_tensor)
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        preds, _ = self.pose_head_convs(x)
        # preds is a list of task dicts; we only have one task
        hm  = preds[0]['hm']   # [B, num_cls, Z, Y, X]
        reg = preds[0]['reg']  # [B, num_kp*3, Z, Y, X]
        return hm, reg


def load_sample_tensor(npy_path: str, cfg) -> torch.Tensor:
    """Load and preprocess one npy frame exactly as the dataloader does."""
    arr = np.load(npy_path).astype(np.float32)

    # Apply the same ROI crop as cruw_pose.py get_cube():
    # list_roi_idx_cb = [0, z_shape-1, 0, y_shape-1, 0, x_shape-1]
    dzyx_cfg = cfg.DATASET.DZYX
    z_shape, y_shape, x_shape = dzyx_cfg.SHAPE
    if arr.ndim == 4:
        arr = arr[:, 0:z_shape, 0:y_shape, 0:x_shape]   # [D, Z, Y, X]
    elif arr.ndim == 3:
        arr = arr[np.newaxis, 0:z_shape, 0:y_shape, 0:x_shape]  # add D=1 dummy

    norm_start, norm_scale = dzyx_cfg.NORMALIZING_VALUE
    arr = np.log1p(arr)
    arr = (arr - norm_start) / norm_scale
    arr[arr < 0.0] = 0.0

    # [D, Z, Y, X] → [1, D, Z, Y, X]
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',       help='det3d config file')
    parser.add_argument('--checkpoint', required=True, help='trained .pth file')
    parser.add_argument('--sample-npy', required=True,
                        help='one radar npy frame (to infer D and verify shapes)')
    parser.add_argument('--output',     default='rt_pose.onnx')
    parser.add_argument('--opset',      type=int, default=17,
                        help='ONNX opset (17 works with TRT 8.6+ / 10.x)')
    parser.add_argument('--fp16',       action='store_true',
                        help='cast model to fp16 before export (smaller ONNX)')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # Build model
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    # Sample input
    sample = load_sample_tensor(args.sample_npy, cfg)
    print(f'Input tensor shape : {tuple(sample.shape)}')   # [1, D, Z, Y, X]
    print(f'Input dtype        : {sample.dtype}')

    wrapper = ExportWrapper(model)
    wrapper.eval()

    if args.fp16:
        print('NOTE: --fp16 on x86 CPUs is emulated (slow). Exporting fp32 ONNX and '
              'using trtexec --fp16 on Jetson gives the same TRT engine — consider '
              'dropping --fp16 here.')
        wrapper = wrapper.half()
        sample  = sample.half()

    # Sanity-check forward on GPU to avoid slow CPU fp16 emulation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        hm, reg = wrapper.to(device)(sample.to(device))
    print(f'hm  shape : {tuple(hm.shape)}')    # [1, 1, Z, Y, X]
    print(f'reg shape : {tuple(reg.shape)}')   # [1, 69, Z, Y, X]

    # ONNX export runs a traced forward — keep on CPU for portability
    wrapper_cpu = wrapper.cpu()
    sample_cpu  = sample.cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.onnx')
        torch.onnx.export(
            wrapper_cpu,
            (sample_cpu,),
            str(output_path),
            opset_version=args.opset,
            input_names=['rdr_tensor'],
            output_names=['hm', 'reg'],
            dynamic_axes={
                'rdr_tensor': {0: 'batch'},
                'hm':         {0: 'batch'},
                'reg':        {0: 'batch'},
            },
        )
    print(f'\nONNX model saved → {output_path}')
    print('Next: copy to Jetson and run build_trt_engine.sh')


if __name__ == '__main__':
    main()
