# Training RT-Pose On The Custom Fitness Radar Dataset

This note explains how to train and evaluate the modified RT-Pose code on the
custom dataset where each session folder contains:

```text
<DATA_ROOT>/
  Session01/
    DZYX_npy_f16/
      00000.npy
      00001.npy
      ...
    Train.json
  Session02/
    DZYX_npy_f16/
    Train.json
  ...
```

The expected radar tensor saved in each `.npy` frame is:

```text
(D, Z, Y, X) = (128, 16, 4, 64)
```

The custom model config supports 23 keypoints and uses the one-heatmap RT-Pose
formulation: it predicts one pelvis/pose-center heatmap plus `23 * 3`
keypoint coordinate offsets.

## 1. Dataset Format

Each session `Train.json` should follow the RT-Pose-style nested structure:

```json
{
  "Session01": {
    "00000": [
      {
        "Radar_frameID": "00000",
        "pose": [
          [x0, y0, z0],
          [x1, y1, z1]
        ]
      }
    ]
  }
}
```

For this custom setup:

- top-level key: session name, usually the same as the session folder name
- frame key: camera/radar-aligned frame id, such as `00000`
- `Radar_frameID`: matching `.npy` filename stem inside `DZYX_npy_f16`
- `pose`: list of 23 `[x, y, z]` keypoints **in the radar Cartesian frame (metres)**

**Coordinate frame assumption** — the radar tensor axes are:

```text
x  (depth, index 0):  1.00 to 3.00 m
y  (lateral, index 1): -1.25 to 1.25 m
z  (height, index 2): -1.15 to 1.40 m
```

Ground-truth labels must use this same axis order.  The label generation script
`mmWaveRadar/get_ground_truth_label.py` projects 2D camera keypoints with a
fixed depth and saves them in radar frame order `[depth, lateral, height]`.
Run it once per session before building splits:

```bash
cd /data1/shanmu/ai-fitness-coach
python mmWaveRadar/get_ground_truth_label.py \
  --exp_name boelter_closer_session1 \
  --fixed_z 2.15
```

The script prints the coordinate ranges at the end — verify that `x` (depth)
falls in `[1.0, 3.0]`, `y` in `[-1.25, 1.25]`, and `z` in `[-1.15, 1.40]`.


## 2. Environment

Install the RT-Pose dependencies as described in the original README. The custom
radar-only path does not need Apex, `spconv`, deformable convolution, or iou3d
CUDA extensions.

For custom training/evaluation commands, set these flags to skip optional
detection modules:

```bash
export RTPOSE_DISABLE_IOU3D=1
export RTPOSE_DISABLE_SPCONV=1
```

You may still see harmless messages such as `no apex` or `Deformable Convolution
not built!` because this config uses `dcn_head=False`.

Additional Python dependencies for Hydra and Weights & Biases:

```bash
pip install hydra-core wandb
```


## 3. Smoke Test One Batch

Before training, verify that the dataloader and model can consume one batch.

```bash
cd /data1/shanmu/ai-fitness-coach/RT-POSE

RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
python tools/custom_batch_forward.py \
  --root-dir ../ssd_datas/fitness_data/synchronized/boelter_closer_session1 \
  --label-file Train.json \
  --radar-subdir DZYX_npy_f16 \
  --device cpu
```

Expected shapes:

```text
Batch radar tensor shape: (B, 128, 16, 4, 64)
Batch heatmap shape:      (B, 1, 16, 4, 64)
Batch pose target shape:  (B, 1, 69)
```

The predictions are random unless a checkpoint is passed with `--checkpoint`.


## 4. Build Leave-Session-Out Splits

RT-Pose can train from one merged JSON whose top-level keys are session names.
Use the helper script to merge per-session `Train.json` files into split files.

```bash
cd /data1/shanmu/ai-fitness-coach/RT-POSE

python tools/custom_make_session_split.py \
  --root-dir ../ssd_datas/fitness_data/synchronized \
  --train-sessions \
                   boelter_loc1_session_1 \
                   boelter_loc1_session_2 \
                   boelter_loc1_session_3 \
                   boelter_loc1_session_4 \
                   boelter_loc1_session_5 \
                   boelter_loc2_session_1 \
                   boelter_loc2_session_2 \
                   boelter_loc2_session_3 \
                   boelter_loc2_session_4 \
                   boelter_loc2_session_5 \
                   boelter_loc2_session_6 \
                   boelter_loc2_session_7 \
                   boelter_loc2_session_8 \
                   boelter_loc2_session_9 \
                   boelter_loc2_session_10 \
  --eval-sessions  printer_room_loc1_session_1 \
                   printer_room_loc1_session_2 \
                   printer_room_loc1_session_3 \
                   printer_room_loc1_session_4 \
                   printer_room_loc1_session_5 \
  --session-label Train.json \
  --out-dir leave_out_printer_eval_printer_loc1
```

This writes:

```text
<DATA_ROOT>/splits/train_sessions.json
<DATA_ROOT>/splits/eval_sessions.json
```

For multiple folds use `--train-out fold01_train.json --eval-out fold01_eval.json`.


## 5. Train

### 5.1 Hydra config

All overrideable hyperparameters live in `configs/hydra/train.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 4        # per GPU
  lr_max: 0.001
  lr_div_factor: 10.0
  lr_pct_start: 0.4
  eval_interval: 5     # run validation every N epochs

wandb:
  project: rt-pose-fitness
  name: null           # null → auto-generated by wandb
  tags: []
  mode: online         # online | offline | disabled
  notes: ""
  log_interval: 20     # log train metrics every N iterations
```

Pass any field as a trailing positional override on the command line
(Hydra `key=value` syntax).

### 5.2 Single GPU

```bash
cd /data1/shanmu/ai-fitness-coach/RT-POSE

CUDA_VISIBLE_DEVICES=2  \
PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=../ssd_datas/fitness_data/synchronized \
python tools/train.py configs/custom_fitness_body18/hr3d_one_hm_18j_dzyx_leaveout.py \
training.lr_max=0.0003 training.batch_size=128 \
RTPOSE_PRETRAINED=work_dirs/hr3d_one_hm_18j_dzyx_leaveout/polar4d/epoch_35.pth
```

### 5.3 Multi-GPU (torchrun)

`--gpus N` does **not** enable real multi-GPU training; use `torchrun` instead:

```bash
PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=../ssd_datas/fitness_data/synchronized \
torchrun --nproc_per_node=4 \
  tools/train.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --autoscale-lr
```

### Finetune the trained model with temporal module
```bash
RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 RTPOSE_DATA_ROOT=../ssd_datas/fitness_data/synchronized RTPOSE_PRETRAINED=work_dirs/hr3d_one_hm_23j_dzyx_leaveout/20260429_165334/epoch_35.pth RTPOSE_TEMPORAL_WINDOW=5 PYTHONPATH=. /data1/shanmu/envs/py39/bin/python tools/train.py   configs/custom_fitness/hr3d_one_hm_23j_dzyx_temporal.py   training.batch_size=32   training.lr_max=0.0001 training.epochs=10

RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 RTPOSE_DATA_ROOT=../ssd_datas/fitness_data/synchronized RTPOSE_PRETRAINED=work_dirs/hr3d_one_hm_23j_dzyx_leaveout/20260429_165334/epoch_35.pth RTPOSE_TEMPORAL_WINDOW=5 PYTHONPATH=. torchrun --nproc_per_node=4 tools/train.py   configs/custom_fitness/hr3d_one_hm_23j_dzyx_temporal.py   training.batch_size=32   training.lr_max=0.0001 training.epochs=10
```

`--autoscale-lr` multiplies `lr_max` by the number of GPUs to compensate for
the larger effective batch size.

### 5.4 Hydra overrides and named wandb runs

Append `key=value` pairs after the config path to override any field in
`configs/hydra/train.yaml`:

```bash
torchrun --nproc_per_node=4 \
  tools/train.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --autoscale-lr \
  training.lr_max=0.002 training.batch_size=8 training.epochs=100 \
  wandb.name=lr2e-3_bs8 "wandb.tags=[high_lr,bs8]"
```

Disable wandb logging (e.g. during debugging):

```bash
... tools/train.py configs/... wandb.mode=disabled
```

### 5.5 What is logged to wandb

| Metric | Frequency |
|---|---|
| `train/loss`, `train/hm_loss`, `train/loc_loss`, `train/num_positive` | Every `log_interval` iters |
| `train/lr` | Every `log_interval` iters |
| `val/MPJPE_mm`, `val/ABS_MPJPE_mm` | Every `eval_interval` epochs |

Hydra overrides are saved as the wandb run config, so you can compare
hyperparameter sweeps in the wandb table view.

### 5.6 Named fold training

```bash
RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/path/to/synchronized \
RTPOSE_TRAIN_LABEL=splits/fold01_train.json \
RTPOSE_EVAL_LABEL=splits/fold01_eval.json \
python tools/train.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  wandb.name=fold01
```

Checkpoints are written under:

```text
RT-POSE/work_dirs/hr3d_one_hm_23j_dzyx_leaveout/<wandb-run-name>/
```


## 6. Evaluate (offline, full test set)

```bash
cd /data1/shanmu/ai-fitness-coach/RT-POSE

PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=../ssd_datas/fitness_data/synchronized \
python tools/test.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --checkpoint work_dirs/hr3d_one_hm_23j_dzyx_leaveout/<run-name>/latest.pth \
  --testset
```

The script saves a `test_prediction.json` and prints per-sequence and overall
MPJPE / ABS-MPJPE in millimetres.

For a named fold:

```bash
RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/path/to/synchronized \
RTPOSE_TRAIN_LABEL=splits/fold01_train.json \
RTPOSE_EVAL_LABEL=splits/fold01_eval.json \
python tools/test.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --checkpoint work_dirs/hr3d_one_hm_23j_dzyx_leaveout/fold01/latest.pth \
  --testset
```


## 7. Visualize Predictions

`tools/visualize_session.py` runs the model on a session (or a single frame)
and saves a PNG per frame showing the radar intensity projections alongside the
predicted and ground-truth skeletons.  The PNGs can be assembled into a video
with `ffmpeg`.

### 7.1 Whole session

```bash
cd /data1/shanmu/ai-fitness-coach/RT-POSE

PYTHONPATH=. RTPOSE_DISABLE_IOU3D=1 RTPOSE_DISABLE_SPCONV=1 \
python tools/visualize_session.py \
  --session-dir ../ssd_datas/fitness_data/synchronized/boelter_closer_session13 \
  --checkpoint ./work_dirs/hr3d_one_hm_23j_dzyx_leaveout/20260429_165334/epoch_35.pth \
  --out-dir ./viz_session1 --skeleton-only --max-frames 20
```

### 7.2 Single npy frame

```bash
PYTHONPATH=. python tools/visualize_session.py \
  --npy-file ../ssd_datas/fitness_data/synchronized/boelter_closer_session13/DZYX_npy_f16/00042.npy \
  --checkpoint work_dirs/hr3d_one_hm_23j_dzyx_leaveout/<run-name>/latest.pth \
  --out-dir /tmp/viz_frame42 --skeleton-only
```

### 7.3 Useful flags

| Flag | Default | Description |
|---|---|---|
| `--max-frames N` | all | Process only the first N frames (quick check) |
| `--no-gt` | off | Skip GT overlay even if Train.json is present |
| `--device` | `cuda:0` | Device for inference |
| `--label-file` | `Train.json` | Label filename inside session dir |

### 7.4 Convert to video

```bash
ffmpeg -framerate 10 -i /tmp/viz_session13/%05d.png \
  -c:v libx264 -pix_fmt yuv420p output.mp4
```

Each frame panel layout:

```text
Row 1:  Bird's-eye (X-Y)  |  Side (X-Z)  |  Front (Y-Z)   ← radar heatmap + skeleton
Row 2:  3D view            |  Top-down     |  Side 3D        ← skeleton only
```

Green = prediction, Yellow = ground truth.  Abs-MPJPE (cm) is shown in the title.


## 8. Important Config Values

The main custom config is:

```text
configs/custom_fitness/hr3d_one_hm_23j_dzyx.py
```

The leave-session-out wrapper is:

```text
configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py
```

Key settings:

```python
NUM_KEYPOINTS = 23
DATASET['RDR_TYPE'] = 'dzyx_real'
DATASET['DZYX']['SHAPE'] = [16, 4, 64]
DATASET['DZYX']['GRID_SIZE'] = [2.0 / 63.0, 2.5 / 3.0, 2.55 / 15.0]  # [x, y, z]
DATASET['ROI']['roi1'] = {
    'z': [-1.15, 1.40],
    'y': [-1.25, 1.25],
    'x': [1.0, 3.0],
}
common_heads = {'reg': (NUM_KEYPOINTS * 3, 2)}
score_threshold = -1.0   # accept heatmap peak regardless of confidence
```

If you change the tensor grid or spatial range in `get_4D_tensor.py`, update
`SHAPE`, `GRID_SIZE`, and `ROI` in the config accordingly.

Batch size and learning rate live in `configs/hydra/train.yaml` and can be
overridden at the command line without editing any Python file.


## 9. Code Changes Made For The Custom Dataset

The original RT-Pose code assumed the authors' dataset layout, tensor shape, and
15-keypoint skeleton. These files were modified or added:

- `mmWaveRadar/get_ground_truth_label.py`
  - saves keypoints in radar frame order `[depth, lateral, height]` (was camera order)

- `det3d/datasets/cruw_pose/cruw_pose.py`
  - supports optional `file_meta.txt`
  - supports custom `RADAR_ROOT_DIR` and `RADAR_NPY_DIR`
  - supports tensors already cropped to a custom `SHAPE`
  - accepts empty poses without wrapping them as one invalid pose
  - evaluation matches keypoints by joint ID (robust to missing joints)
  - evaluation skips frames with empty or malformed GT pose

- `det3d/datasets/pipelines/pose.py`
  - `AssignLabelPose2` now uses configurable `num_keypoints`
  - pose regression target size is `num_keypoints * 3`, so 23 joints → 69 values

- `det3d/models/pose_heads/center_head.py`
  - prediction decoding returns however many keypoints the regression head predicts
  - removed the hard-coded 15-joint output loop

- `det3d/models/backbones/hrnet3D_config.py`
  - added `hr_tiny_feat128_zyx_l4_in128` for `(128, Z, Y, X)` Doppler-channel input

- `det3d/models/__init__.py`, `det3d/core/bbox/box_torch_ops.py`,
  `det3d/core/bbox/box_np_ops.py`, `det3d/torchie/trainer/checkpoint.py`
  - guards so radar-only training skips optional `spconv` and iou3d CUDA code
  - `torch.load` uses `weights_only=False` for PyTorch ≥ 2.6 compatibility

- `det3d/torchie/apis/train.py`
  - replaced `apex.parallel.convert_syncbn_model` with `torch.nn.SyncBatchNorm.convert_sync_batchnorm`
  - registers `_wandb_hook` and `_eval_hook` from `cfg` if present

- `tools/train.py`
  - Hydra `compose` API for hyperparameter overrides (DDP-safe, no CWD change)
  - Weights & Biases integration (`WandbHook` logs train metrics; `PoseEvalHook` logs val MPJPE)
  - `LOCAL_RANK` read from environment so `torchrun` multi-GPU works correctly
  - file log handler created on rank 0 only (fixes race condition on ranks 1-3)

- `tools/test.py`
  - removed hard-coded `/mnt/.../file_meta_merge.txt` dependency for saving predictions

- Added:
  - `configs/custom_fitness/hr3d_one_hm_23j_dzyx.py`
  - `configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py`
  - `configs/hydra/train.yaml`
  - `tools/custom_batch_forward.py`
  - `tools/custom_make_session_split.py`
  - `tools/visualize_session.py`
