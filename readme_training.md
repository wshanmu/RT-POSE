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
- `pose`: list of 23 `[x, y, z]` keypoints in meters

Important coordinate assumption: RT-Pose expects labels in the radar Cartesian
frame, not the camera pixel frame. The provided custom config currently uses:

```text
x:  1.00 to 3.00 m
y: -1.25 to 1.25 m
z: -1.15 to 1.40 m
```

If the labels are generated from camera keypoints, transform or remap them into
the radar coordinate frame before training.

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

## 3. Smoke Test One Batch

Before training, verify that the dataloader and model can consume one batch.

```bash
cd /home/shanmu/Projects/ai-fitness-coach/RT-POSE

python tools/custom_batch_forward.py \
  --root-dir /home/shanmu/Projects/ai-fitness-coach/ssd_datas/fitness_data/synchronized/Session01 \
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

Example: train on 10 sessions and evaluate on 5 held-out sessions.

```bash
cd /home/shanmu/Projects/ai-fitness-coach/RT-POSE

python tools/custom_make_session_split.py \
  --root-dir /home/shanmu/Projects/ai-fitness-coach/ssd_datas/fitness_data/synchronized \
  --train-sessions Session01 Session02 Session03 Session04 Session05 Session06 Session07 Session08 Session09 Session10 \
  --eval-sessions Session11 Session12 Session13 Session14 Session15 \
  --session-label Train.json
```

This writes:

```text
<DATA_ROOT>/splits/train_sessions.json
<DATA_ROOT>/splits/eval_sessions.json
```

If your per-session label file is lowercase, use:

```bash
--session-label train.json
```

For multiple folds, create fold-specific output names:

```bash
python tools/custom_make_session_split.py \
  --root-dir /path/to/synchronized \
  --train-sessions ... \
  --eval-sessions ... \
  --train-out fold01_train.json \
  --eval-out fold01_eval.json
```

## 5. Train

Use the custom leave-out config:

```bash
cd /home/shanmu/Projects/ai-fitness-coach/RT-POSE

RTPOSE_DISABLE_IOU3D=1 \
RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/home/shanmu/Projects/ai-fitness-coach/ssd_datas/fitness_data/synchronized \
python tools/train.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py
```

By default, the config reads:

```text
splits/train_sessions.json
splits/eval_sessions.json
```

To train a named fold:

```bash
RTPOSE_DISABLE_IOU3D=1 \
RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/path/to/synchronized \
RTPOSE_TRAIN_LABEL=splits/fold01_train.json \
RTPOSE_EVAL_LABEL=splits/fold01_eval.json \
python tools/train.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py
```

Checkpoints are written under:

```text
RT-POSE/work_dirs/custom_fitness_leaveout/
```

## 6. Evaluate

Evaluate the held-out sessions with:

```bash
RTPOSE_DISABLE_IOU3D=1 \
RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/home/shanmu/Projects/ai-fitness-coach/ssd_datas/fitness_data/synchronized \
python tools/test.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --checkpoint work_dirs/custom_fitness_leaveout/latest.pth \
  --testset
```

For a named fold:

```bash
RTPOSE_DISABLE_IOU3D=1 \
RTPOSE_DISABLE_SPCONV=1 \
RTPOSE_DATA_ROOT=/path/to/synchronized \
RTPOSE_TRAIN_LABEL=splits/fold01_train.json \
RTPOSE_EVAL_LABEL=splits/fold01_eval.json \
python tools/test.py configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py \
  --checkpoint work_dirs/custom_fitness_leaveout/latest.pth \
  --testset
```

The script saves prediction JSONs and prints MPJPE metrics. The evaluation
compares predicted keypoints against the `pose` arrays in the eval split JSON.

## 7. Important Config Values

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
DATASET['DZYX']['GRID_SIZE'] = [2.0 / 63.0, 2.5 / 3.0, 2.55 / 15.0]
DATASET['ROI']['roi1'] = {
    'z': [-1.15, 1.40],
    'y': [-1.25, 1.25],
    'x': [1.0, 3.0],
}
common_heads = {'reg': (NUM_KEYPOINTS * 3, 2)}
```

If you change the tensor grid or spatial range in `get_4D_tensor.py`, update
`SHAPE`, `GRID_SIZE`, and `ROI` in the config accordingly.

## 8. Code Changes Made For The Custom Dataset

The original RT-Pose code assumed the authors' dataset layout, tensor shape, and
15-keypoint skeleton. These files were modified or added:

- `det3d/datasets/cruw_pose/cruw_pose.py`
  - supports optional `file_meta.txt`
  - supports custom `RADAR_ROOT_DIR` and `RADAR_NPY_DIR`
  - supports tensors already cropped to a custom `SHAPE`
  - accepts empty poses without wrapping them as one invalid pose
  - evaluation now handles dynamic keypoint counts

- `det3d/datasets/pipelines/pose.py`
  - `AssignLabelPose2` now uses configurable `num_keypoints`
  - pose regression target size is `num_keypoints * 3`, so 23 joints become 69 values

- `det3d/models/pose_heads/center_head.py`
  - prediction decoding now returns however many keypoints the regression head predicts
  - removed the hard-coded 15-joint output loop

- `det3d/models/backbones/hrnet3D_config.py`
  - added `hr_tiny_feat128_zyx_l4_in128` for `(128, Z, Y, X)` Doppler-channel input

- `det3d/models/__init__.py`, `det3d/core/bbox/box_torch_ops.py`,
  `det3d/core/bbox/box_np_ops.py`, `det3d/torchie/trainer/checkpoint.py`
  - added guards so radar-only training can skip optional `spconv` and iou3d CUDA code

- `tools/test.py`
  - removed the hard-coded `/mnt/.../file_meta_merge.txt` dependency for saving predictions

- Added:
  - `configs/custom_fitness/hr3d_one_hm_23j_dzyx.py`
  - `configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py`
  - `tools/custom_batch_forward.py`
  - `tools/custom_make_session_split.py`

