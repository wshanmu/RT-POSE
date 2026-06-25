import itertools
import numpy as np

BATCH_SIZE = 64
NUM_KEYPOINTS = 18

BODY18_KEYPOINT_NAMES = [
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

tasks = [
    dict(num_class=1, class_names=["HipMidpoint"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

DATASET = dict(
  DIR=dict(
    ROOT_DIR='.',
    META_FILE='file_meta.txt',
    KEYPOINT_META='Keypoints_meta_body18.txt',
    RADAR_NPY_DIR='DZYX_npy_f16',
  ),
  LABEL=dict(
    IS_CONSIDER_ROI=True,
    ROI_TYPE='roi1',
    ROI_DEFAULT=[],
    IS_CHECK_VALID_WITH_AZIMUTH=False,
    MAX_AZIMUTH_DEGREE=[-50, 50],
    CONSIDER_RADAR_VISIBILITY=False,
  ),
  ROI=dict(
    roi1={'z': [-1.15, 1.40], 'y': [-1.25, 1.25], 'x': [0.9, 3.5]}
  ),
  RDR_TYPE='dzyx_real',
  RDR_CUBE=dict(
      IS_CONSIDER_ROI=True,
      ROI_TYPE='roi1',
      SHAPE=[16, 8, 64],
      GRID_SIZE=[2.6 / 63.0, 2.5 / 7.0, 2.55 / 15.0], # [x, y, z]
      NORMALIZING_VALUE=(0.0, 16.0),
  ),
  DZYX=dict(
    IS_CONSIDER_ROI=True,
    ROI_TYPE='roi1',
    SHAPE=[16, 8, 64],
    GRID_SIZE=[2.6 / 63.0, 2.5 / 7.0, 2.55 / 15.0], # [x, y, z]
    NORMALIZING_VALUE=(0.0, 16.0),
    REDUCE_TYPE='none',
  ),
  ENABLE_SENSOR=['RADAR'],
)

hr_final_conv_out = 256

model = dict(
    type="RadarPoseNet",
    pretrained=None,
    reader=dict(
        type='RadarFeatureNet',
    ),
    backbone=dict(
        type="HRNet3D",
        backbone_cfg='hr_tiny_feat128_zyx_l4_in128',
        final_conv_in=sum([128, 128, 256, 256]),
        final_conv_out=hr_final_conv_out,
        final_fuse='conat_conv',
        ds_factor=1,
    ),
    pose_head=dict(
      type='CenterHead',
      tasks=tasks,
      in_channels=hr_final_conv_out,
      share_conv_channel=256,
      dataset='cruw_pose',
      weight=0.5,
      code_weights=np.ones(NUM_KEYPOINTS * 3).tolist(),
      common_heads={'reg': (NUM_KEYPOINTS * 3, 2)},
      dcn_head=False,
    ),
    neck=None,
)

dataset_type = "CRUW_POSE_Dataset"

target_assigner = dict(
    tasks=tasks,
)

out_size_factor = [1, 1, 1]

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=out_size_factor,
    gaussian_overlap=0.1,
    max_poses=1,
    min_radius=2,
    num_keypoints=NUM_KEYPOINTS,
    consider_radar_visibility=DATASET['LABEL']['CONSIDER_RADAR_VISIBILITY'],
)

train_cfg = dict(assigner=assigner)

test_cfg_range = DATASET['ROI'][DATASET['LABEL']['ROI_TYPE']]
test_cfg = dict(
    post_center_limit_range=[test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0],
                             test_cfg_range['x'][1], test_cfg_range['y'][1], test_cfg_range['z'][1]],
    circular_nms=True,
    nms=dict(
        use_rotate_nms=False,
        use_multi_class_nms=False,
        nms_pre_max_size=1,
        nms_post_max_size=1,
        nms_iou_threshold=0.1,
    ),
    score_threshold=-1.0,
    pc_range=[test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0]],
    out_size_factor=out_size_factor,
    voxel_size=DATASET['DZYX']['GRID_SIZE'],
    input_type='rdr_cube',
)

train_pipeline = [
    dict(type="AssignLabelPose2", cfg=train_cfg["assigner"]),
]

test_pipeline = [
    dict(type="AssignLabelPose2", cfg=train_cfg["assigner"]),
]

data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Train.json',
        pipeline=train_pipeline,
        class_names=class_names,
    ),
    test=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Train.json',
        pipeline=test_pipeline,
        class_names=class_names,
    ),
    val=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Train.json',
        pipeline=test_pipeline,
        class_names=class_names,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
total_epochs = 50
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]

cuda_device = '0'
enable_amp = False
