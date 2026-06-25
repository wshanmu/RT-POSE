from hr3d_one_hm_18j_dzyx import *  # noqa: F401,F403
import os

# Set RTPOSE_DATA_ROOT to the parent folder containing all session folders, e.g.
# /data1/shanmu/ai-fitness-coach/ssd_datas/fitness_data/synchronized
DATASET['DIR']['ROOT_DIR'] = os.environ.get('RTPOSE_DATA_ROOT', '.')
DATASET['DIR']['RADAR_ROOT_DIR'] = DATASET['DIR']['ROOT_DIR']
DATASET['DIR']['RADAR_NPY_DIR'] = os.environ.get('RTPOSE_RADAR_NPY_DIR', 'DZYX_npy_f16_doppler_compensated')

data['train']['cfg']['DATASET'] = DATASET
data['val']['cfg']['DATASET'] = DATASET
data['test']['cfg']['DATASET'] = DATASET

data['train']['label_file'] = os.environ.get('RTPOSE_TRAIN_LABEL', 'splits/train_sessions.json')
data['val']['label_file'] = os.environ.get('RTPOSE_EVAL_LABEL', 'splits/eval_sessions.json')
data['test']['label_file'] = os.environ.get('RTPOSE_EVAL_LABEL', 'splits/eval_sessions.json')

# A 23-joint checkpoint has an incompatible regression head for BODY_18 labels.
load_from = os.environ.get('RTPOSE_LOAD_FROM', None)
work_dir = './work_dirs/custom_fitness_body18_leaveout/'
