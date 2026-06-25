from hr3d_one_hm_18j_dzyx_leaveout import *  # noqa: F401,F403
import copy
import os


TEMPORAL_WINDOW_SIZE = int(os.environ.get("RTPOSE_TEMPORAL_WINDOW", 5))

DATASET = copy.deepcopy(DATASET)
data = copy.deepcopy(data)
model = copy.deepcopy(model)

DATASET["TEMPORAL"] = dict(
    ENABLED=True,
    WINDOW_SIZE=TEMPORAL_WINDOW_SIZE,
    PAD_MODE="repeat_first",
)

for split in ["train", "val", "test"]:
    data[split]["cfg"]["DATASET"] = DATASET

model["type"] = "TemporalRadarPoseNet"
model["temporal_neck"] = dict(
    type="CausalFeatureTCN",
    in_channels=hr_final_conv_out,
    kernel_size=3,
    dilations=[1, 2],
    num_groups=8,
    dropout=0.0,
    zero_init_residual=True,
)

# Optional: initialize reader/backbone/head from a trained BODY_18 per-frame checkpoint.
model["pretrained"] = os.environ.get("RTPOSE_PRETRAINED", None)

work_dir = "./work_dirs/custom_fitness_body18_leaveout_temporal"
