import importlib
import os
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None and os.environ.get("RTPOSE_DISABLE_SPCONV", "0") != "1"
from .backbones import *  # noqa: F401,F403
if not found:
    print("No spconv, sparse convolution disabled!")
from .pose_heads import *  # noqa: F401,F403
from .builder import (
    build_backbone,
    build_detector,
    build_head,
    build_loss,
    build_neck,
    build_roi_head,
    build_feat_transform
)
from .detectors import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .readers import *
from.feat_transforms import *
from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    FEAT_TRANSFORMS
)
from .second_stage import * 
if os.environ.get("RTPOSE_DISABLE_IOU3D", "0") == "1":
    print("ROI heads disabled: RTPOSE_DISABLE_IOU3D=1")
else:
    try:
        from .roi_heads import * 
    except ImportError as e:
        print(f"ROI heads disabled: {e}")

__all__ = [
    "READERS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "FEAT_TRANSFORMS",
    "build_feat_transform",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
    "build_detector",
]
