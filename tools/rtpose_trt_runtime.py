"""Shared runtime helpers for RT-Pose TensorRT inference.

The PyTorch visualizer reads preprocessing and decode geometry from the
training config.  TensorRT inference must do the same; otherwise a valid engine
can be decoded with the wrong normalization, ROI, or voxel spacing.
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np


RT_POSE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RT_POSE_ROOT / "configs/custom_fitness/hr3d_one_hm_23j_dzyx_leaveout.py"


@dataclass(frozen=True)
class RuntimeSpec:
    config_path: str
    shape_zyx: tuple
    voxel_size: tuple
    pc_range: tuple
    roi: dict
    out_size_factor: tuple
    norm_start: float
    norm_scale: float
    num_keypoints: int
    score_threshold: float


def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({key: _to_attr_dict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_attr_dict(value) for value in obj)
    return obj


def _load_python_config(config_path: Path):
    module_name = f"_rtpose_runtime_config_{abs(hash(str(config_path)))}"
    config_dir = str(config_path.parent)
    sys.path.insert(0, config_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(config_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    cfg_dict = {
        name: value
        for name, value in module.__dict__.items()
        if not name.startswith("__")
    }
    return _to_attr_dict(cfg_dict)


def _as_float_tuple(values):
    return tuple(float(v) for v in values)


def _as_int_tuple(values):
    return tuple(int(v) for v in values)


def load_runtime_spec(config_path=None) -> RuntimeSpec:
    """Load preprocessing and decode settings from the training config."""
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        repo_relative = RT_POSE_ROOT / config_path
        if repo_relative.exists():
            config_path = repo_relative
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = _load_python_config(config_path)
    dzyx_cfg = cfg.DATASET.DZYX
    label_cfg = cfg.DATASET.LABEL
    roi_type = label_cfg.ROI_TYPE
    roi_cfg = cfg.DATASET.ROI[roi_type]

    norm_start = float(dzyx_cfg.NORMALIZING_VALUE[0])
    norm_scale = float(dzyx_cfg.NORMALIZING_VALUE[1]) - norm_start
    if norm_scale == 0.0:
        raise ValueError(
            f"Invalid NORMALIZING_VALUE in {config_path}: scale is zero"
        )

    test_cfg = _cfg_get(cfg, "test_cfg", {})
    voxel_size = _as_float_tuple(_cfg_get(test_cfg, "voxel_size", dzyx_cfg.GRID_SIZE))
    pc_range = _as_float_tuple(
        _cfg_get(
            test_cfg,
            "pc_range",
            [roi_cfg["x"][0], roi_cfg["y"][0], roi_cfg["z"][0]],
        )
    )
    out_size_factor = _as_int_tuple(_cfg_get(test_cfg, "out_size_factor", [1, 1, 1]))
    score_threshold = float(_cfg_get(test_cfg, "score_threshold", -1.0))

    common_heads = cfg.model.pose_head.common_heads
    reg_head = _cfg_get(common_heads, "reg", None)
    num_keypoints = int(_cfg_get(cfg, "NUM_KEYPOINTS", 0) or (int(reg_head[0]) // 3))

    roi = {
        "x": _as_float_tuple(roi_cfg["x"]),
        "y": _as_float_tuple(roi_cfg["y"]),
        "z": _as_float_tuple(roi_cfg["z"]),
    }

    return RuntimeSpec(
        config_path=str(config_path),
        shape_zyx=_as_int_tuple(dzyx_cfg.SHAPE),
        voxel_size=voxel_size,
        pc_range=pc_range,
        roi=roi,
        out_size_factor=out_size_factor,
        norm_start=norm_start,
        norm_scale=norm_scale,
        num_keypoints=num_keypoints,
        score_threshold=score_threshold,
    )


def format_runtime_spec(spec: RuntimeSpec) -> str:
    return (
        f"config={spec.config_path}\n"
        f"  DZYX shape={spec.shape_zyx}, keypoints={spec.num_keypoints}\n"
        f"  norm=(start={spec.norm_start:g}, scale={spec.norm_scale:g})\n"
        f"  voxel_size={spec.voxel_size}, pc_range={spec.pc_range}, "
        f"out_size_factor={spec.out_size_factor}"
    )


def preprocess(arr: np.ndarray, spec: RuntimeSpec) -> np.ndarray:
    """Match CRUW_POSE_Dataset.get_cube(): crop, log1p, normalize, clamp."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, :, :, :]
    if arr.ndim != 4:
        raise ValueError(f"Expected npy shape [D,Z,Y,X] or [Z,Y,X], got {arr.shape}")

    z_shape, y_shape, x_shape = spec.shape_zyx
    if arr.shape[1] < z_shape or arr.shape[2] < y_shape or arr.shape[3] < x_shape:
        raise ValueError(
            "Input radar tensor is smaller than config DATASET.DZYX.SHAPE: "
            f"arr={arr.shape}, required ZYX={spec.shape_zyx}"
        )
    arr = arr[:, :z_shape, :y_shape, :x_shape]
    arr = np.log1p(arr)
    arr = (arr - spec.norm_start) / spec.norm_scale
    np.maximum(arr, 0.0, out=arr)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def decode(hm: np.ndarray, reg: np.ndarray, spec: RuntimeSpec) -> dict:
    """Decode raw hm/reg tensors into 3D keypoints in metres."""
    if hm.ndim != 5 or reg.ndim != 5:
        raise ValueError(f"Expected hm/reg rank 5, got hm={hm.shape}, reg={reg.shape}")
    if hm.shape[0] != 1 or reg.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported here, got hm={hm.shape}")
    if reg.shape[1] % 3 != 0:
        raise ValueError(f"reg channel count must be divisible by 3, got {reg.shape}")

    num_keypoints_reg = reg.shape[1] // 3
    if num_keypoints_reg != spec.num_keypoints:
        raise ValueError(
            "Engine reg output keypoint count does not match config: "
            f"engine={num_keypoints_reg}, config={spec.num_keypoints}"
        )

    _, _, z_size, y_size, x_size = hm.shape
    hm_logits = np.clip(hm[0, 0], -60.0, 60.0)
    hm_sig = 1.0 / (1.0 + np.exp(-hm_logits))
    peak_idx = int(np.argmax(hm_sig.reshape(-1)))
    score = float(hm_sig.reshape(-1)[peak_idx])
    zi, yi, xi = np.unravel_index(peak_idx, (z_size, y_size, x_size))

    reg0 = reg[0]
    keypoints = np.zeros((spec.num_keypoints, 3), dtype=np.float32)
    for k in range(spec.num_keypoints):
        dx = reg0[3 * k, zi, yi, xi]
        dy = reg0[3 * k + 1, zi, yi, xi]
        dz = reg0[3 * k + 2, zi, yi, xi]
        keypoints[k, 0] = (
            (xi + dx) * spec.out_size_factor[2] * spec.voxel_size[0]
            + spec.pc_range[0]
        )
        keypoints[k, 1] = (
            (yi + dy) * spec.out_size_factor[1] * spec.voxel_size[1]
            + spec.pc_range[1]
        )
        keypoints[k, 2] = (
            (zi + dz) * spec.out_size_factor[0] * spec.voxel_size[2]
            + spec.pc_range[2]
        )

    return {
        "keypoints": keypoints,
        "score": score,
        "peak_index": (int(zi), int(yi), int(xi)),
    }


class TRTInference:
    """TensorRT runner with dynamic-shape-safe buffer allocation."""

    def __init__(self, engine_path: str):
        import torch

        jetpack_site_packages = "/usr/lib/python3/dist-packages"
        if jetpack_site_packages not in sys.path:
            sys.path.insert(0, jetpack_site_packages)
        import tensorrt as trt

        self._torch = torch
        self._trt = trt
        self._buffers = {}

        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode.name == "INPUT":
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected one input tensor, found {self.input_names}")

    def _torch_dtype(self, tensor_name):
        dtype = self.engine.get_tensor_dtype(tensor_name)
        dtype_name = str(dtype)
        if dtype_name == "DataType.HALF":
            return self._torch.float16
        if dtype_name == "DataType.FLOAT":
            return self._torch.float32
        raise TypeError(f"Unsupported TensorRT dtype for {tensor_name}: {dtype}")

    def _ensure_buffer(self, name, shape):
        shape = tuple(int(v) for v in shape)
        if any(v < 0 for v in shape):
            raise RuntimeError(f"Tensor {name} still has dynamic shape {shape}")
        dtype = self._torch_dtype(name)
        buf = self._buffers.get(name)
        if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
            buf = self._torch.empty(shape, dtype=dtype, device="cuda")
            self._buffers[name] = buf
        return buf

    def infer(self, rdr_tensor: np.ndarray) -> dict:
        """Run inference on numpy [1,D,Z,Y,X] float32, already preprocessed."""
        torch = self._torch
        input_name = self.input_names[0]
        rdr_tensor = np.ascontiguousarray(rdr_tensor)

        ok = self.context.set_input_shape(input_name, tuple(rdr_tensor.shape))
        if ok is False:
            raise RuntimeError(
                f"TensorRT rejected input shape {rdr_tensor.shape} for {input_name}"
            )

        in_buf = self._ensure_buffer(input_name, rdr_tensor.shape)
        src = torch.as_tensor(rdr_tensor, device="cuda", dtype=in_buf.dtype)
        in_buf.copy_(src)
        self.context.set_tensor_address(input_name, in_buf.data_ptr())

        for name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            out_buf = self._ensure_buffer(name, out_shape)
            self.context.set_tensor_address(name, out_buf.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        return {
            name: self._buffers[name].detach().cpu().float().numpy()
            for name in self.output_names
        }
