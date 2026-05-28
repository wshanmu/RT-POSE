"""
Shared onnxruntime-TensorRT session and preprocessing helpers.

Uses onnxruntime's TensorRT EP instead of the raw TRT Python API.  This is
important because the TRT ONNX parser alone does not handle all ops in this
model (trilinear resize, GroupNorm patterns) — onnxruntime's TRT EP provides
automatic per-op fallback to CUDA/CPU for any op TRT cannot accelerate.

Requires: onnxruntime-gpu   (JetPack 6 ships this via pip)
Optional: cupy              (for zero-copy GPU → TRT inference)
"""

import os
import numpy as np
import onnxruntime as ort


def preprocess(path: str, norm_scale: float = 16.0) -> np.ndarray:
    """
    Load a raw radar npy (float32 magnitude) and apply log1p normalisation on CPU.

    Input npy : float32  (128, 16, Y, X)  raw magnitude (abs already taken)
    Output    : float32  (1, 128, 16, Y, X)  batch-ready, values in [0, ~1]
    """
    arr = np.load(path).astype(np.float32)
    arr = np.log1p(arr)
    arr /= norm_scale
    np.clip(arr, 0.0, None, out=arr)
    return np.ascontiguousarray(arr[np.newaxis])


def preprocess_cupy(cart_complex, norm_scale: float = 16.0):
    """
    GPU preprocessing starting from a cupy complex array (output of
    apply_polar_to_cart_mapper_cupy with output='complex').

    cart_complex : cupy complex64, shape (D, Z, Y, X) = (128, 16, Y, X)
    Returns      : cupy float32,   shape (1, D, Z, Y, X) — batch dim prepended
    """
    import cupy as cp
    magnitude = cp.abs(cart_complex)              # complex → float32
    magnitude = cp.log1p(magnitude)               # compress dynamic range
    magnitude /= cp.float32(norm_scale)
    cp.clip(magnitude, 0.0, None, out=magnitude)
    return cp.ascontiguousarray(magnitude[cp.newaxis])   # (1, D, Z, Y, X)


class TrtSession:
    """
    Onnxruntime session backed by the TensorRT EP with engine caching.

    On the first call the TRT engine is compiled and written to `cache_dir`.
    Every subsequent call (in the same or a new process) loads from cache and
    starts in < 1 s.

    Parameters
    ----------
    onnx_path : path to model.onnx
    cache_dir : directory for the compiled engine + timing cache
    fp16      : enable FP16 (recommended for Orin)
    """

    def __init__(self, onnx_path: str, cache_dir: str, fp16: bool = True):
        os.makedirs(cache_dir, exist_ok=True)

        trt_options = {
            "trt_fp16_enable":            fp16,
            "trt_engine_cache_enable":    True,
            "trt_engine_cache_path":      cache_dir,
            "trt_timing_cache_enable":    True,
            "trt_timing_cache_path":      cache_dir,
            "trt_builder_optimization_level": 3,
        }
        providers = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self._sess = ort.InferenceSession(onnx_path, providers=providers)

    def __call__(self, rdr_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference from a CPU numpy array.
        rdr_tensor : float32 ndarray, shape (1, D, Z, Y, X), preprocessed.
        Returns (hm, reg) as CPU numpy arrays.
        """
        binding = self._sess.io_binding()
        binding.bind_cpu_input("rdr_tensor", rdr_tensor)
        binding.bind_output("hm")
        binding.bind_output("reg")
        self._sess.run_with_iobinding(binding)
        hm  = binding.get_outputs()[0].numpy()
        reg = binding.get_outputs()[1].numpy()
        return hm, reg

    def run_cupy(self, arr) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference from a cupy GPU array.

        arr.get() converts cupy → numpy.  On Jetson Orin's unified memory
        (CPU and GPU share the same physical LPDDR5X), this is a cache-
        coherence flush, not a physical data copy, so the overhead is small.

        arr : cupy float32, shape (1, D, Z, Y, X), C-contiguous
        Returns (hm, reg) as CPU numpy arrays.
        """
        import cupy as cp
        arr_np = cp.asnumpy(cp.ascontiguousarray(arr))
        binding = self._sess.io_binding()
        binding.bind_cpu_input("rdr_tensor", arr_np)
        binding.bind_output("hm")
        binding.bind_output("reg")
        self._sess.run_with_iobinding(binding)
        hm  = binding.get_outputs()[0].numpy()
        reg = binding.get_outputs()[1].numpy()
        return hm, reg
