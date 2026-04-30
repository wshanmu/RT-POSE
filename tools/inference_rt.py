"""
Real-time RT-Pose inference on Jetson Orin AGX using TensorRT.

Usage (replay pre-captured npy frames):
    python tools/inference_rt.py \
        --engine rt_pose_fp16.engine \
        --npy-dir /path/to/DZYX_npy_f16 \
        [--fps 15] [--visualize]

The script:
  1. Loads the TRT engine once
  2. Loops over sorted npy files (or blocks on a live queue)
  3. Preprocesses each frame (log1p + normalize)
  4. Runs TRT inference
  5. Decodes hm + reg → 23 3D keypoints
  6. Optionally visualizes with matplotlib

Dependencies (Jetson):
    pip install tensorrt cuda-python numpy matplotlib
    (tensorrt and cuda-python come with JetPack — just add to Python path)
"""

import argparse
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Decode helpers (mirrors CenterHead.predict / post_processing in Python)
# ---------------------------------------------------------------------------

# From hr3d_one_hm_23j_dzyx.py:
_VOXEL_SIZE   = [2.0 / 63.0, 2.5 / 7.0, 2.55 / 15.0]   # [x, y, z] — from config GRID_SIZE
_PC_RANGE     = [1.0, -1.25, -1.15]                       # [x_min, y_min, z_min]
_OUT_FACTOR   = [1, 1, 1]
_NORM_START   = 0.0
_NORM_SCALE   = 16.0
_NUM_KP       = 23


def preprocess(arr: np.ndarray) -> np.ndarray:
    """log1p + normalize, matching cruw_pose.py get_cube(). No ROI crop needed
    since SHAPE=[16,8,64] matches the full npy shape (128,16,8,64)."""
    arr = np.log1p(arr)
    arr = (arr - _NORM_START) / _NORM_SCALE
    arr[arr < 0.0] = 0.0
    return arr.astype(np.float32)


def decode(hm: np.ndarray, reg: np.ndarray) -> dict:
    """
    Decode raw head outputs into 3D keypoints (numpy, no torch).

    hm  : [1, 1, Z, Y, X]   (raw logits, before sigmoid)
    reg : [1, K*3, Z, Y, X]  K = NUM_KP

    Returns dict with 'keypoints' ([K, 3] in metres) and 'score' (float).
    """
    _, _, Z, Y, X = hm.shape

    # sigmoid
    hm_sig = 1.0 / (1.0 + np.exp(-hm[0, 0]))    # [Z, Y, X]
    reg0   = reg[0]                               # [K*3, Z, Y, X]

    # flatten and find peak
    flat_hm  = hm_sig.reshape(-1)                # [Z*Y*X]
    peak_idx = int(np.argmax(flat_hm))
    score    = float(flat_hm[peak_idx])

    # unravel peak to grid coords
    zi, yi, xi = np.unravel_index(peak_idx, (Z, Y, X))

    # decode each keypoint from the regression offsets at the peak voxel
    keypoints = np.zeros((_NUM_KP, 3), dtype=np.float32)
    for k in range(_NUM_KP):
        dx = reg0[3*k,   zi, yi, xi]
        dy = reg0[3*k+1, zi, yi, xi]
        dz = reg0[3*k+2, zi, yi, xi]

        x_vox = xi + dx
        y_vox = yi + dy
        z_vox = zi + dz

        keypoints[k, 0] = x_vox * _OUT_FACTOR[2] * _VOXEL_SIZE[0] + _PC_RANGE[0]  # metres
        keypoints[k, 1] = y_vox * _OUT_FACTOR[1] * _VOXEL_SIZE[1] + _PC_RANGE[1]
        keypoints[k, 2] = z_vox * _OUT_FACTOR[0] * _VOXEL_SIZE[2] + _PC_RANGE[2]

    return {'keypoints': keypoints, 'score': score}


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------

class TRTInference:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import cuda                          # cuda-python

        self._trt  = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate page-locked host buffers and device buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        import cuda.cudart as cudart

        self.bindings     = []
        self.host_in      = []
        self.host_out     = []
        self.device_bufs  = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            size  = int(np.prod(shape))

            np_dtype = np.float16 if str(dtype) == 'DataType.HALF' else np.float32

            host_buf  = np.empty(size, dtype=np_dtype)
            err, dev_buf = cudart.cudaMalloc(host_buf.nbytes)
            assert err.value == 0, f'cudaMalloc failed: {err}'

            self.bindings.append(int(dev_buf))
            if self.engine.get_tensor_mode(name).name == 'INPUT':
                self.host_in.append((name, host_buf, dev_buf, shape))
            else:
                self.host_out.append((name, host_buf, dev_buf, shape))
                self.output_names.append(name)
            self.device_bufs.append(dev_buf)

    def infer(self, rdr_tensor: np.ndarray) -> dict:
        """rdr_tensor: numpy [1, D, Z, Y, X] float32 (already normalised)."""
        import cuda.cudart as cudart

        # Copy input to device
        name, host_buf, dev_buf, shape = self.host_in[0]
        np.copyto(host_buf, rdr_tensor.ravel().astype(host_buf.dtype))
        cudart.cudaMemcpy(
            int(dev_buf), host_buf.ctypes.data,
            host_buf.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        self.context.set_tensor_address(name, int(dev_buf))

        for out_name, _, dev_buf_out, _ in self.host_out:
            self.context.set_tensor_address(out_name, int(dev_buf_out))

        self.context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)

        # Copy outputs to host
        outputs = {}
        for out_name, host_buf_out, dev_buf_out, out_shape in self.host_out:
            cudart.cudaMemcpy(
                host_buf_out.ctypes.data, int(dev_buf_out),
                host_buf_out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
            outputs[out_name] = host_buf_out.reshape(out_shape).astype(np.float32)

        return outputs


# ---------------------------------------------------------------------------
# Optional live visualizer (matplotlib — disable if headless)
# ---------------------------------------------------------------------------

_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),          # left foot
    (0,5),(5,6),(6,7),(7,8),          # right foot
    (9,11),(11,13),(13,15),           # left leg
    (10,12),(12,14),(14,16),          # right leg
    (11,12),                          # hips
    (5,11),(6,12),                    # torso sides
    (5,6),                            # shoulders
    (5,7),(7,9),                      # left arm
    (6,8),(8,10),                     # right arm
]

def _visualize(keypoints: np.ndarray, score: float):
    """Non-blocking matplotlib 3D update."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    if not hasattr(_visualize, '_fig'):
        _visualize._fig = plt.figure(figsize=(6, 6), facecolor='white')
        _visualize._ax  = _visualize._fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

    ax = _visualize._ax
    ax.cla()
    ax.set_facecolor('white')
    ax.set_title(f'RT-Pose  score={score:.3f}', fontsize=10)

    for i, j in _SKELETON:
        ax.plot(*zip(keypoints[i], keypoints[j]), c='#37474F', lw=1.5)
    ax.scatter(*keypoints.T, c='#2979FF', s=20, zorder=5)

    # Equal-aspect cube around skeleton
    c = keypoints.mean(axis=0)
    h = max((keypoints.max(0) - keypoints.min(0)).max() / 2 + 0.15, 0.3)
    for setter, ci in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], c):
        setter(ci - h, ci + h)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

    _visualize._fig.canvas.draw()
    _visualize._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine',    required=True,
                        help='TRT engine file (built by build_trt_engine.sh)')
    parser.add_argument('--npy-dir',   required=True,
                        help='directory of DZYX npy frames')
    parser.add_argument('--fps',       type=float, default=15.0,
                        help='target playback / processing rate (Hz)')
    parser.add_argument('--visualize', action='store_true',
                        help='show 3D skeleton with matplotlib')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='stop after N frames (0 = unlimited)')
    args = parser.parse_args()

    npy_files = sorted(Path(args.npy_dir).glob('*.npy'))
    if not npy_files:
        raise FileNotFoundError(f'No npy files found in {args.npy_dir}')
    print(f'Found {len(npy_files)} frames in {args.npy_dir}')

    print('Loading TRT engine...')
    trt_model = TRTInference(args.engine)
    print('Engine ready.\n')

    frame_period = 1.0 / args.fps
    latencies = []

    for fi, npy_path in enumerate(npy_files):
        if args.max_frames and fi >= args.max_frames:
            break

        t0 = time.perf_counter()

        # Load + preprocess
        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis]              # add Doppler channel if missing
        arr = preprocess(arr)
        inp = arr[np.newaxis]                  # [1, D, Z, Y, X]

        # TRT inference
        t1 = time.perf_counter()
        outputs = trt_model.infer(inp)
        t2 = time.perf_counter()

        hm  = outputs['hm']
        reg = outputs['reg']
        result = decode(hm, reg)

        t3 = time.perf_counter()
        latencies.append((t2 - t1) * 1000)

        kp = result['keypoints']
        print(
            f'[{fi+1:04d}] {npy_path.name}  '
            f'score={result["score"]:.3f}  '
            f'infer={latencies[-1]:.1f} ms  '
            f'hip=({kp[11,0]:.2f}, {kp[11,1]:.2f}, {kp[11,2]:.2f})'
        )

        if args.visualize:
            _visualize(kp, result['score'])

        # Pace to target fps
        elapsed = time.perf_counter() - t0
        if elapsed < frame_period:
            time.sleep(frame_period - elapsed)

    if latencies:
        print(f'\n--- Stats over {len(latencies)} frames ---')
        print(f'  Mean inference : {np.mean(latencies):.1f} ms')
        print(f'  P99  inference : {np.percentile(latencies, 99):.1f} ms')
        print(f'  Max  inference : {np.max(latencies):.1f} ms')


if __name__ == '__main__':
    main()
