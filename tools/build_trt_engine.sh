#!/bin/bash
# Build TensorRT engine from ONNX on Jetson Orin AGX.
# Run this ON the Jetson after copying rt_pose.onnx over.
#
# Usage:
#   ./build_trt_engine.sh rt_pose.onnx rt_pose_fp16.engine
#
# Requirements:
#   - JetPack 5.x / 6.x with TensorRT (trtexec is at /usr/src/tensorrt/bin/trtexec)
#   - ~2-10 min build time (engine is cached for later runs)

set -euo pipefail

ONNX="${1:-rt_pose.onnx}"
ENGINE="${2:-rt_pose_fp16.engine}"
TRTEXEC="${TRTEXEC_PATH:-/usr/src/tensorrt/bin/trtexec}"

if [ ! -f "$ONNX" ]; then
  echo "ERROR: ONNX file not found: $ONNX"
  exit 1
fi

echo "=== Building TRT engine ==="
echo "  ONNX   : $ONNX"
echo "  Engine : $ENGINE"
echo "  trtexec: $TRTEXEC"

# --fp16 : use FP16 precision (Orin AGX has excellent FP16 throughput)
# --memPoolSize : workspace for layer fusion
# --saveEngine  : output engine file
"$TRTEXEC" \
  --onnx="$ONNX" \
  --fp16 \
  --memPoolSize=workspace:512 \
  --saveEngine="$ENGINE" \
  --minShapes=rdr_tensor:1x128x16x8x64 \
  --optShapes=rdr_tensor:1x128x16x8x64 \
  --maxShapes=rdr_tensor:1x128x16x8x64 \
  --verbose

echo ""
echo "=== Engine ready: $ENGINE ==="
echo "Run inference with:"
echo "  python tools/inference_rt.py --engine $ENGINE --npy-dir /path/to/DZYX_npy_f16"
