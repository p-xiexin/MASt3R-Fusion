#!/bin/bash
[ -n "$BASH_VERSION" ] || exec /bin/bash "$0" "$@"
set -e

cd "$(dirname "$0")/.."

HOST="${FOXGLOVE_HOST:-0.0.0.0}"
PORT="${FOXGLOVE_PORT:-8765}"
CONFIG="${CONFIG:-sparse_vio_demo/configs/kitti360_sparse.yaml}"
PYTHON="${PYTHON:-/home/pxx/miniconda3/envs/mast3r_fusion/bin/python}"
USE_IMU="${USE_IMU:-0}"

"$PYTHON" sparse_vio_demo/main.py \
  --config "$CONFIG" \
  --frontend vins \
  --no-video \
  --foxglove \
  --foxglove-host "$HOST" \
  --foxglove-port "$PORT" \
  $([ "$USE_IMU" = "1" ] && echo --use-imu || echo --no-use-imu) \
  ${START:+--start "$START"} \
  ${END:+--end "$END"} \
  ${SUBSAMPLE:+--subsample "$SUBSAMPLE"}
