#!/bin/bash
[ -n "$BASH_VERSION" ] || exec /bin/bash "$0" "$@"
set -e

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-sparse_vio_demo/configs/kitti360_sparse.yaml}"
FRONTEND="${FRONTEND:-vins}"
OUT_DIR="${OUT_DIR:-sparse_vio_demo/output/kitti360_eval}"
EST="${EST:-$OUT_DIR/estimate.tum}"
RUN_MANIFEST="${RUN_MANIFEST:-$OUT_DIR/sparse_slam_run_manifest.json}"
RUN_SUMMARY="${RUN_SUMMARY:-$OUT_DIR/sparse_slam_run_summary.json}"
GT="${GT:-data/KITTI-360/2013_05_28_drive_0000_sync/gt_local.txt}"
EVO_HOME="${EVO_HOME:-$OUT_DIR/evo_home}"
PYTHON="${PYTHON:-/home/pxx/miniconda3/envs/mast3r_fusion/bin/python}"
EVO_APE="${EVO_APE:-/home/pxx/miniconda3/envs/mast3r_fusion/bin/evo_ape}"
EVO_RPE="${EVO_RPE:-/home/pxx/miniconda3/envs/mast3r_fusion/bin/evo_rpe}"
USE_IMU="${USE_IMU:-1}"
FOXGLOVE="${FOXGLOVE:-1}"
FOXGLOVE_HOST="${FOXGLOVE_HOST:-0.0.0.0}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
MIN_EVAL_FRAMES="${MIN_EVAL_FRAMES:-1200}"
ALLOW_SHORT_EVAL="${ALLOW_SHORT_EVAL:-0}"

start_value="${START:-0}"
subsample_value="${SUBSAMPLE:-1}"
if [ -n "${END:-}" ]; then
  frame_count=$(( (END - start_value + subsample_value - 1) / subsample_value ))
  if [ "$frame_count" -lt "$MIN_EVAL_FRAMES" ] && [ "$ALLOW_SHORT_EVAL" != "1" ]; then
    echo "Refusing short evaluation: frame_count=$frame_count < MIN_EVAL_FRAMES=$MIN_EVAL_FRAMES."
    echo "Use ALLOW_SHORT_EVAL=1 only for smoke/basic checks; do not treat that output as an effective experiment."
    exit 2
  fi
fi

mkdir -p "$OUT_DIR"
mkdir -p "$EVO_HOME"
rm -f "$OUT_DIR/ape_se3.zip" "$OUT_DIR/rpe_se3.zip"

cat > "$RUN_MANIFEST" <<JSON
{
  "frontend": "$FRONTEND",
  "use_imu": $([ "$USE_IMU" = "0" ] && echo false || echo true),
  "foxglove": {
    "enabled": $([ "$FOXGLOVE" = "1" ] && echo true || echo false),
    "host": "$FOXGLOVE_HOST",
    "port": $FOXGLOVE_PORT
  },
  "range": {
    "start": ${START:-null},
    "end": ${END:-null},
    "subsample": ${SUBSAMPLE:-1},
    "min_eval_frames": $MIN_EVAL_FRAMES
  },
  "outputs": {
    "estimate_tum": "$EST",
    "summary": "$RUN_SUMMARY"
  }
}
JSON

echo "[sparse_slam_eval] frontend=$FRONTEND use_imu=$USE_IMU out_dir=$OUT_DIR"
if [ "$FOXGLOVE" = "1" ]; then
  echo "[sparse_slam_eval] foxglove=ws://$FOXGLOVE_HOST:$FOXGLOVE_PORT"
fi

"$PYTHON" sparse_vio_demo/main.py \
  --config "$CONFIG" \
  --frontend "$FRONTEND" \
  --no-video \
  --result-tum "$EST" \
  --trajectory-frame body \
  $([ "$FOXGLOVE" = "1" ] && echo --foxglove --foxglove-host "$FOXGLOVE_HOST" --foxglove-port "$FOXGLOVE_PORT") \
  $([ "$USE_IMU" = "0" ] && echo --no-use-imu || echo --use-imu) \
  ${START:+--start "$START"} \
  ${END:+--end "$END"} \
  ${SUBSAMPLE:+--subsample "$SUBSAMPLE"}

"$PYTHON" sparse_vio_demo/summarize_sparse_run.py \
  --estimate-tum "$EST" \
  --out "$RUN_SUMMARY" \
  --frontend "$FRONTEND" \
  $([ "$USE_IMU" = "0" ] && echo --no-use-imu || echo --use-imu) \
  ${START:+--start "$START"} \
  ${END:+--end "$END"} \
  --subsample "${SUBSAMPLE:-1}" \
  --min-eval-frames "$MIN_EVAL_FRAMES"

HOME="$EVO_HOME" "$EVO_APE" tum "$GT" "$EST" --align --save_results "$OUT_DIR/ape_se3.zip"
HOME="$EVO_HOME" "$EVO_RPE" tum "$GT" "$EST" --align --delta 1 --delta_unit f --save_results "$OUT_DIR/rpe_se3.zip"
