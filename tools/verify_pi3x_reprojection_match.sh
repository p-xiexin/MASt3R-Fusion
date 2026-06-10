#!/usr/bin/env bash
set -euo pipefail

IMAGE_A="${1:?Usage: $0 IMAGE_A IMAGE_B [extra verify_pi3x_reprojection_match.py args...]}"
IMAGE_B="${2:?Usage: $0 IMAGE_A IMAGE_B [extra verify_pi3x_reprojection_match.py args...]}"
shift 2

python tools/verify_pi3x_reprojection_match.py \
  --image-a "$IMAGE_A" \
  --image-b "$IMAGE_B" \
  --weights checkpoints/pi3x/model.safetensors \
  --device cuda:0 \
  --target-size 224 840 \
  --output-dir pi3x_reprojection_verify \
  "$@"
