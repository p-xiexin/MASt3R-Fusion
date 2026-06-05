#!/bin/bash


GPU_ID=0
IMAGE_DIR="/path/to/kitti/image_00/data"
OUTPUT_DIR="mast3r_batch_pair_outputs"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
python tools/batch_dump_mast3r_pair_outputs.py \
    "$IMAGE_DIR" \
    "$OUTPUT_DIR" \
    --device cuda
