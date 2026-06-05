#!/bin/bash


GPU_ID=0
IMAGE_A="data/img0.png"
IMAGE_B="data/img1.png"
OUTPUT_DIR="mast3r_pair_outputs"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
python tools/dump_mast3r_pair_outputs.py \
    "$IMAGE_A" \
    "$IMAGE_B" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda
