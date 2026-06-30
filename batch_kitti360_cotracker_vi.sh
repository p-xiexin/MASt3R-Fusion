#!/bin/bash

set -e

base_dataset_path="/home/pxx/Documents/MASt3R-Fusion/data/KITTI-360"
config_file="config/base_kitti360_cotracker.yaml"
calib_file="config/intrinsics_kitti360.yaml"
frontend_weights="checkpoints/pi3x/model.safetensors"
cotracker_checkpoint="checkpoints/cotracker/scaled_offline.pth"
imu_dt="-0.04"

# for folder in 0000 0002 0003 0004 0005 0006 0009 0010; do
for folder in 0000; do
    GPU_ID=0
    echo "Using GPU $GPU_ID for CoTracker sequence $folder"

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --dataset "${base_dataset_path}/2013_05_28_drive_${folder}_sync/image_00/data_rgb" \
        --config "$config_file" \
        --calib "$calib_file" \
        --imu_path "${base_dataset_path}/2013_05_28_drive_${folder}_sync/imu.txt" \
        --imu_dt "$imu_dt" \
        --stamp_path "${base_dataset_path}/2013_05_28_drive_${folder}_sync/camstamp.txt" \
        --result_path "result_cotracker_${folder}.txt" \
        --frontend-model cotracker \
        --frontend-weights "$frontend_weights" \
        --cotracker-checkpoint "$cotracker_checkpoint" \
        --save_h5 \
        --no-viz \
        --foxglove \
        --foxglove-host 0.0.0.0
    mv -f graph.pkl graph_cotracker_${folder}.pkl
    mv -f data.h5 data_cotracker_${folder}.h5
done
