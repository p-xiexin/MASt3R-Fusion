# Sparse VIO Demo

Minimal sparse frontend plus GTSAM backend demo, isolated from the
MASt3R/PI3X fusion runtime.

## Kept Scope

- `vins` frontend: VINS-style feature lifecycle with KLT tracking.
- `xfeat` frontend: XFeat feature spawning plus the same KLT tracking path.
- `SparseBackend`: GTSAM sparse landmark backend with optional IMU factors.
- Foxglove debug overlay for tracking image, trajectory, body/camera TF, and
  sparse landmarks.
- KITTI-360 run/evaluation scripts.

The demo does not contain PI3X/MASt3R dense reconstruction, fragment graphs,
feed-forward window alignment, or alternate VINS backend rewrites.

## Run

```bash
/home/pxx/miniconda3/envs/mast3r_fusion/bin/python \
  sparse_vio_demo/main.py \
  --config sparse_vio_demo/configs/kitti360_sparse.yaml \
  --frontend vins
```

Foxglove quick runs:

```bash
sparse_vio_demo/run_vins_foxglove.sh
sparse_vio_demo/run_xfeat_foxglove.sh
```

The Foxglove server binds to `0.0.0.0:8765` by default in the quick/eval
scripts. Open `ws://<host-ip>:8765`.

Published topics:

- `/sparse_vio/image`
- `/sparse_vio/tracking_image`
- `/tf`
- `/sparse_vio/current_pose`
- `/sparse_vio/trajectory`
- `/sparse_vio/points`
- `/sparse_vio/sparse_points`

## Evaluation

Effective runs should use at least 1200 frames:

```bash
START=0 END=1200 sparse_vio_demo/run_vio_eval_kitti360.sh
```

Visual-only run:

```bash
USE_IMU=0 START=0 END=1200 sparse_vio_demo/run_vio_eval_kitti360.sh
```

The evaluation script writes:

- `estimate.tum`
- `sparse_slam_run_manifest.json`
- `sparse_slam_run_summary.json`
- `ape_se3.zip`
- `rpe_se3.zip`

Short runs are smoke checks only:

```bash
ALLOW_SHORT_EVAL=1 END=40 FOXGLOVE_PORT=8766 \
  sparse_vio_demo/run_vio_eval_kitti360.sh
```
