# Sparse Landmark SLAM Baseline

Current scope is deliberately small:

1. Run a sparse frontend.
2. Estimate camera/body poses and sparse landmarks with a GTSAM backend.
3. Publish tracking, trajectory, TF, and sparse landmarks through Foxglove.
4. Export a TUM trajectory and basic run summary for evaluation.

The previous PI3X/MASt3R fragment export and overlap-sweep code has been
removed from `sparse_vio_demo/`. Feed-forward model fusion should be rebuilt as
a separate module only after this sparse scaffold is stable enough to act as a
pose/keyframe/landmark provider.

## Kept Demo

Code kept under `sparse_vio_demo/`:

- `main.py`
- `configs/kitti360_sparse.yaml`
- `run_vins_foxglove.sh`
- `run_xfeat_foxglove.sh`
- `run_vio_eval_kitti360.sh`
- `summarize_sparse_run.py`
- `sparse_vio/backend/sparse_ba.py`
- `sparse_vio/frontend/vins_frontend.py`
- `sparse_vio/frontend/xfeat_frontend.py`
- dataset, geometry, IMU, trajectory, type, and visualization utilities.

## Run

```bash
START=0 END=1200 sparse_vio_demo/run_vio_eval_kitti360.sh
```

Foxglove is enabled by default in the evaluation script:

```text
ws://0.0.0.0:8765
```

Use a different port for smoke runs:

```bash
ALLOW_SHORT_EVAL=1 END=40 FOXGLOVE_PORT=8766 \
  sparse_vio_demo/run_vio_eval_kitti360.sh
```

## Boundary For Future Feed-Forward Fusion

Future PI3X/MASt3R fusion should consume this sparse scaffold through explicit
artifacts:

- selected keyframes,
- sparse poses,
- sparse landmarks,
- track IDs and observations,
- camera calibration.

It should not be mixed back into the minimal sparse frontend/backend demo.
