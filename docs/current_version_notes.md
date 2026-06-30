# Current Version Notes

This document records the current working changes and the checks needed when
deploying or debugging this branch on a server.

## Change Summary

### Frontend Model Adapter

- `main.py` now accepts frontend model selection:
  - `--frontend-model mast3r`
  - `--frontend-model pi3`
  - `--frontend-model pi3x`
  - `--frontend-weights <checkpoint_path>`
- `mast3r_fusion/frontend_model/factory.py` maps both `pi3` and `pi3x` to the PI3X adapter.
- `mast3r_fusion/frontend_model/pi3_adapter.py` is implemented as the adapter entrypoint.
- `mast3r_fusion/frontend_model/pi3x_utils.py` provides PI3X loading, pair inference, window inference, mono bootstrap inference, asymmetric matching, and symmetric batch matching.

The PI3X implementation assumes PI3X can run on a pair of images, similar to how
MASt3R is used in this project. It adapts PI3X pair outputs into the existing
backend contract:

- `Xii`, `Xji`: point maps used by pose tracking and factor construction.
- `Cii`, `Cji`: point confidence maps.
- `Qii`, `Qji`: match confidence maps, currently derived from PI3X confidence.
- `idx_i2j`: dense correspondence index from the existing projection matcher.

PI3X does not expose MASt3R's private `_decoder` / `_downstream_head` API or a
MASt3R-compatible descriptor head. Matching currently uses PI3X point maps,
camera poses, and confidence to derive geometric dense correspondences.

### PI3X Directly Adapted Capabilities

The current adapter does use several PI3X outputs directly instead of
fabricating the whole frontend result:

- PI3X directly supports multi-image forward inference with input shaped like
  `(B, N, 3, H, W)`. The adapter uses this path for pair inference, window
  inference, and same-frame mono bootstrap inference.
- PI3X directly predicts dense per-view local point maps. The adapter maps
  `local_points` / `points_local` / `pts3d` to MASt3R-Fusion point maps such as
  `Xii` and `Xjj`.
- PI3X directly predicts per-view camera poses. The adapter uses
  `camera_poses` / `poses` / `extrinsics` to transform one view's local point
  map into the other view's coordinate convention, producing `Xji` and `Xij`
  for the existing geometric matcher.
- PI3X camera poses are required for geometry matching. If the model output
  does not contain camera poses, the PI3X path fails instead of fabricating
  degraded correspondences.
- PI3X directly predicts point confidence. The adapter converts PI3X `conf`
  logits with `sigmoid()` and uses them as point confidence `C` and provisional
  match confidence `Q`.
- PI3X window inference refreshes the recent keyframe point maps and stores the
  PI3X camera poses. Cached point maps are reused for backend factors only when
  both frames come from the same PI3X window, avoiding mixed local coordinate
  frames.
- In PI3X window mode, backend visual optimization is queued only after the
  keyframe count reaches `pi3x.window_size`, so local factors are not built from
  early under-filled windows.
- `config/base_kitti360_pi3x.yaml` disables automatic legacy VI initialization
  with `ms_opt.enable_vi_init=false`. The current PI3X prototype keeps the
  visual Sim3BA path active first; enabling the legacy VI init currently needs
  separate validation because it can create inconsistent GTSAM graphs after
  pending keyframes are committed in batches.
- The lightweight frontend uses XFeat keypoints, LK optical flow, a local sparse
  map, and PnP-RANSAC to provide only a relative pose prior. After backend
  optimization, the recent sparse map is refreshed from shared keyframes so it
  sees the corrected keyframe poses and PI3X point maps.

### PI3X Adapter Core Limitations

PI3X is not a drop-in MASt3R matcher. The current adapter is intended for
server-side smoke testing first, and these limitations must be checked before
trusting full SLAM results:

- PI3X does not expose MASt3R-style dense descriptors. The current backend path
  therefore relies on geometric projection matching from PI3X point maps and
  camera poses rather than descriptor refinement.
- PI3X does not output explicit pair matching results such as `idx_i2j`,
  `idx_j2i`, or `valid_match`. The adapter derives dense correspondences from
  PI3X `local_points` and `camera_poses`, then runs the existing projection
  matcher. This is a geometric approximation, not a native PI3X match head.
- PI3X point confidence has a different scale from MASt3R descriptor
  confidence. PI3X `conf` is treated as raw logits and converted with
  `sigmoid()`, producing values in `[0, 1]`. Use PI3X-specific `Q_conf`
  thresholds, such as those in `config/base_kitti360_pi3x.yaml`, instead of the
  MASt3R default threshold `1.5`.

### Retrieval Behavior

The MASt3R retrieval database depends on a MASt3R-compatible backbone. For
non-MASt3R frontends, `main.py` sets `retrieval_database = None` and disables
retrieval-based loop candidates. Consecutive local factors still run.

### Checkpoint And Local Documentation

Do not modify the upstream-style `README.md` for local adapter deployment notes.
Keep PI3X setup and server debugging details in this document and `AGENTS.md`.
Install PI3X as an editable third-party package:

```bash
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
pip install -e thirdparty/Pi3

mkdir -p checkpoints/pi3x/
wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors -O checkpoints/pi3x/model.safetensors
```

Use:

```bash
python main.py --frontend-model pi3x --frontend-weights checkpoints/pi3x/model.safetensors ...
```

### KITTI-360 Evaluation Plot Labels

`evaluation/evaluate_kitti360.py` has local chart naming and labeling cleanup:

- meaningful figure names,
- sequence-specific output filenames,
- titles, axes, legends, and grids.

This change is unrelated to the PI3X adapter and should be reviewed separately
before committing if the branch should stay focused on frontend adapter work.

### Foxglove Debug Topics

The Foxglove publisher now includes lightweight frontend diagnostics:

- `/current_image`: current tracking image with LK tracks drawn from the
  reference sparse map to the current frame. Green points/segments are PnP
  inliers; red ones are rejected tracks.
- `/current_image/camera_info`: camera calibration for the image panel.
- The overlay text shows reference frame id, current frame id, inlier count,
  inlier ratio, median parallax, and median reprojection error.

Run the PI3X batch with Foxglove enabled:

```bash
FOXGLOVE=1 GPU_ID=0 SEQ=0000 START_FROM=0 END_AT=20 bash batch_kitti360_pi3x_vi.sh
```

Default batch runs stay headless and do not start Foxglove.

## Server Deployment Checklist

### 1. Sync The Correct Branch

On the server:

```bash
git fetch origin
git checkout frontend-model-adapter
git reset --hard origin/frontend-model-adapter
```

Only use `reset --hard` when there are no server-local changes to keep. Check
first:

```bash
git status --short
```

### 2. Install Project Dependencies

```bash
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

If testing PI3X, install the PI3 codebase as an editable third-party package:

```bash
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
pip install -e thirdparty/Pi3
```

The PI3X lightweight frontend uses XFeat as a hard dependency. The
`thirdparty*` path is git-ignored, so install it on every runtime machine:

```bash
git clone https://github.com/verlab/accelerated_features.git thirdparty/xfeat
pip install -r thirdparty/xfeat/requirements.txt
test -f thirdparty/xfeat/weights/xfeat.pt
```

If the PI3X checkpoint is a `.safetensors` file, the environment must include
`safetensors`. Verify with:

```bash
python -c "import safetensors; print('safetensors ok')"
```

### 3. Verify Checkpoints

For MASt3R:

```bash
test -f checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

For PI3X:

```bash
test -f checkpoints/pi3x/model.safetensors
```

If either command returns non-zero, download the missing checkpoint before
running.

### 4. Smoke Test Imports

Run these before a full sequence:

```bash
python -m compileall main.py mast3r_fusion/frontend_model mast3r_fusion/light_tracker.py
python -c "from mast3r_fusion.frontend_model import load_frontend_model; print('frontend import ok')"
python -c "from pi3.models.pi3x import Pi3X; print('pi3x import ok')"
python -c "from cotracker.predictor import CoTrackerPredictor; print('cotracker import ok')"
python -c "import sys; sys.path.insert(0, 'thirdparty/xfeat'); from modules.xfeat import XFeat; XFeat(weights='thirdparty/xfeat/weights/xfeat.pt', top_k=16); print('xfeat ok')"
```

For CoTracker, prefer an explicit local pretrained checkpoint on servers without
network access. If `--cotracker-checkpoint` / `COTRACKER_CHECKPOINT` is omitted,
the adapter falls back to `torch.hub.load("facebookresearch/co-tracker",
"cotracker3_offline")`.

If `torch` is missing, these imports will fail before reaching project code.
Confirm the server is using the intended conda or virtualenv:

```bash
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Runtime Validation Plan

### MASt3R Baseline

Before testing PI3X, run the branch with MASt3R to ensure the adapter refactor did
not break existing behavior:

```bash
python main.py \
  --frontend-model mast3r \
  --config config/base_kitti360.yaml \
  --calib config/intrinsics_kitti360.yaml \
  --dataset <dataset_path> \
  --imu_path <imu_path> \
  --stamp_path <stamp_path> \
  --result_path result_mast3r.txt \
  --no-viz
```

Expected:

- model loads from the existing MASt3R checkpoint,
- first frame initializes a point map,
- tracking prints `track` progress,
- result file is written.

### PI3X Smoke Run

Use a short range first:

```bash
python main.py \
  --frontend-model pi3x \
  --frontend-weights checkpoints/pi3x/model.safetensors \
  --config config/base_kitti360_pi3x.yaml \
  --calib config/intrinsics_kitti360.yaml \
  --dataset <dataset_path> \
  --imu_path <imu_path> \
  --stamp_path <stamp_path> \
  --start_from 0 \
  --end_at 20 \
  --result_path result_pi3x_smoke.txt \
  --no-viz
```

Expected:

- PI3X model loads,
- first frame runs `infer_single`,
- light tracking logs `light ...` lines in `tracker.log`,
- keyframes run PI3X window inference once enough keyframes are available,
- no MASt3R retrieval database error appears,
- `result_pi3x_smoke.txt` is created.

Summarize frontend health after the run:

```bash
python tools/summarize_tracker_log.py tracker.log
```

The batch smoke script defaults to a short headless run and does not write H5
debug output unless requested:

```bash
GPU_ID=0 SEQ=0005 START_FROM=0 END_AT=20 bash batch_kitti360_pi3x_vi.sh
SAVE_H5=1 GPU_ID=0 SEQ=0005 START_FROM=0 END_AT=20 bash batch_kitti360_pi3x_vi.sh
```

### CoTracker Lifecycle Smoke Run

Use the dedicated CoTracker config and keep the first run short:

```bash
python tools/validate_cotracker_lifecycle.py
python tools/preflight_cotracker_runtime.py \
  --config config/base_kitti360_cotracker.yaml \
  --dataset-root <kitti360_root> \
  --seq 0005 \
  --pi3x-weights checkpoints/pi3x/model.safetensors \
  --cotracker-checkpoint <cotracker_checkpoint_or_omit_for_torchhub> \
  --require-cuda

python main.py \
  --frontend-model cotracker \
  --frontend-weights checkpoints/pi3x/model.safetensors \
  --config config/base_kitti360_cotracker.yaml \
  --calib config/intrinsics_kitti360.yaml \
  --dataset <dataset_path> \
  --imu_path <imu_path> \
  --stamp_path <stamp_path> \
  --start_from 0 \
  --end_at 20 \
  --result_path result_cotracker_smoke.txt \
  --cotracker-checkpoint <cotracker_checkpoint_or_omit_for_torchhub> \
  --no-viz
```

Or use the batch wrapper:

```bash
GPU_ID=0 SEQ=0005 START_FROM=0 END_AT=20 \
  COTRACKER_CHECKPOINT=<local_cotracker_checkpoint> \
  bash batch_kitti360_cotracker_vi.sh
```

The batch wrapper runs `tools/preflight_cotracker_runtime.py --require-cuda`
before `main.py`. By default it also requires `COTRACKER_CHECKPOINT` so the run
does not depend on GitHub/torchhub availability. Set `ALLOW_TORCHHUB=1` only if
network/cache access for `torch.hub.load("facebookresearch/co-tracker",
"cotracker3_offline")` has already been verified. Set `SKIP_PREFLIGHT=1` only
when the same checks have already been run in the active environment.

Expected:

- CoTracker and PI3X both load in the active environment.
- Tracking uses the `cotracker` frontend path.
- When the oldest tracked cohort drops below `lifecycle.oldest_visibility_threshold`,
  the lifecycle manager logs `PI3X window refresh`.
- PI3X window inference receives current SLAM `T_WC` poses as pose priors through
  `pi3x.use_pose_prior`.
- PI3X pointmaps are written back to the active keyframe window.
- PI3X camera poses are converted to sparse native
  `gtsam.BetweenFactorPose3` measurements.
- Dense pair factors are still added through `FactorGraph.add_factors()`.
- `result_cotracker_smoke.txt` and `graph_cotracker_<seq>.pkl` are created.

## Common Problems And How To Verify Them

### PI3X Import Fails

Symptom:

```text
ImportError: PI3X is not installed
```

Checks:

```bash
python -c "import pi3; print(pi3.__file__)"
python -c "from pi3.models.pi3x import Pi3X; print(Pi3X)"
pip show pi3
```

Likely causes:

- `thirdparty/Pi3` was not installed in the active environment.
- The upstream package path differs from `pi3.models.pi3x`.

If the upstream import path differs, update `load_pi3x()` in
`mast3r_fusion/pi3x_utils.py`.

### Checkpoint Not Found

Symptom:

```text
FileNotFoundError: PI3X checkpoint not found
```

Checks:

```bash
ls -lh checkpoints/pi3x/
python - <<'PY'
from pathlib import Path
p = Path('checkpoints/pi3x/model.safetensors')
print(p.resolve(), p.exists(), p.stat().st_size if p.exists() else None)
PY
```

Fix:

```bash
wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors -O checkpoints/pi3x/model.safetensors
```

### PI3X Output Keys Do Not Match

Symptom:

```text
KeyError: PI3X output is missing one of: ...
```

Checks:

Add a temporary print in `_call_pi3x()`:

```python
print(output.keys())
```

Expected keys are currently one of:

- points: `local_points`, `points_local`, or `pts3d`
- confidence: `conf`, `confidence`, or `confidences`
- poses: `camera_poses`, `poses`, or `extrinsics`

If upstream PI3X uses different names, update `_pair_output_to_maps()`.

### Shape Mismatch In Shared Frame Buffers

Symptom:

```text
RuntimeError: The expanded size of the tensor ...
```

Checks:

```bash
python - <<'PY'
from mast3r_fusion.frontend_model.pi3_adapter import PI3Adapter
print(PI3Adapter.name)
print(PI3Adapter.__mro__)
PY
```

Also log `feature_spec` in `main.py`:

```python
print(model.get_feature_spec())
```

Expected for PI3X:

```text
FeatureSpec(feat_dim=1024, patch_size=14)
```

This is required because PI3X stores encoder patch tokens in shared keyframes.

### Matching Produces Too Few Valid Points

Symptom:

```text
Skipped frame <id>
```

Checks:

- Print `valid_match_k.float().mean()` in `FrameTracker.track()`.
- Print median confidence:

```python
print(Qff.median(), Qkf.median(), Cff.median(), Ckf.median())
```

Likely causes:

- PI3X point maps are in a different coordinate convention.
- `camera_poses` convention is opposite of what `_pair_output_to_maps()` assumes.
- PI3X geometric correspondences are too sparse for the current thresholds.

Debug approach:

1. Run on two adjacent frames only.
2. Save or inspect `Xii[..., 2]`, `Xji[..., 2]`, and confidence histograms.
3. Temporarily relax `tracking.Q_conf`, `tracking.C_conf`, and
   `matching.dist_thresh`.
4. If valid matches increase only after relaxing distance, inspect the transform
   from `camera_poses`.

### Backend Cholesky Failure

Symptom:

```text
Cholesky failed <frame_id>
```

Checks:

- Confirm enough valid matches reach optimization.
- Inspect whether depth is positive:

```python
print((Xff[..., 2] > 0).float().mean(), (Xkf[..., 2] > 0).float().mean())
```

Likely causes:

- invalid PI3X scale or coordinate frame,
- low-confidence matches,
- too aggressive calibration constraints.

Debug approach:

1. Disable loop factors by keeping retrieval disabled for PI3X.
2. Test only local tracking on a short range.
3. Compare the same frame range with `--frontend-model mast3r`.

## Notes Before Committing

Current working tree also contains unrelated local edits:

- `.gitignore`
- `evaluation/evaluate_kitti360.py`
- `mast3r_fusion.md`

Do not include them in the PI3X adapter commit unless they are intentionally part
of the same change.
