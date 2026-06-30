# VINS-Fusion Sparse VIO Rewrite Plan

This note records the implementation boundary for rewriting
`sparse_vio_demo` against the VINS-Fusion codebase in
`thirdparty/VINS-Fusion`. The goal is not to tune the current demo, but to
replace its frontend/backend structure with a VINS-style pipeline.

The formula-level system design is maintained separately in
`docs/vins_like_sparse_vio_design.md`.

## Source Files Read

- `vins_estimator/src/featureTracker/feature_tracker.{h,cpp}`
- `vins_estimator/src/estimator/feature_manager.{h,cpp}`
- `vins_estimator/src/estimator/estimator.{h,cpp}`
- `vins_estimator/src/initial/initial_sfm.{h,cpp}`
- `vins_estimator/src/initial/initial_aligment.cpp`
- `vins_estimator/src/factor/projectionTwoFrameOneCamFactor.{h,cpp}`
- `vins_estimator/src/factor/marginalization_factor.h`
- `vins_estimator/src/estimator/parameters.cpp`
- `vins_estimator/src/rosNodeTest.cpp`

## VINS-Fusion Functional Boundary

VINS-Fusion is designed for mono+IMU, stereo+IMU, and stereo-only. Pure
monocular VO is not a first-class supported mode in the original estimator:
`changeSensorType()` rejects `!USE_IMU && !STEREO`. For this repo, `use_imu =
false` is only a frontend/visual-optimizer diagnostic mode unless another
metric scale source is added. It must not be used as a Sim3-backed success
criterion.

## Frontend Boundary

The VINS frontend only tracks features and publishes feature measurements. It
does not decide keyframes and does not run BA.

Required behavior:

- KLT tracking with optional estimator prediction.
- If prediction tracking succeeds on fewer than 10 tracks, rerun normal KLT.
- Reverse flow check with threshold `<= 0.5`.
- Border check with one-pixel margin.
- Preserve long-lived tracks first: sort by `track_cnt`, apply `MIN_DIST` mask,
  then detect new `goodFeaturesToTrack` points.
- Do not call `rejectWithF()` by default. It exists in code but is commented in
  `trackImage()`.
- Output per feature:
  `feature_id -> [(camera_id, [x, y, z, u, v, velocity_x, velocity_y])]`.
  `x/y/z` are normalized camera coordinates, `u/v` are pixel coordinates, and
  velocity is computed in normalized coordinates.

The current `VinsFrontend` should be replaced with a tracker whose API mirrors
this measurement packet. Keyframe flags should be removed from the frontend.

## FeatureManager Boundary

`FeatureManager` is the visual state container and keyframe decision layer.

Required behavior:

- Store `FeaturePerId(feature_id, start_frame)` and its ordered
  `FeaturePerFrame` observations.
- `addFeatureCheckParallax(frame_count, image, td)` both inserts measurements
  and returns whether the current frame is a keyframe.
- Keyframe decision follows VINS:
  - keyframe if `frame_count < 2`
  - keyframe if `last_track_num < 20`
  - keyframe if `long_track_num < 40`
  - keyframe if `new_feature_num > 0.5 * last_track_num`
  - otherwise compare average compensated parallax between the second-last and
    third-last frame against `MIN_PARALLAX`
- `getFeatureCount()` only counts tracks with `used_num >= 4`.
- Depth state is inverse depth anchored at the first observation frame.
- `triangulate()` initializes `estimated_depth`; invalid depth falls back to
  `INIT_DEPTH`.
- `removeBackShiftDepth()`, `removeBack()`, and `removeFront()` must be tied to
  sliding-window marginalization.

The current backend scattered parts of this logic across frontend and BA. The
rewrite should introduce a dedicated Python FeatureManager module first.

## Initialization Boundary

Mono+IMU initialization in VINS has these stages:

1. Fill a window of `WINDOW_SIZE + 1` frames.
2. Use `relativePose()` to find an older frame `l` with enough parallax against
   the newest frame. Required average parallax is approximately `30 / 460` in
   normalized coordinates.
3. Run `GlobalSFM.construct()`:
   - fix frame `l` and newest frame using relative pose,
   - solve intermediate and earlier frame poses by PnP,
   - triangulate features,
   - run visual-only BA.
4. Solve PnP for all image frames not in the keyframe headers.
5. Run `VisualIMUAlignment()`:
   - solve gyroscope bias,
   - linearly solve velocities, gravity, scale,
   - refine gravity on tangent basis,
   - reject bad gravity norm or negative scale.
6. Apply scale and camera-IMU extrinsic, align gravity, re-triangulate depths.

The current implementation used a much weaker initialization path. The rewrite
must implement GlobalSFM-equivalent initialization before relying on IMU scale.

For `use_imu=false`, only the visual GlobalSFM part is applicable. Scale remains
free for monocular input, so this mode cannot be considered a complete metric
SLAM method without an added scale source.

## Optimization Boundary

VINS optimizes a fixed window with these variables:

- Pose `para_Pose[i]` for each frame in the window.
- Speed/bias `para_SpeedBias[i]` when `USE_IMU`.
- Camera extrinsic `para_Ex_Pose`.
- Optional time offset `para_Td`.
- Inverse depth `para_Feature[k][0]` for each eligible feature.

Visual residuals are not `Point3` projection factors. The primary monocular
factor is `ProjectionTwoFrameOneCamFactor`:

- Anchor observation `pts_i` is backprojected by inverse depth in camera `i`.
- It is transformed through `pose_i`, camera extrinsic, world, then `pose_j`.
- Residual is normalized plane reprojection error against `pts_j`.
- Measurement velocity and time offset are part of the residual model.
- Huber loss with VINS sqrt information `FOCAL_LENGTH / 1.5`.

In GTSAM this requires a custom factor or an equivalent expression factor. A
`Point3` landmark graph is not equivalent and should not be the primary backend.

## Marginalization Boundary

When the window is full:

- `MARGIN_OLD`:
  - marginalize old pose/speed-bias,
  - include the first IMU factor,
  - include visual factors anchored at frame 0 and drop their inverse depths,
  - shift addresses and call `FeatureManager.removeBackShiftDepth()`.
- `MARGIN_SECOND_NEW`:
  - keep newest frame, remove the second-newest frame,
  - merge newest IMU buffer into the previous slot,
  - update the prior if the previous marginalization factor involved that pose,
  - call `FeatureManager.removeFront(frame_count)`.

The Python/GTSAM rewrite should preserve this semantics. If exact Schur
marginalization is not available at first, the implementation must isolate that
gap behind a `MarginalizationManager` instead of replacing it with ad-hoc loose
priors spread through the estimator.

## Rewrite File Boundary

Create a new implementation path instead of patching the current demo backend:

- `sparse_vio_demo/sparse_vio/vins_like/tracker.py`
- `sparse_vio_demo/sparse_vio/vins_like/feature_manager.py`
- `sparse_vio_demo/sparse_vio/vins_like/initializer.py`
- `sparse_vio_demo/sparse_vio/vins_like/factors.py`
- `sparse_vio_demo/sparse_vio/vins_like/estimator.py`
- `sparse_vio_demo/sparse_vio/vins_like/marginalization.py`

The existing XFeat frontend can later be adapted to emit the same measurement
packet, but the first rewrite should target the VINS KLT frontend only.

`main.py` should instantiate the new estimator path explicitly. The old
`SparseBackend` and old `VinsFrontend` should remain only as legacy references
until the new path is verified.

## Verification Boundary

Validation should be staged:

1. Compile/import checks for the new modules.
2. Tracker-only run with Foxglove overlay:
   - stable track count,
   - long-lived tracks preserved,
   - feature ids consistent.
3. Visual-only initialization:
   - GlobalSFM succeeds on a full window,
   - depths are positive after re-triangulation.
4. Mono+IMU initialization:
   - gyro bias finite,
   - gravity norm close to config,
   - positive scale,
   - repropagation uses solved gyro bias.
5. Sliding-window run:
   - no window-size growth,
   - old and second-new marginalization paths both exercised,
   - Foxglove publishes image, sparse point cloud, camera/body/world TF.
6. Metric evaluation:
   - mono+IMU must be evaluated with SE3 alignment only,
   - no `--correct_scale`/Sim3 result should be used as acceptance evidence.

## Current Implementation Status

Implemented in `sparse_vio_demo/sparse_vio/vins_like/`:

- VINS-style KLT tracker and feature packet.
- VINS-style FeatureManager and parallax keyframe decision.
- GlobalSFM-style monocular visual initialization with GTSAM-native full BA:
  `Pose3` camera states, `Point3` landmarks, `GenericProjectionFactorCal3_S2`
  normalized observations, a fixed `l` pose prior, and a newest-frame
  translation prior for scale gauge. Avoid SciPy handwritten optimization in
  this initialization path.
- GTSAM inverse-depth visual factors, batched per feature track.
- Mono+IMU visual-inertial alignment using the VINS scale/gravity/bias solve
  order: gyro bias solve, biased re-preintegration, linear velocity/gravity/
  scale solve, gravity refinement, and yaw-aligned gravity rotation.
- Online IMU factors connected through body pose, velocity, and bias variables.
- VINS-style average reprojection-error outlier rejection after nonlinear
  optimization and before sliding-window update.
- Fixed-lag marginalization prior using GTSAM linearization plus VINS-style
  Hessian Schur complement with eigenvalue pseudo-inverse thresholding. The
  resulting dense prior is rekeyed into the next sliding window as a custom
  `r + J * dx` factor. This avoids GTSAM Cholesky failures on weakly constrained
  inverse-depth variables.
- `run_vio_eval_kitti360.sh` defaults to the `vins_like` frontend and runs the
  full configured sequence unless `START`/`END` are explicitly provided.

Short ranges are treated only as basic runtime checks, not valid experiment
evidence. Effective evaluation should use at least 1200 frames.

Basic checks with SE3-only KITTI-360 sequence 0000:

- `START=0 END=40 FRONTEND=vins_like USE_IMU=1`
  - SE3 APE RMSE: `0.036034 m`
  - SE3 RPE RMSE: `0.023491 m`
- `START=0 END=200 FRONTEND=vins_like USE_IMU=1`
  - SE3 APE RMSE: `0.361308 m`
  - SE3 RPE RMSE: `0.045895 m`
  - The previous `Indeterminant linear system near variable d6865`
    marginalization crash is gone after replacing Cholesky elimination with
    VINS-style pseudo-inverse marginalization.
- `START=0 END=100 FRONTEND=vins_like USE_IMU=0`
  - The visual-only path initializes, slides the window, and outputs a
    trajectory and sparse points.
  - SE3 metric errors are large because pure monocular input has no metric scale
    source. This is a diagnostic mode only and is not counted as a VIO result.

Current effective SE3-only KITTI-360 sequence 0000 result:

- `START=0 END=1200 FRONTEND=vins_like USE_IMU=1`
  - SE3 APE RMSE: `20.821016 m`
  - SE3 RPE RMSE: `0.151033 m`
  - The system completes 1200 frames, but long-range drift is still too high.
    The rewrite is therefore not complete.

Remaining work before calling the rewrite complete:

- Continue improving 1200+ frame drift; short 40/100/200 frame checks are only
  smoke tests.
- Compare marginalization factor selection and drop sets against VINS-Fusion
  line by line for `MARGIN_OLD` and `MARGIN_SECOND_NEW`.
- Compare fixed-lag prior contents and gauge handling against VINS-Fusion,
  especially interaction between the dense marginal prior, body-camera pose
  coupling, and first-state gauge priors.
- Add Foxglove topics for the new `vins_like` debug state if the existing
  publisher does not expose enough internal feature/depth/marginalization data.
- Decide whether pure monocular `use_imu=false` remains a diagnostic mode only
  or receives an explicit metric scale source.
