# VINS-Fusion Style Sparse VIO Design

## Abstract

This document specifies the `sparse_vio_demo/sparse_vio/vins_like` rewrite.
The implementation goal is a VINS-Fusion-style sparse monocular visual-inertial
odometry pipeline implemented as an isolated demo path, with GTSAM-native
optimization where practical and without Sim3 scale correction in evaluation.

The system is intentionally separated from MASt3R/PI3X dense reconstruction.
The sparse frontend and backend estimate a metric SE(3) trajectory from
monocular images plus IMU. Pure monocular VO is retained only as a diagnostic
mode because monocular input without IMU has no metric scale source.

## Functional Boundary

VINS-Fusion supports monocular+IMU, stereo+IMU, and stereo-only. The monocular
case requires IMU for metric scale. Therefore:

- `use_imu=true` is the target VIO mode.
- `use_imu=false` is a visual optimizer diagnostic, not a complete metric SLAM
  method.
- Evaluation must use SE(3) alignment only. Sim3 alignment or
  `--correct_scale` is not valid acceptance evidence.
- Effective experiments must run at least 1200 frames. Short ranges such as
  40, 100, or 200 frames are smoke checks only.

## Coordinate Frames

The main frames are:

- `w`: world frame.
- `b_i`: IMU/body frame at image time `i`.
- `c_i`: camera frame at image time `i`.

The calibrated body-to-camera transform is fixed:

```math
T_{bc} =
\begin{bmatrix}
R_{bc} & p_{bc} \\
0 & 1
\end{bmatrix},
\qquad
T_{wc_i} = T_{wb_i} T_{bc}.
```

The current GTSAM implementation keeps both a camera pose variable `x_i` and a
body pose variable `z_i`, linked by a tight `BetweenFactorPose3(z_i, x_i,
T_bc)`. This was kept because the attempted single-body-pose graph matched the
VINS parameterization more directly but degraded the 1200-frame result under
the current marginalization/gauge implementation. The design target remains a
single body pose parameter block equivalent to VINS:

```math
\mathcal{X}_i =
\{T_{wb_i}, v_i, b^a_i, b^g_i\}.
```

The present dual-pose implementation should be treated as a GTSAM engineering
bridge, not the final mathematical ideal.

## Frontend Design

The frontend mirrors VINS-Fusion's `feature_tracker`.

For each image `I_i`, KLT tracks previous features:

```math
u^k_{i-1} \rightarrow u^k_i.
```

Tracking uses:

- pyramidal LK optical flow,
- reverse flow consistency,
- border rejection,
- long-track-first masking,
- new Shi-Tomasi features to refill the budget.

The reverse flow check accepts a track when:

```math
\| \hat{u}^{k}_{i-1} - u^{k}_{i-1} \|_2 \le 0.5.
```

Feature selection sorts by track age and applies a minimum-distance mask. This
preserves long-lived tracks, matching VINS-Fusion's `setMask()` behavior.

The frontend output is only a measurement packet:

```math
\text{feature\_id}
\mapsto
\left[
\left(
\text{camera\_id},
\begin{bmatrix}
x & y & z & u & v & \dot{x} & \dot{y}
\end{bmatrix}^{T}
\right)
\right].
```

Here `(x,y,z)` is the normalized bearing, `(u,v)` is the pixel coordinate, and
`(\dot{x},\dot{y})` is normalized-plane velocity. The frontend does not decide
BA keyframes and does not optimize poses.

## Feature Manager

The `FeatureManager` owns visual tracks and keyframe decisions. Each feature is:

```math
\mathcal{F}_k =
\{ id_k, s_k, \rho_k, (f_{s_k}, f_{s_k+1}, \ldots, f_{e_k}) \},
```

where `s_k` is the anchor frame and `\rho_k = 1 / d_k` is inverse depth.

The keyframe rule follows VINS-Fusion:

```math
\text{keyframe} =
\begin{cases}
\text{true}, & i < 2,\\
\text{true}, & N_\text{tracked} < 20,\\
\text{true}, & N_\text{long} < 40,\\
\text{true}, & N_\text{new} > 0.5 N_\text{tracked},\\
\bar{p} \ge \text{MIN\_PARALLAX}, & \text{otherwise}.
\end{cases}
```

The average parallax is computed in normalized coordinates between the
second-last and third-last frame:

```math
\bar{p} =
\frac{1}{|\mathcal{S}|}
\sum_{k \in \mathcal{S}}
\left\|
\pi(f^k_{i-2}) - \pi(f^k_{i-1})
\right\|_2.
```

Only features with at least four observations enter the backend:

```math
|\mathcal{F}_k| \ge 4.
```

## Initialization

Initialization follows VINS-Fusion's monocular visual structure plus
visual-inertial alignment.

### Relative Pose

Given correspondences between an older frame `l` and the newest frame `n`, the
relative pose is accepted when:

```math
N_\text{corres} > 20,
\qquad
\bar{p} f > 30.
```

OpenCV five-point recovery returns camera-frame motion. To match VINS-Fusion,
the stored relative transform is:

```math
R_{l n} = R^\top,
\qquad
t_{l n} = -R^\top t.
```

### GlobalSFM

The initialization fixes frame `l` and the newest frame translation gauge,
triangulates seed points, then solves intermediate poses by PnP.

Triangulation solves:

```math
A X = 0,
```

where each observation contributes:

```math
u P_3 - P_1,
\qquad
v P_3 - P_2.
```

The full visual BA is implemented with GTSAM-native factors, following the
style used in the official GTSAM examples:

```math
\min_{\{T_{wc_i}\}, \{P_k\}}
\sum_{(i,k)\in\mathcal{O}}
\left\|
\pi(T_{cw_i} P_k) - z_{ik}
\right\|^2_{\Sigma^{-1}}.
```

Implementation mapping:

- camera pose: `gtsam.Pose3`,
- landmark: `gtsam.Point3`,
- projection: `gtsam.GenericProjectionFactorCal3_S2`,
- fixed `l` pose: `gtsam.PriorFactorPose3`,
- newest translation gauge: `gtsam.PoseTranslationPrior3D`.

No SciPy `least_squares` optimization is used in this path.

### Visual-Inertial Alignment

After visual SFM, the system solves gyro bias, re-preintegrates IMU, then solves
velocity, gravity, and scale.

Gyro bias solves:

```math
\min_{\delta b^g}
\sum_i
\left\|
J^g_i \delta b^g -
2 \, \text{vec}\left(
\Delta q_i^{-1}
\otimes
(R_i^\top R_{i+1})
\right)
\right\|^2.
```

Then each IMU interval is re-integrated with the estimated gyro bias.

Linear alignment solves:

```math
\Delta p_{i,i+1}
=
R_i^\top
\left(
s(p_{i+1}-p_i)
- v_i \Delta t
- \frac{1}{2} g \Delta t^2
\right),
```

```math
\Delta v_{i,i+1}
=
R_i^\top
\left(
v_{i+1} - v_i - g \Delta t
\right).
```

The unknown vector is:

```math
x =
\begin{bmatrix}
v_0^\top & \cdots & v_N^\top & g^\top & s
\end{bmatrix}^{\top}.
```

Gravity is refined on the tangent plane:

```math
g \leftarrow
\frac{g + B(g)\delta g}{\|g + B(g)\delta g\|} \|G\|.
```

Finally gravity is aligned to the world vertical and yaw is normalized as in
VINS-Fusion.

## Backend Optimization

For a fixed window, the target VINS state is:

```math
\mathcal{X} =
\left\{
T_{wb_i}, v_i, b^a_i, b^g_i
\right\}_{i=0}^{N}
\cup
\left\{\rho_k\right\}_{k \in \mathcal{F}}.
```

The current GTSAM bridge additionally uses camera pose variables:

```math
x_i = T_{wc_i},
\qquad
z_i = T_{wb_i},
\qquad
x_i \approx z_i T_{bc}.
```

The optimized objective is:

```math
\min
\left(
\|r_p\|^2
+ \sum_i \|r^\text{imu}_i\|^2
+ \sum_k \sum_{j\in\mathcal{O}_k} \|r^\text{cam}_{k,j}\|^2
+ \|r_m\|^2
\right).
```

### Visual Residual

For a feature anchored in frame `i`, inverse depth `\rho`, and observed in
frame `j`:

```math
p_{c_i} = \frac{\bar{u}_i}{\rho},
```

```math
p_w = R_{wc_i} p_{c_i} + p_{wc_i},
```

```math
p_{c_j} =
R_{wc_j}^{\top}
(p_w - p_{wc_j}),
```

```math
r^\text{cam}_{ij}
=
\begin{bmatrix}
p_x / p_z \\
p_y / p_z
\end{bmatrix}
-
\begin{bmatrix}
u_j \\
v_j
\end{bmatrix}.
```

The factor uses VINS sqrt information:

```math
\sqrt{\Lambda} = \frac{f}{1.5} I_2,
```

and Huber loss with threshold `1.0`.

The implementation batches all target observations of one anchored feature into
one GTSAM `CustomFactor`, which reduces Python factor overhead while preserving
the same residual block semantics.

### IMU Residual

The IMU factor uses GTSAM `PreintegratedCombinedMeasurements` and
`CombinedImuFactor`. For interval `[i,i+1]`:

```math
r^\text{imu}_i =
r(
T_{wb_i}, v_i, b_i,
T_{wb_{i+1}}, v_{i+1}, b_{i+1},
\hat{\alpha}_{i,i+1},
\hat{\beta}_{i,i+1},
\hat{\gamma}_{i,i+1}
).
```

Bias random walk is included by the combined IMU factor.

## Marginalization

VINS-Fusion uses fixed-lag marginalization. When the window is full, two cases
exist.

### MARGIN_OLD

Drop the oldest state and inverse depths anchored at the oldest frame:

```math
\mathcal{D} =
\{x_0, v_0, b_0\}
\cup
\{\rho_k \mid s_k = 0\}.
```

The marginalization graph includes:

- previous marginal prior if it touches dropped variables,
- first IMU factor,
- visual factors anchored at frame `0`.

### MARGIN_SECOND_NEW

Drop the second-newest frame while keeping the newest frame:

```math
\mathcal{D} =
\{x_{N-1}, v_{N-1}, b_{N-1}\}.
```

The prior is updated only if the previous prior includes the dropped block.
Features remove the corresponding observation via `removeFront()`.

### Schur Complement Prior

Let the linearized normal equation be partitioned into dropped variables `m`
and retained variables `r`:

```math
\begin{bmatrix}
H_{mm} & H_{mr} \\
H_{rm} & H_{rr}
\end{bmatrix}
\begin{bmatrix}
\delta x_m \\
\delta x_r
\end{bmatrix}
=
\begin{bmatrix}
g_m \\
g_r
\end{bmatrix}.
```

The retained prior is:

```math
H' = H_{rr} - H_{rm} H_{mm}^{+} H_{mr},
```

```math
g' = g_r - H_{rm} H_{mm}^{+} g_m.
```

The pseudo-inverse uses eigenvalue thresholding as in VINS:

```math
H_{mm}^{+}
=
V
\operatorname{diag}
\left(
\lambda_i^{-1} \mathbf{1}_{\lambda_i > \epsilon}
\right)
V^\top.
```

The final dense prior is stored as:

```math
r_m(\delta x_r) =
r_0 + J \delta x_r,
```

where:

```math
J = \sqrt{\Lambda} V^\top,
\qquad
r_0 = -\Lambda^{-1/2} V^\top g'.
```

This avoids GTSAM Cholesky failures on weakly constrained inverse-depth
variables.

## Outlier Rejection

After nonlinear optimization, VINS rejects visual tracks by average normalized
reprojection error:

```math
\bar{e}_k =
\frac{1}{|\mathcal{O}_k|-1}
\sum_{j\neq s_k}
\left\|
\pi(p_{c_j}) - z_{k,j}
\right\|_2.
```

The track is removed if:

```math
f \bar{e}_k > 3.
```

This happens before sliding-window update. Features with negative solved depth
are removed after sliding.

## GTSAM-Native Implementation Policy

The rewrite avoids SciPy handwritten nonlinear optimization in the core path.
Native GTSAM components are used for:

- GlobalSFM visual BA,
- IMU preintegration and combined IMU factors,
- pose, velocity, bias, and inverse-depth fixed-lag optimization,
- linearization for marginalization.

Custom GTSAM factors are used only where VINS-specific residuals have no direct
off-the-shelf equivalent:

- inverse-depth anchored monocular projection,
- dense marginalization prior.

The GTSAM examples repository is used as a style reference for native factor
graph construction:

- https://github.com/gtbook/gtsam-examples/tree/main

## Visualization Contract

Foxglove debug output should expose:

- tracking image with VINS-style feature overlay,
- sparse 3D points,
- camera pose,
- body pose,
- world/body/camera TF,
- trajectory path,
- optional counters for active tracks, outliers, visual residual count, and
  marginalization mode.

The visualization is diagnostic and must not change estimator state.

## Evaluation Protocol

Commands:

```bash
USE_IMU=1 START=0 END=1200 FRONTEND=vins_like \
  OUT_DIR=sparse_vio_demo/output/vins_like_eval_1200 \
  bash sparse_vio_demo/run_vio_eval_kitti360.sh
```

The script evaluates:

```bash
evo_ape tum GT EST --align
evo_rpe tum GT EST --align --delta 1 --delta_unit f
```

No Sim3 or scale correction is allowed.

## Current Evidence

The current GTSAM-native GlobalSFM BA and VINS-style outlier rejection run
successfully.

Basic checks:

- 40 frames: SE3 APE RMSE `0.036034 m`, RPE RMSE `0.023491 m`.
- 200 frames: SE3 APE RMSE `0.361308 m`, RPE RMSE `0.045895 m`.

Effective check:

- 1200 frames: SE3 APE RMSE `20.821016 m`, RPE RMSE `0.151033 m`.

A single-body-pose GTSAM graph experiment was also tested:

- 1200 frames: SE3 APE RMSE `33.222133 m`, RPE RMSE `0.170472 m`.

This experiment was not kept in the active estimator path because it degraded
the effective result, despite being closer to the ideal VINS parameterization.
The likely reason is incomplete equivalence in marginal prior/gauge treatment.

## Remaining Technical Risks

The rewrite is not complete. The dominant remaining issue is long-range drift
over 1200+ frames. The next work should focus on:

- exact VINS marginalization residual-block selection,
- prior gauge handling in the GTSAM bridge,
- whether the dual camera/body pose bridge should be replaced only after a
  fully equivalent marginal prior is available,
- IMU factor weighting and first-state priors,
- parity of feature removal timing with VINS-Fusion.

