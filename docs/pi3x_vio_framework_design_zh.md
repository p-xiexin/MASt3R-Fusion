# PI3X 兼容单目 VIO 新框架方案设计

本文是算法框架讨论稿，目标是在当前 MASt3R-Fusion 代码基础上，设计一种更适合 PI3X 这类多视图前馈几何模型的单目 VIO 架构。当前仓库已实现一个最小侵入原型，完整 VIO 后端重构仍属于后续工作。

> 参考 EC3R，用 XFeat/光流+ local sparse map + PnP-RANSAC 作为轻量前端，攒够一小批关键帧，丢给 pi3x做 feed-forward reconstruction输出 depth、confidence、camera parameters，反投影成 dense point cloud / local submap，用 shared old keyframes 的点云对应做 weighted Sim(3) registration，Sim3 约束进入 pose graph，mapping 结果再反过来修正 local sparse map 里的 3D 点，帮助前端 tracking 不漂

## 0. 当前原型实现状态

当前代码已经落地的部分：

- `LightTracker` 使用 XFeat keypoints、LK optical flow、local sparse map 和 PnP-RANSAC，给前端提供相对位姿先验。
- `light_tracking.pose_only=true` 时，普通帧不再逐帧调用 PI3X pair inference，只用轻量前端 tracking；关键帧再触发 PI3X。
- `pi3x.window_size` 控制攒够多少关键帧后调用 PI3X window inference。当前默认是 4。
- PI3X window 模式下，后端视觉优化只在关键帧数量达到 `window_size` 后入队；未攒够窗口时不提前构造 pairwise 后端因子。
- PI3X window inference 输出 pointmap、confidence 和 camera poses，刷新最近关键帧的 dense point cloud / local submap。
- 后端 local factors 仍复用现有 MASt3R-Fusion `FactorGraph` / `AlignCoreCalib` / GTSAM HessianFactor 路径，使用 pointmap correspondence 和 confidence 做 weighted Sim(3) BA。
- 同一 PI3X window 内的 keyframe factor 会复用 cached pointmap 和 PI3X camera pose；跨窗口边当前仍通过 PI3X pair inference 重新生成同一局部坐标系下的几何对应，避免混用不同 PI3X window 坐标系。
- 后端优化后会从 shared keyframes 刷新最近 local sparse map，使轻量前端看到更新后的 keyframe pose 和点云。

当前仍未落地的部分：

- 独立的 metric `SE(3)+v+bias` PI3X-VIO 后端。
- PI3X window 输出的显式 relative pose factor、cycle consistency 验收和视觉-IMU disagreement gating。
- 完整质量驱动滑窗、退化模式和回环分层。
- 服务器 GPU 上的短序列闭环验证和阈值调参。

## 1. 背景判断

PI3X、DUSt3R、MASt3R、VGGT 这类前馈视觉几何模型的共同点，是直接从一组图像预测点图、相机位姿、深度、置信度或轨迹等几何量。它们与传统 VIO 前端的稀疏角点/KLT/描述子匹配有明显差异：

- 传统 VIO 的优势主要来自工程成熟的后端：初始化、IMU 预积分、bias 重传播、滑窗边缘化、坏观测删除和失败恢复。
- 学习型 VO/SLAM 的优势主要来自更强的视觉先验：弱纹理、大视角、低重复纹理区可能比传统特征更稳。
- 多视图前馈模型的优势不只是 pairwise matching，而是可以在一个窗口内联合解释多帧几何。

因此，PI3X 不应只被设计成 MASt3R pair matcher 的替代品。更合理的方向是：

```text
multi-view feed-forward frontend + tightly-coupled VIO backend
```

也就是让 PI3X 生成强视觉几何 prior，由 VIO 后端负责 metric scale、时间连续性、bias、速度、重力和失败恢复。

## 2. 相关方法脉络

### 2.1 成熟传统 VIO

代表系统包括 VINS-Mono/VINS-Fusion、OKVIS、OpenVINS、Basalt、Kimera/Kimera2。它们的核心经验是：

- 状态包含 `R, p, v, bg, ba`，而不是只有相机 pose。
- IMU 预积分贯穿初始化、滑窗优化和状态传播。
- bias 更新后需要 repropagation，使旧预积分量与最新 bias 保持一致。
- 视觉残差需要优化前降权、优化后剔除。
- 边缘化前要清理坏观测，否则错误会固化进 prior。
- 低视差、纯旋转、弱 IMU 激励时，要限制不可观自由度。
- 回环和实时 VIO 应分层处理，不能让未验证的回环边污染局部滑窗。

这类系统给 PI3X-VIO 的启发是：强前端不能替代后端健康管理。

### 2.2 学习型 VO/SLAM

DROID-SLAM 和 DPVO 代表了另一条路线。DROID-SLAM 通过 dense correspondence、recurrent update 和 dense bundle adjustment 联合优化 pose/depth。DPVO 则证明 sparse patch + differentiable BA 可以在效率和精度之间取得更好的工程平衡。

这类方法的启发是：

- 学习前端输出不必直接成为最终轨迹，可以作为迭代优化的强初值或观测。
- dense 信息量大，但必须有稀疏化、置信度建模和 outlier 管理。
- 实时系统不一定要保存全部 dense 点，关键是保存可诊断、可删除、可边缘化的观测。

### 2.3 前馈 3D prior

DUSt3R 将双目/多视图几何统一为 pointmap regression。MASt3R 在此基础上增强匹配能力。MASt3R-SLAM 使用 pointmap matching、tracking、local fusion、graph construction、loop closure 和二阶全局优化构建 dense SLAM。VGGT 直接从一张到多张图预测 camera、point map、depth 和 3D tracks。PI3X/π³ 进一步强调 permutation-equivariant 多视图几何学习，适合窗口级输入。

这类方法给新框架的直接启发是：

- 前端接口应从 pairwise 扩展到 window-level。
- 模型输出的 camera pose、point map、confidence 应作为观测候选，而不是无条件真值。
- 多视图 cycle consistency、尺度一致性、IMU 一致性应参与观测验收。

## 3. 当前 PI3X 适配的关键限制

当前本地适配更像是把 PI3X 输出压入现有 MASt3R-Fusion pairwise 接口，主要限制包括：

- PI3X 不暴露 MASt3R 风格 dense descriptor。
- PI3X 不直接输出 `idx_i2j`、`idx_j2i`、`valid_match` 等显式 pair matching 结果。
- 当前匹配由 PI3X local point maps 和 camera poses 推导，再走投影几何近似。
- PI3X confidence 与 MASt3R descriptor confidence 不是同一尺度。
- MASt3R retrieval database 不能直接复用到 PI3X 前端。

因此，如果后续只把 PI3X 当作 pair matcher 替换 MASt3R，容易损失 PI3X 的多视图优势，同时把不适合的观测形态强塞给旧后端。

## 4. 推荐总体框架

推荐框架命名可以暂定为 `PI3X-VIO` 或 `FeedForward-VIO`。

```text
image stream + IMU stream
        |
IMU propagation / motion prediction
        |
MultiViewWindowManager
        |
PI3XWindowInferencer
        |
VisualObservationBuilder
        |
SlidingWindowVIOBackend
        |
local odometry + optional loop/global graph
```

系统分为五层：

1. **运动预测层**：使用 IMU propagation 给相机 pose 初值、窗口选择、退化判断和前端 sanity check 提供先验。
2. **多视图前端层**：PI3X 接收 3 到 8 帧窗口，输出 per-view local point maps、camera poses、confidence 和可选 tracks。
3. **观测构造层**：把模型输出转成可诊断、可删除、可边缘化的视觉因子。
4. **紧耦合 VIO 后端层**：维护 metric `SE(3)` pose、velocity、IMU bias 和重力方向。
5. **回环/全局层**：只接收强几何验证后的闭环或重定位约束，不直接污染实时滑窗。

## 5. 前端设计

### 5.1 Window 选择

PI3X 的优势在于多视图窗口，而不是只跑相邻 pair。建议窗口大小初期取 3 到 5 帧，稳定后扩展到 5 到 8 帧。

窗口选择指标：

- IMU 预测的旋转量和平移量。
- 图像间视差。
- 最近局部 tracking 质量。
- pointmap confidence 分布。
- 运动退化状态，例如纯旋转、低平移、弱 IMU 激励。

策略上：

- 正常运动：保留当前帧、最近关键帧、局部共视关键帧。
- 低视差/纯旋转：减少新关键帧固化，更多依赖 IMU 姿态传播，进入 `HOLD_SCALE`。
- 大运动/快速旋转：增加 IMU prior 权重，允许 PI3X 重新估计局部几何，但后端严格验收。
- 重定位：构造当前帧 + 候选历史帧窗口，输出只作为 candidate，必须经过几何验证。

### 5.2 PI3X 输出适配

PI3X 前端输出建议抽象为：

```text
WindowInferenceResult:
  frames
  camera_poses_model
  local_pointmaps
  confidences
  optional_features
  optional_tracks
  metadata
```

其中 `camera_poses_model` 不应直接覆盖 VIO pose，而应进入 `VisualObservationBuilder` 形成候选视觉约束。

### 5.3 观测类型

建议构造四类视觉观测。

**Relative pose factor**

从 PI3X 多视图 pose 得到窗口内 `T_ij`，再和 IMU prediction、当前 VIO 估计比较。通过验收后，作为相对位姿因子进入后端。

需要保存：

```text
i, j
T_ij
pose_confidence
scale_confidence
cycle_error
imu_disagreement
```

**Pointmap consistency factor**

将不同帧 local point map 转到同一参考系，检查几何一致性。高置信区域形成点图一致性残差，低置信或高动态区域不进入后端。

残差可以先用稀疏采样版本实现：

```text
r = T_WCi * X_i(u) - T_WCj * X_j(v)
```

**Bearing-depth pseudo factor**

从高置信 pointmap 中采样稳定点，将其转成 bearing/depth 或 inverse-depth 观测。该形式更接近传统 VIO 的 landmark residual，利于后验残差检查和边缘化。

**Track factor**

如果 PI3X 或后续 VGGT 类前端能输出 2D/3D tracks，应优先使用 track factor。相比 dense projection matching，track factor 更容易做生命周期管理、outlier 删除和滑窗边缘化。

## 6. 后端设计

### 6.1 状态变量

后端应保持传统 metric VIO 状态：

```text
x_i = {R_i, p_i, v_i, b_gi, b_ai}
```

可选状态：

```text
gravity direction
camera-IMU extrinsic
camera-IMU time offset
short-lived visual scale latent
```

不建议让每帧长期自由维护 `Sim(3)` scale。PI3X/MASt3R 的尺度不确定性可以作为视觉测量内部变量或短期 latent，但最终轨迹应由 IMU、重力和初始化约束到 metric `SE(3)`。

### 6.2 因子图

基础因子：

```text
IMU preintegration factor
bias random walk factor
pose/velocity/bias prior
visual relative pose factor
pointmap consistency factor
bearing-depth factor
optional GNSS factor
optional loop factor
```

视觉因子必须满足三个条件：

- 可计算后验 residual。
- 可按边、点簇或 track 删除。
- 可在边缘化前完成健康检查。

### 6.3 初始化

初始化不能只依赖视觉窗口成功。建议初始化流程：

1. 收集短窗口图像和 IMU。
2. PI3X 估计局部 pose/pointmaps。
3. IMU 预积分估计重力方向、速度、bias 初值。
4. 对齐视觉局部轨迹和 IMU 预积分，估计 metric scale。
5. 检查视差、IMU 激励、scale 稳定性、gravity 一致性。
6. 通过后进入正常 VIO；失败则继续纯视觉/IMU propagation 等待更好激励。

初始化验收指标：

```text
parallax
gyro excitation
acc excitation
scale variance
gravity direction residual
PI3X cycle error
visual-IMU rotation disagreement
```

### 6.4 滑窗与边缘化

滑窗策略应从固定窗口改为质量驱动：

- 保留高视差、高连接度、高置信关键帧。
- 对纯旋转帧，允许作为短期跟踪帧，但避免过早固化为强关键帧。
- 边缘化前执行视觉因子 residual 检查。
- 对冲突边进行降权、switchable constraint 或直接删除。
- bias 更新后对相关 IMU preintegration repropagation。

## 7. 退化与健康管理

建议引入运行模式：

```text
NORMAL
HOLD_SCALE
DEGRADED
RELOCALIZING
LOST
```

核心健康指标：

```text
valid dense match ratio
pointmap confidence percentile
edge cost percentile
scale jump magnitude
rotation-baseline ratio
visual-IMU rotation residual
visual-IMU translation residual
PI3X pose cycle error
loop edge disagreement
solver convergence status
bias norm and bias update magnitude
```

模式切换示例：

- `rotation-baseline ratio` 高、平移弱：进入 `HOLD_SCALE`，固定或强正则化尺度。
- 视觉-IMU rotation residual 大：拒绝当前视觉边，尝试重跑前端或进入 `DEGRADED`。
- pointmap confidence 整体低：降低视觉权重，依赖 IMU 短期传播。
- 回环候选与局部 VIO 强冲突：不进入实时滑窗，只记录为待全局验证 candidate。

## 8. 回环与重定位

PI3X 前端下不建议直接复用 MASt3R retrieval database，除非后续建立 PI3X-compatible global descriptor。

推荐分层：

```text
Realtime VIO graph:
  consecutive/local visual-inertial factors only

Loop/relocalization graph:
  retrieval candidates
  PI3X multi-view verification
  geometric consistency
  switchable loop factors
```

回环边进入全局图前至少需要：

- 外观检索通过。
- PI3X 多视图几何一致。
- Sim(3)/SE(3) RANSAC 或 PnP 验证通过。
- 与当前 VIO 局部轨迹不产生极端冲突。
- 通过 switchable constraint 或 robust kernel 接入。

## 9. 推荐实现路线

### 阶段 1：观测诊断，不改主优化模型

目标是先量化 PI3X 输出质量。

- 增加 window-level PI3X 离线评估脚本。
- 记录 PI3X pose 与现有 VIO/GT/IMU prediction 的差异。
- 记录 pointmap confidence、cycle consistency、scale jump。
- 不急于接入后端优化。

产物：

```text
PI3X window diagnostic logs
visual-IMU disagreement plots
failure case taxonomy
```

### 阶段 2：PI3X relative pose factor

目标是用最小后端改动验证多视图 pose prior 的价值。

- 从 PI3X window 输出构造 `T_ij`。
- 加入边级 residual、置信度、验收阈值。
- 只接入相邻关键帧或短窗口内局部边。
- 低视差时固定尺度或强正则化尺度。

### 阶段 3：点图/track 因子

目标是让 PI3X 的 dense geometry 真正进入后端。

- 高置信 pointmap 稀疏采样。
- 构造 pointmap consistency residual。
- 后验 residual 删除坏点簇。
- 若可获得 tracks，则用 track lifecycle 替代纯 dense projection matching。

### 阶段 4：完整 VIO 后端重构

目标是形成独立的 PI3X-VIO 后端。

- 完整 metric `SE(3) + v + bias` 状态。
- IMU 初始化和 repropagation。
- 质量驱动关键帧管理。
- 模式切换和失败恢复。
- 回环/局部 VIO 分层。

## 10. 关键设计原则

1. PI3X 是强视觉 prior，不是最终轨迹真值。
2. 多视图窗口优先；跨窗口 pairwise 只用于重新生成必要的同坐标系几何对应。
3. 后端状态保持 metric VIO，不让长期 per-frame scale 自由漂移。
4. 所有视觉约束必须可诊断、可删除、可边缘化。
5. 低视差、纯旋转和弱 IMU 激励时，主动限制不可观自由度。
6. 回环和实时 VIO 分层，未经验证的全局约束不能进入局部滑窗。
7. 第一版先求稳轨迹，再考虑 dense map 质量。

## 11. 参考资料

- VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator, https://arxiv.org/abs/1708.03852
- DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras, https://arxiv.org/abs/2108.10869
- DPVO: Deep Patch Visual Odometry, https://arxiv.org/abs/2208.04726
- DUSt3R: Geometric 3D Vision Made Easy, https://arxiv.org/abs/2312.14132
- MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors, https://arxiv.org/abs/2412.12392
- VGGT: Visual Geometry Grounded Transformer, https://arxiv.org/abs/2503.11651
- PI3 / PI3X: Scalable Permutation-Equivariant Visual Geometry Learning, https://arxiv.org/abs/2507.13347
- MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM, https://arxiv.org/abs/2509.20757
