# MASt3R-Fusion 与 VINS-Fusion 后端鲁棒性对比调研报告

## 摘要

本文围绕一个工程现象展开：在当前本地工作树中，MASt3R-Fusion 的前端 tracking 可视化结果较稳定，甚至在若干场景中观感优于 VINS-Fusion；但系统级轨迹鲁棒性显著弱于 VINS-Fusion。通过阅读当前 MASt3R-Fusion 后端实现，并与 VINS-Fusion/VINS-Mono 的公开论文、官方仓库和源码结构进行对比，可以得到一个明确判断：

> 当前现象更可能由后端鲁棒估计、滑窗管理、坏因子抑制、视觉-惯性初始化与失败恢复机制不足引起，而不是由前端 tracking 能力不足单独导致。

更精确地说，MASt3R-Fusion 的强前端为后端提供了密集匹配、点图和置信度；但当前后端将这些信息主要压缩成 Sim(3) Hessian 视觉因子，再输入 GTSAM 图优化。该路径缺少 VINS-Fusion 中多年来工程化形成的多层鲁棒机制，例如视觉 landmark 级残差剔除、滑窗内状态一致性维护、IMU bias/velocity 约束闭环、初始化可观测性检查、失败检测和状态重启。因此，即使前端可视化质量较高，少量错误匹配、尺度不一致或惯性退化也可能被后端放大为全局轨迹失稳。

当前工程测试中最突出的两个具体现象可以表述为：

1. 在**大角度旋转且平移基线不足**的近似纯旋转运动中，后端 Sim(3) 尺度自由度的可观测性显著下降，容易出现尺度漂移、尺度突变乃至优化发散。
2. 在**室内扫描场景**中，外观检索引入的非连续候选边可能受到重复结构、相似纹理、窄基线视角和局部重叠不足影响。若缺少充分的几何一致性验证，这类 retrieval edge 可能作为错误因子进入后端图优化，对轨迹估计产生负贡献。

## 调研范围与资料来源

本文区分三类证据：

1. **当前本地实现**：以当前工作树中的 `main.py`、`mast3r_fusion/global_opt.py`、`mast3r_fusion/tracker.py`、`mast3r_fusion/backend/src/gn_kernels.cu` 和配置文件为准。
2. **VINS-Fusion/VINS-Mono 公开资料**：包括 VINS-Fusion 官方仓库和相关论文。VINS-Fusion 官方 README 将系统定义为 optimization-based multi-sensor state estimator，支持 monocular/stereo camera + IMU、在线外参/时间偏移标定和 loop closure。
3. **MASt3R-Fusion/MASt3R-SLAM 公开资料**：MASt3R-Fusion 论文强调 feed-forward pointmap regression 与 IMU/GNSS 融合，并将 Sim(3) 视觉约束以 Hessian 形式引入 metric-scale SE(3) factor graph；MASt3R-SLAM 论文强调基于 MASt3R prior 的 pointmap matching、tracking、local fusion、graph construction、loop closure 和 second-order global optimisation。

需要强调：本文比较的是**当前本地代码状态**与 VINS-Fusion 的成熟后端机制，不等价于否定 MASt3R-Fusion 论文方法本身。

## 统一优化视角

两类系统都可以抽象为非线性最小二乘问题：

$$
\mathcal{X}^{*}
= \arg\min_{\mathcal{X}}
\sum_{k \in \mathcal{F}}
\rho_k\left(
\left\|
\mathbf{r}_k(\mathcal{X})
\right\|_{\Omega_k}^{2}
\right),
$$

其中 $\mathcal{X}$ 表示待估状态，$\mathbf{r}_k$ 表示视觉、IMU、先验、回环或其他传感器残差，$\Omega_k$ 为信息矩阵，$\rho_k$ 为鲁棒核或等效加权策略。

VINS-Fusion 的典型滑窗状态包含位姿、速度、IMU bias、外参、时间偏移和 landmark 逆深度：

$$
\mathcal{X}_{vins}
=
\{
\mathbf{R}_i,\mathbf{p}_i,\mathbf{v}_i,\mathbf{b}_{a_i},\mathbf{b}_{g_i},
\lambda_j,\mathbf{T}_{bc},t_d
\}.
$$

当前 MASt3R-Fusion 后端更接近：

$$
\mathcal{X}_{mast3r}
=
\{
\mathbf{T}^{w}_{c_i}, s_i, \mathbf{v}_i, \mathbf{b}_{a_i}, \mathbf{b}_{g_i}, \mathbf{T}_{ic}
\},
$$

其中 dense pointmap 不作为长期 landmark 被显式维护，而是通过点图匹配形成 pairwise Sim(3) 视觉约束。其视觉边在线性化点附近可写成：

$$
E_{ij}
\approx
\frac{1}{2}
\left\|
\mathbf{e}_{ij}
+
\mathbf{J}_{ij}\delta \boldsymbol{\xi}_{ij}
\right\|_{\mathbf{W}_{ij}}^2,
$$

并被压缩为 Hessian 形式：

$$
\mathbf{H}_{ij}=\mathbf{J}_{ij}^{T}\mathbf{W}_{ij}\mathbf{J}_{ij},
\qquad
\mathbf{g}_{ij}=\mathbf{J}_{ij}^{T}\mathbf{W}_{ij}\mathbf{e}_{ij}.
$$

这一路径计算效率较高，但代价是很多点级诊断信息在进入 GTSAM 前被聚合；如果没有额外的因子级筛查、动态权重或后验 outlier rejection，坏匹配对状态的影响会更难隔离。

## 当前 MASt3R-Fusion 后端实现梳理

### 1. 前端 tracking 有基础鲁棒性

`FrameTracker.track()` 调用前端模型进行当前帧与最近关键帧匹配，使用置信度筛选和 Huber 加权优化相对 Sim(3) 位姿。关键逻辑包括：

```python
valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
match_frac = valid_opt.sum() / valid_opt.numel()
```

若 `match_frac` 低于 tracking 阈值，则进入重定位模式。位姿优化中使用：

```python
robust_sqrt_info = sqrt_info * torch.sqrt(huber(whitened_r, k=self.cfg["huber"]))
```

因此，用户观察到“前端 tracking 可视化较稳”是合理的：前端有局部帧间匹配、置信度筛选、Huber 权重和关键帧选择逻辑。

### 2. 后端图构建偏激进，连续边几乎无条件进入

`main.py` 中 `run_backend()` 对每个新关键帧默认添加一条前一关键帧边，并选择局部 retrieval 边：

```python
n_consec = 1
kf_idx.append(idx - 1 - j)
factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])
```

在 `FactorGraph.add_factors()` 中，候选边的双向 match fraction 会用于判定无效边：

```python
invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
consecutive_edges = ii_tensor == (jj_tensor - 1)
invalid_edges = (~consecutive_edges) & invalid_edges
```

这意味着：**非连续 retrieval 边可以按匹配率剔除，但连续关键帧边即使低于阈值也会保留**。在视觉前端偶尔失真、动态物体、弱纹理、过曝、快速转向或尺度突变时，这一策略容易把坏连续边固定进优化图。

### 3. 当前配置使后端门槛更宽松

以 `config/base_kitti360.yaml` 为例：

```yaml
tracking:
  min_match_frac: 0.00
  Q_conf: 1.5
  huber: 1.345

local_opt:
  min_match_frac: 0.0
  pixel_border: -10000
  sigma_pixel: 0.0005
```

这里有两个后果：

1. 后端局部边的 `min_match_frac` 默认为 0，几乎不按匹配比例拒绝边。
2. `pixel_border: -10000` 使投影有效区域判断极宽松，许多几何上已经离开图像范围的投影仍可能进入残差构造。

这并不必然错误，因为 dense pointmap 和 feed-forward prior 的尺度与投影关系不同于传统稀疏特征；但它降低了后端对坏观测的早期拦截能力。

### 4. 检索边缺少独立几何验收，室内扫描易产生负贡献

`run_backend()` 中 retrieval edge 的构造主要依赖检索数据库返回的候选索引，并通过 `find_valid_numbers()` 做非常轻量的距离与聚类筛选：

```python
retrieval_inds = retrieval_database.update(frame, add_after_query=True, ...)
retrieval_inds = find_valid_numbers(idx, retrieval_inds)

for kkk in retrieval_inds:
    if np.fabs(idx - kkk) < 20:
        retrieval_inds_selected.append(kkk)
```

随后这些 retrieval candidates 与当前关键帧一起进入 `factor_graph.add_factors()`。边级筛选主要由双向 match fraction 控制：

```python
invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
valid_edges = ~invalid_edges
```

在室内扫描场景中，这一策略存在一个典型风险：外观相似并不等价于几何一致。重复门框、墙面、走廊、桌椅、纹理块和局部扫描回访都可能触发较高的外观相似度；但当视角重叠不足或平移基线不合适时，MASt3R matching 产生的 dense correspondence 可能在局部视觉上可解释，却不能提供稳定的全局相对位姿约束。当前后端没有看到类似 pose graph loop-closure verification 中常见的 RANSAC/PnP/Sim(3) consistency check、switchable constraint 或优化后边级残差剔除。因此，在室内扫描类数据上，retrieval edge 可能从“补充约束”变成“错误长程约束”。

### 5. 点级 Huber 存在，但缺少边级和后验剔除闭环

CUDA 后端在 `calib_proj_kernel` 与 `calib_proj_kernel_pieces` 中对像素和 log-depth 残差使用 Huber 权重：

```cpp
w[0] = huber(sqrt_w_pixel * err[0]);
w[1] = huber(sqrt_w_pixel * err[1]);
w[2] = huber(sqrt_w_depth * err[2]);
```

并使用置信度与深度一致性项：

```cpp
if(d_diff_thresh > 0 && Xj[2] * (*sij) > Xi[2] * d_diff_thresh)
    downweight_factor = 0.01;
```

这说明 MASt3R-Fusion 后端并非完全没有鲁棒权重。但这些机制主要在**点级残差组装 Hessian 前**生效。当前本地实现中没有看到类似 VINS 的“优化后根据重投影误差删除 feature，再更新 feature manager/track manager”的闭环。因此，一条边只要被接纳，其聚合 Hessian 就会持续参与局部优化、边缘化和图保存。

### 6. GTSAM 优化迭代较浅，且没有显式失败回滚

`solve_GN_calib()` 中，每轮构造视觉 Hessian factor，再用 GTSAM Levenberg-Marquardt 优化：

```python
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(2)
optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initials, params)
cur_result = optimizer.optimize()
```

外层循环由 `local_opt.max_iters` 控制，但每次 GTSAM 内部 LM 只迭代 2 次。若当前线性化点较差，或者视觉 Hessian 与 IMU/先验冲突较强，浅迭代可能不足以收敛，也没有显式检查优化 summary、残差下降、尺度异常或位姿跳变后再决定是否接受结果。

### 7. 视觉-惯性初始化检查存在明显退化

`solve_VI_init()` 中计算 IMU excitation 的方差：

```python
var_g = math.sqrt(var_g / ccount)
if var_g < 0.0:
    print("IMU excitation not enough!")
else:
    vi_result = VisualIMUAlignment(...)
```

由于 `var_g` 是平方和开根号，不可能小于 0。因此该检查事实上永远不会拒绝低激励初始化。这与 VINS-Mono/VINS-Fusion 中强调初始化鲁棒性和 IMU 可观测性判断的设计思想明显不一致。低激励、时间偏差、外参误差或初始尺度不稳时，当前实现更容易进入错误的 VI 状态。

### 8. 滑窗边缘化存在，但信息保留较粗

当前实现中，`solve_GN_calib()` 使用 `window_num` 计算 `pin`，旧状态通过 `gtsam.marginalizeOut()` 形成 `marg_factor`。这说明系统具备滑窗边缘化框架。但边缘化之前的视觉信息已经是 pairwise Hessian factor；如果早期坏边进入并被边缘化，其影响会固化为 prior。VINS 中同样有边缘化固化问题，但其在边缘化前有 feature 级跟踪、三角化、鲁棒核、outlier rejection 和 failure handling 作为缓冲。

## VINS-Fusion 后端鲁棒机制概述

VINS-Fusion/VINS-Mono 代表传统优化式 VIO 的成熟工程路径。其鲁棒性来自多个层级的共同作用。

### 1. 状态设计显式包含 IMU 动力学变量

VINS-Fusion 的局部 odometry 论文将视觉与惯性传感器统一为 factor graph，并显式优化 pose、velocity、accelerometer bias、gyroscope bias、camera landmark depth 等状态。IMU 预积分因子在相邻关键帧间提供高频运动约束，视觉因子通过多帧 feature observation 约束位姿和逆深度。

其基本代价可写为：

$$
\min_{\mathcal{X}}
\left(
\sum_{(i,j)\in \mathcal{I}}
\left\|
\mathbf{r}^{imu}_{ij}
\right\|^2_{\Omega_{ij}^{imu}}
+
\sum_{l,t}
\rho
\left(
\left\|
\mathbf{r}^{cam}_{l,t}
\right\|^2_{\Omega^{cam}}
\right)
+
\left\|
\mathbf{r}^{prior}
\right\|^2_{\Omega^{prior}}
\right).
$$

IMU 不只是给关键帧选择提供辅助，而是贯穿初始化、传播、优化和边缘化。

### 2. 视觉残差使用鲁棒核

VINS-Fusion 官方源码中，视觉 projection factor 加入 Ceres problem 时使用 Huber loss。抽象地说，视觉残差不直接以平方误差进入，而是：

$$
\rho_{\delta}(s)
=
\begin{cases}
s, & s \leq \delta^2,\\
2\delta\sqrt{s}-\delta^2, & s > \delta^2.
\end{cases}
$$

这使单个异常视觉观测的影响从二次增长变成近似线性增长。

### 3. 优化后显式 outlier rejection

VINS-Fusion 在优化后调用 `outliersRejection()`，根据 feature 的平均重投影误差筛选异常 feature；再由 feature manager 与 feature tracker 删除这些 outlier。这是一种重要闭环：

```text
构建视觉残差 -> 非线性优化 -> 计算后验重投影误差 -> 删除异常 feature -> 滑窗推进
```

相比之下，当前 MASt3R-Fusion 后端更接近：

```text
dense match -> 聚合 Hessian factor -> GTSAM 优化 -> 接受状态 -> 滑窗/边缘化
```

缺失的正是“后验误差诊断并删除坏观测”的环节。

### 4. 滑窗状态迁移更完整

VINS-Fusion 的 `slideWindow()` 会同步迁移：

- pose；
- velocity；
- accelerometer bias；
- gyroscope bias；
- IMU preintegration；
- feature manager 中的观测关系；
- 边缘化 prior。

当前 MASt3R-Fusion 也维护 `wTcs/ss/vs/bs/preintegrations/marg_factor`，但视觉观测不以长期 landmark track 形式存在，旧视觉信息主要以 Hessian factor 和边缘化 prior 形式保留。因此当视觉因子本身质量控制不足时，滑窗反而会加速错误固化。

### 5. 初始化与失败恢复体系更完整

VINS-Mono 论文明确强调 robust initialization and failure recovery。VINS-Fusion 官方源码中虽有部分 failure detection 逻辑被直接 `return false` 短路，但整体系统仍保留了初始化、滑窗、outlier rejection、feature failure removal、IMU propagation 和 loop fusion 等链路。当前 MASt3R-Fusion 本地实现没有看到同等级别的状态回滚、重启、坏窗口拒绝或优化结果验收机制。

## 逐项问题-机制对照

上文分别说明了当前 MASt3R-Fusion 的后端问题和 VINS-Fusion 的后端机制。为了更清晰地解释“为什么相同现象下 VINS 更稳”，本节按**问题 -> VINS 对应处理 -> 稳定性来源**的方式展开。

### 1. 近似纯旋转下尺度退化

**MASt3R-Fusion 当前问题。**  
当前后端视觉变量包含 Sim(3) 尺度自由度，尺度比直接进入相对位姿：

```cpp
float si_inv = 1.0 / si[0];
sij[0] = si_inv * sj[0];
actSim3(tij, qij, sij, Xj, Xj_Ci);
```

在大角度旋转但平移基线很小的片段中，像素残差主要解释旋转，平移和尺度方向缺少足够视差约束。此时 log-depth 残差：

```cpp
err[2] = zj_log - zi_log;
```

高度依赖前端 pointmap 的尺度一致性。一旦点图尺度或匹配索引有局部偏差，后端可能通过调整 `s_i/s_j` 来吸收残差，导致尺度漂移或尺度跳变。

**VINS 的对应处理。**  
VINS-Fusion/VINS-Mono 不在每条视觉边中自由优化 pairwise Sim(3) 尺度，而是在滑窗状态中维护 metric SE(3) 位姿、速度和 IMU bias。尺度主要通过初始化阶段的视觉-惯性对齐、IMU 预积分、重力方向和 bias 估计来建立：

$$
\mathcal{X}_{vins}
=
\{\mathbf{R}_i,\mathbf{p}_i,\mathbf{v}_i,\mathbf{b}_{a_i},\mathbf{b}_{g_i},\lambda_j\}.
$$

也就是说，VINS 的尺度不是在每条视觉匹配边里单独漂移，而是受到 IMU 动力学和滑窗多帧一致性共同约束。对于低视差片段，VINS 的视觉前端/初始化流程还会利用 parallax、tracked feature 数量、PnP/三角化质量等条件决定是否适合初始化或加入强约束。

**为什么更稳。**  
近似纯旋转时，单纯视觉几何对尺度不可观；VINS 将尺度绑定到 IMU 传播和多帧运动一致性上，因此不会像当前 Sim(3) Hessian 边那样轻易通过局部尺度自由度吸收误差。其代价是 VINS 依赖 IMU 激励和标定质量，但当 IMU 可用且初始化合格时，尺度稳定性明显强于自由 Sim(3) 后端。

### 2. 连续坏边被强制保留

**MASt3R-Fusion 当前问题。**  
`FactorGraph.add_factors()` 中虽然计算了双向 match fraction，但连续边会绕过无效边筛选：

```python
invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
consecutive_edges = ii_tensor == (jj_tensor - 1)
invalid_edges = (~consecutive_edges) & invalid_edges
```

因此相邻关键帧即使匹配质量低，仍可能进入后端图优化。对大角度旋转、弱平移、动态遮挡或前端点图尺度异常片段，这会把局部坏约束固化进 Hessian factor。

**VINS 的对应处理。**  
VINS 中相邻帧之间并不是无条件加入一个“整帧相对位姿因子”。视觉约束来自长期跟踪的 feature observation，只有满足观测次数、三角化深度、重投影误差等条件的 feature 才持续参与优化。优化之后还会执行 outlier rejection：

```text
optimization()
outliersRejection(removeIndex)
f_manager.removeOutlier(removeIndex)
featureTracker.removeOutliers(removeIndex)
```

**为什么更稳。**  
VINS 的相邻帧约束是由许多可被单独删除的 feature residual 组成的，而当前 MASt3R-Fusion 后端更接近把一批 dense match 聚合为一条 pairwise Hessian edge。前者可以在优化后删除坏观测，后者一旦整边进入图中，缺少等价的边级开关或后验剔除机制。

### 3. 室内 retrieval edge 可能变成负约束

**MASt3R-Fusion 当前问题。**  
当前 retrieval edge 的构造先通过外观检索得到候选，再进行轻量索引筛选：

```python
retrieval_inds = retrieval_database.update(frame, add_after_query=True, ...)
retrieval_inds = find_valid_numbers(idx, retrieval_inds)

for kkk in retrieval_inds:
    if np.fabs(idx - kkk) < 20:
        retrieval_inds_selected.append(kkk)
```

这类候选随后进入 `add_factors()`，主要依赖 match fraction 进行边筛选。室内扫描场景中重复结构、相似纹理和局部视角重叠不足非常常见，外观近邻并不等价于几何闭环。若候选帧在几何上不一致，retrieval edge 会作为错误 Hessian factor 拉扯局部轨迹。

**VINS 的对应处理。**  
VINS 的局部 odometry 后端并不依赖外观检索边来维持实时滑窗稳定性；实时约束主要来自连续 feature track、IMU preintegration 和滑窗 prior。VINS-Fusion 的 loop/global fusion 是独立模块，通常需要额外的几何验证、位姿图约束和全局融合，而不是把外观检索结果直接当作局部 BA 的强边使用。

**为什么更稳。**  
VINS 将“局部里程计稳定性”和“回环/重定位约束”分层处理。当前 MASt3R-Fusion 把 retrieval candidate 更直接地接入局部优化，若缺少 RANSAC/PnP/Sim(3) 几何一致性验证、switchable constraint 或后验边残差删除，室内外观混淆就更容易污染实时后端。

### 4. 点级 Huber 有限，缺少优化后闭环剔除

**MASt3R-Fusion 当前问题。**  
CUDA kernel 中确实存在点级 Huber 和置信度加权：

```cpp
w[0] = huber(sqrt_w_pixel * err[0]);
w[1] = huber(sqrt_w_pixel * err[1]);
w[2] = huber(sqrt_w_depth * err[2]);
```

但这些权重主要发生在 Hessian 组装前。进入 GTSAM 后，当前 Python 层没有看到基于优化后重投影残差或 per-edge cost 的删除、降权和重新优化闭环。

**VINS 的对应处理。**  
VINS 同时做两层鲁棒处理：第一层是在 Ceres 中对 projection residual 使用 Huber loss；第二层是在优化后计算 feature 平均重投影误差，并删除超过阈值的 outlier feature。

```text
HuberLoss -> Solve -> reprojection error check -> remove outlier feature
```

**为什么更稳。**  
Huber 只能降低异常值影响，不能保证错误观测永远无害。VINS 的优势在于有“优化后诊断并清除”的闭环；当前 MASt3R-Fusion 后端缺的正是这一步。因此当错误匹配或错误 retrieval edge 仍能通过初筛时，Huber 不足以阻止其长期污染状态。

### 5. VI 初始化退化检查不足

**MASt3R-Fusion 当前问题。**  
当前 `solve_VI_init()` 中存在如下判断：

```python
var_g = math.sqrt(var_g / ccount)
if var_g < 0.0:
    print("IMU excitation not enough!")
else:
    vi_result = VisualIMUAlignment(...)
```

由于 `var_g` 不可能小于 0，该检查不会真正拒绝低激励初始化。若近似纯旋转片段本身已经导致视觉尺度弱可观，错误 VI 初始化会进一步把错误尺度、重力或 bias 带入后端。

**VINS 的对应处理。**  
VINS-Mono 的初始化流程强调足够视差、陀螺 bias 估计、视觉结构恢复、视觉-惯性对齐和重力/尺度求解。VINS-Fusion 在进入 nonlinear sliding-window optimization 前，也会根据传感器配置执行 PnP、triangulation、gyroscope bias solve、preintegration repropagation 等步骤。

**为什么更稳。**  
VINS 的初始化不是“达到固定帧数就切换融合状态”，而是把 feature parallax、IMU 预积分、bias 和尺度一致性放在同一个初始化门槛中检查。当前 MASt3R-Fusion 的 VI 初始化门槛较弱，容易在正好低视差/近似纯旋转的困难片段中误入融合状态。

### 6. 边缘化前的信息清洁度不同

**MASt3R-Fusion 当前问题。**  
当前后端有 `gtsam.marginalizeOut()`，但如果坏连续边或错误 retrieval edge 已经进入图，边缘化会把其影响固化为 prior。由于 dense match 已被聚合为 Hessian factor，边缘化前不易再恢复到可逐点清理的观测层。

**VINS 的对应处理。**  
VINS 在边缘化前已经经历 feature tracking、triangulation、Huber loss、optimization、outlier rejection 和 feature failure removal。进入 marginalization prior 的信息相对更“干净”。

**为什么更稳。**  
两者都会受到错误 prior 固化的风险影响；差异在于 VINS 在固化之前有更多清洗步骤。当前 MASt3R-Fusion 若缺少边级验收，边缘化会放大早期错误，而不是简单地平滑它。

## 鲁棒性差异总表

| 维度 | 当前 MASt3R-Fusion 本地实现 | VINS-Fusion/VINS-Mono 典型机制 | 对鲁棒性的影响 |
| --- | --- | --- | --- |
| 前端观测 | dense pointmap + dense/subpixel matching + confidence | KLT/sparse feature + stereo/temporal tracking | MASt3R 前端潜力更强，尤其弱纹理和大视角 |
| 局部 tracking | Huber + confidence + Sim(3) tracking | PnP/feature tracking + IMU propagation | 两者都有局部鲁棒性；MASt3R 可视化好是合理现象 |
| 后端视觉因子 | dense match 聚合为 pairwise Hessian | feature reprojection residual，landmark inverse depth | MASt3R 信息量大但可诊断性较弱 |
| 坏边过滤 | match fraction 阈值，但连续边例外；默认阈值常为 0 | feature 级观测生命周期管理 | MASt3R 坏连续边更容易进入图 |
| 检索边验收 | retrieval 后主要依赖 match fraction；缺少独立几何一致性与后验边剔除 | loop/relocalization 通常有几何验证和异常观测删除 | 室内重复结构中 retrieval edge 可能负贡献 |
| 鲁棒核 | 点级 Huber 和置信度加权 | Ceres Huber loss + 后验 outlier rejection | VINS 有更完整闭环 |
| 后验 outlier rejection | 未见等价机制 | 优化后按重投影误差删 feature | 这是核心差距之一 |
| IMU 融合 | GTSAM CombinedImuFactor，但 VI 初始化检查弱 | 预积分贯穿初始化、优化、传播、滑窗 | VINS 更像完整 VIO 后端 |
| 滑窗边缘化 | 有 `marginalizeOut`，但视觉信息已聚合 | prior + preintegration + feature manager 共同维护 | MASt3R 坏 prior 固化风险更高 |
| 失败检测/回滚 | 未见系统级验收和回滚 | 保留 failure/restart 体系和 feature failure removal | MASt3R 更容易“一错到底” |
| 回环/全局优化 | 有离线 loop/global optimization 路径 | VIO + loop fusion + global fusion | 实时前端弱点不能完全靠后处理修复 |

## 对用户现象的解释

用户观察到“前端 tracking 可视化比 VINS 更鲁棒，但整体 mast3r-fusion 更容易崩”，从系统分解角度看并不矛盾。

MASt3R 类前端学习了强 3D prior，能在传统稀疏角点较困难的区域给出密集点图和匹配。这会带来更好的局部可视化观感。但后端估计不是简单地“前端越强，轨迹越稳”。后端还必须回答以下问题：

1. 哪些匹配可信，哪些只是视觉上看起来合理？
2. 哪些边应该进入图，哪些边应该等待更多证据？
3. 优化后哪些观测被状态解释不了，应当删除？
4. 当前窗口是否退化，是否应拒绝状态更新？
5. IMU 是否可观测，尺度和重力是否可靠？
6. 边缘化前的信息是否足够干净？

当前 MASt3R-Fusion 后端在这些问题上采用了较轻量、较激进的策略，而 VINS-Fusion 的优势正来自对这些问题的长期工程化处理。因此，用户猜想“主要问题在后端”是成立的。

## 主要失效模式推断

### 1. 大角度近似纯旋转导致尺度退化

当相机发生大角度姿态变化但平移基线不足时，视觉观测对尺度的约束会显著减弱。对传统稀疏几何而言，纯旋转下三角化深度退化；对当前 MASt3R-Fusion 后端而言，问题表现为 Sim(3) 视觉边中的尺度自由度缺少足够的视差支撑。

从源码看，CUDA kernel 中相对位姿显式包含尺度比：

```cpp
float si_inv = 1.0 / si[0];
sij[0] = si_inv * sj[0];
```

点从 $j$ 帧变换到 $i$ 帧时使用 Sim(3)：

```cpp
actSim3(tij, qij, sij, Xj, Xj_Ci);
```

深度残差采用 log-depth：

```cpp
err[2] = zj_log - zi_log;
```

在近似纯旋转时，像素残差主要约束旋转，平移和尺度的耦合项缺少有效基线；log-depth 残差又高度依赖前端 pointmap 的尺度一致性。一旦点图尺度或匹配索引存在局部偏差，后端可能通过调整 `s_i/s_j` 吸收残差，从而出现尺度漂移或突变。与 VINS-Fusion 相比，当前实现没有在接受优化结果前显式检查尺度跳变，也没有对低视差窗口做拒绝更新或降权处理。

可将该现象概括为：

$$
\text{large rotation} + \text{small translation}
\Rightarrow
\text{weak scale observability}
\Rightarrow
\text{unstable Sim(3) scale update}.
$$

### 2. 坏连续边固化

当相邻关键帧的 dense match 局部错误、动态物体占比高或点图尺度异常时，连续边仍被保留。其 Hessian factor 被加入优化后，可能直接拉偏最新关键帧位姿。若随后边缘化，该错误还会变成 prior。该问题在近似纯旋转场景中更严重，因为后端缺少平移基线来区分“真实尺度变化”和“由旋转匹配误差诱导的尺度补偿”。

### 3. 尺度自由度不稳定

MASt3R-Fusion 使用 Sim(3) 视觉约束并维护尺度 `s_i`。在视觉-only 阶段，尺度靠先验和点图几何约束维持；如果连续边存在尺度冲突，后端可能通过错误尺度吸收残差。VINS 在 monocular + IMU 场景中通过 IMU 预积分、重力和 bias 建立 metric scale，但也依赖严格初始化。

### 4. 室内检索边引入错误长程约束

室内扫描场景通常包含大量重复结构和相似局部外观。当前 retrieval edge 的生成更偏外观近邻召回，而不是严格几何闭环验证。`find_valid_numbers()` 仅排除与当前索引过近的帧，并在候选密集时保留局部代表；随后又限制 `abs(idx - kkk) < 20`，使其更像局部重叠边，而不是全局闭环边。该策略在室外大尺度轨迹中可能补充共视约束，但在室内扫描中可能出现如下问题：

1. 外观相似但三维结构并非同一局部区域；
2. 当前帧与候选帧视角重叠不足，dense match 产生局部一致但全局错误的对应；
3. 大角度转向后检索到的候选帧提供错误尺度或错误相对旋转约束；
4. 错误 retrieval edge 被转为 Hessian factor 后缺少 switch 或后验残差删除机制。

因此，室内 retrieval edge 的负贡献并非检索模块本身必然错误，而是当前后端对检索边缺少足够严格的几何验收和优化后剔除。

### 5. 低激励 VI 初始化误入融合状态

当前 `var_g < 0.0` 的检查使低激励不会被拒绝。若 VI 初始化在尺度、重力、bias 上不可靠，后续 IMU factor 不但不能稳定视觉，反而会与视觉 Hessian 发生系统性冲突。

### 6. 后验残差不可见

`hessian_pieces()` 计算了 per-edge cost，但 Python 层没有看到基于该 cost 的边剔除、降权或优化验收策略。系统缺少一个类似：

```text
if edge_cost_after_optimization > threshold:
    remove_or_downweight_edge()
```

的闭环。

### 7. 回环与全局优化不能补救实时坏状态

`main_loop.py` 和 `main_global_optimization.py` 提供离线 loop/global 路径，但实时估计如果已经把坏关键帧、坏尺度和坏 prior 写入 `data.h5/graph.pkl`，后处理只能在已有信息上修正，不能完全恢复被错误滑窗污染的局部结构。

## 建议的后续验证实验

本文按用户要求不修改代码，仅提出验证方案。

### 1. 记录边级质量与轨迹跳变的相关性

建议在服务器环境同步记录：

- 每条后端边的 `match_frac_j/match_frac_i`；
- 每条边的 `c11` cost；
- 最新关键帧优化前后位姿差；
- scale `s_i` 的变化；
- IMU bias/velocity 范数。

若轨迹崩溃前出现低 match fraction 连续边、高 edge cost 或 scale 跳变，即可强证后端坏边固化假设。

### 2. 针对近似纯旋转场景记录尺度可观测性指标

建议在包含“大角度旋转、低平移”的片段上记录：

$$
\theta_{ij}=\|\log(\mathbf{R}_{ij})\|,
\qquad
b_{ij}=\|\mathbf{t}_{ij}\|,
\qquad
\Delta s_{ij}=|\log(s_j/s_i)|.
$$

若在 $\theta_{ij}$ 较大、$b_{ij}$ 较小的片段上持续出现 $\Delta s_{ij}$ 异常增大，且前端 tracking 可视化仍保持稳定，则说明崩溃主要来自后端尺度自由度退化，而非前端完全失配。

### 3. 对室内 retrieval edge 做消融实验

建议对室内扫描数据分别运行：

```text
仅连续边
连续边 + retrieval edge
连续边 + 通过更严格几何/残差门控的 retrieval edge
```

如果第二种设置比第一种更差，而第三种恢复稳定，则可直接支持“室内检索边负贡献来自后端验收不足”的判断。

### 4. 对比三种后端门控配置

不改算法，仅调配置可做：

```yaml
local_opt:
  min_match_frac: 0.03
  pixel_border: 5
```

并额外测试“不强制保留连续边”的实验分支。若鲁棒性明显改善，说明边级质量控制是主因之一。

### 5. 分离视觉-only 与 VI 阶段

建议分别评估：

```bash
python main.py --start_from 0 --end_at 200 --no-viz ...
```

并记录 `T_WCs.shape[0] == 7` 触发 VI 初始化前后的轨迹、尺度和 bias 变化。若初始化后误差突增，重点检查 VI 初始化和 IMU factor；若初始化前已漂移或跳变，则重点检查视觉 Hessian 因子与边筛选。

### 6. 对后端优化结果做只读诊断

建议导出每轮优化的：

$$
\Delta T_i = T_{i,\text{after}}^{-1}T_{i,\text{before}},
\qquad
\Delta s_i = s_{i,\text{after}} - s_{i,\text{before}}.
$$

若单次后端调用产生大幅跳变，而前端 tracking 仍稳定，则可直接定位为后端状态接受策略不足。

## 结论

综合源码阅读和公开资料对比，可以确认：

1. 当前 MASt3R-Fusion 的前端 tracking 具备较强局部匹配和点图能力，用户观察到其可视化比 VINS 鲁棒是可信的。
2. 对近似纯旋转尺度退化问题，当前 MASt3R-Fusion 后端在 Sim(3) 视觉边中保留自由尺度，低平移基线下容易由尺度吸收残差；VINS 则通过 metric SE(3) 状态、速度、IMU bias、预积分和视觉-惯性初始化共同约束尺度，因此在 IMU 可用且初始化合格时更不容易发生尺度跳变。
3. 对连续坏边问题，当前 MASt3R-Fusion 会让连续关键帧边绕过 match fraction 剔除；VINS 的连续视觉约束来自可单独管理的 feature residual，并在优化后通过重投影误差删除 outlier feature，因此不会把整帧坏相对约束无条件固化进图。
4. 对室内 retrieval edge 负贡献问题，当前 MASt3R-Fusion 更直接地将外观检索候选加入局部优化；VINS 将局部里程计与回环/全局融合分层处理，实时滑窗主要依赖连续 feature track 和 IMU 预积分，回环类约束通常需要额外几何验证，因此更不容易被室内重复结构污染局部后端。
5. 对后验残差不可见问题，当前 MASt3R-Fusion 的点级 Huber 发生在 Hessian 聚合前，缺少优化后的边级/观测级清理；VINS 同时使用 Ceres Huber loss 和优化后 outlier rejection，形成“先降权、再删除”的闭环。
6. 对 VI 初始化和边缘化固化问题，当前 MASt3R-Fusion 的低激励检查不足，坏视觉边或错误尺度一旦边缘化会固化为 prior；VINS 在初始化、预积分重传播、feature failure removal 和边缘化前清理方面有更完整的工程流程。
7. 因此，造成“前端好、系统差”的关键不是单个 bug，而是后端缺少与强前端匹配的鲁棒估计闭环。若目标是在服务器环境中获得接近或超过 VINS-Fusion 的系统鲁棒性，优先方向应是低视差/纯旋转退化检测、连续边质量验收、retrieval edge 几何验证、后验残差剔除、边级动态权重、VI 初始化验收和失败恢复，而不是盲目更换前端模型。

## 参考资料

1. Yuxuan Zhou et al., *MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM*, arXiv:2509.20757, https://arxiv.org/abs/2509.20757
2. GREAT-WHU, *MASt3R-Fusion official repository*, https://github.com/GREAT-WHU/MASt3R-Fusion
3. Riku Murai, Eric Dexheimer, Andrew J. Davison, *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors*, arXiv:2412.12392, https://arxiv.org/abs/2412.12392
4. Tong Qin, Peiliang Li, Shaojie Shen, *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*, arXiv:1708.03852, https://arxiv.org/abs/1708.03852
5. Tong Qin, Jie Pan, Shaozu Cao, Shaojie Shen, *A General Optimization-based Framework for Local Odometry Estimation with Multiple Sensors*, arXiv:1901.03638, https://arxiv.org/abs/1901.03638
6. Tong Qin, Shaozu Cao, Jie Pan, Shaojie Shen, *A General Optimization-based Framework for Global Pose Estimation with Multiple Sensors*, arXiv:1901.03642, https://arxiv.org/abs/1901.03642
7. HKUST-Aerial-Robotics, *VINS-Fusion official repository*, https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
