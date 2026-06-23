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

## VINS 中额外可借鉴的后端设计

本节只列出前面逐项对比中尚未展开的 VINS 设计点。也就是说，这里不再重复几何验收、后验 outlier rejection、低视差退化检测、VI 初始化验收、边缘化前清理和前后端分层，而是补充一些更细的工程机制。

### 1. IMU 传播用于实时状态预测和优化初值

VINS 不仅在后端优化中加入 IMU factor，还会用 IMU propagation 维持最新状态预测，为下一帧 tracking、PnP 或滑窗优化提供更稳定的初值。该设计的核心价值是降低视觉前端在快速旋转、短时模糊或帧间视差不足时对上一帧视觉位姿的单点依赖。

对 MASt3R-Fusion 来说，这一点可以借鉴为：即使暂不完全依赖 IMU 约束尺度，也可以把 IMU prediction 作为 tracking 和后端局部优化的 motion prior，尤其用于大角度旋转但平移不足的片段。这样做的目标不是替代视觉约束，而是给 Sim(3)/SE(3) 优化一个更合理的初始姿态。

### 2. Bias 更新后的预积分重传播

VINS 在估计陀螺 bias 后，会对已有 preintegration 执行 repropagation，使 IMU 约束与最新 bias 估计保持一致。这个细节很重要：IMU bias 一旦变化，旧的预积分量如果不更新，会把错误惯性信息继续带入滑窗。

对应到当前系统，如果后端维护 `bs/vs/preintegrations`，就需要关注 bias 更新和预积分量之间的一致性。否则视觉约束已经在修正尺度或姿态，而 IMU factor 仍使用旧 bias 线性化结果，可能产生视觉-惯性冲突。

### 3. 按关键帧质量选择边缘化策略

VINS 的滑窗不是简单固定删除最老帧；它会区分 marginalize old frame 和 marginalize second-newest frame 等策略，使窗口保留对当前运动更有价值的关键帧。这个设计没有直接等同于前文的“边缘化前清理信息”，它更强调**该保留哪一帧、该丢弃哪一帧**。

MASt3R-Fusion 当前按 `window_num` 和 `pin` 推进窗口，更偏固定窗口。可借鉴 VINS 的策略，将关键帧保留与视差、旋转量、retrieval 可靠性、尺度稳定性和边连接质量关联起来。对近似纯旋转片段，不一定应该急于把低基线帧作为强关键帧固化进窗口。

### 4. 参数估计采用“条件开启”而非始终自由

VINS 中外参、时间延迟等参数并非始终无条件优化，而是根据运动激励和配置决定是否开启估计。例如外参估计通常需要足够运动，否则保持固定更稳。这一设计原则可以抽象为：**只有当数据对某个自由度有足够可观测性时，才释放该自由度**。

对 MASt3R-Fusion 最直接的启发是尺度、外参和可能的相机-IMU对齐参数不应始终等权自由优化。在低视差、纯旋转或弱 IMU 激励阶段，应该临时固定或强正则化不可观自由度；等运动提供足够约束后再释放。

### 5. 用运行时健康指标驱动模式切换

VINS 内部维护诸如 tracked feature 数量、parallax、solver 状态、bias 范数、最新位姿变化等健康指标。这些指标不仅用于日志，也用于初始化、滑窗、失败检测或输出状态判断。

MASt3R-Fusion 可以建立类似的后端健康指标体系，但指标应适配 dense 前端，例如：

```text
valid dense match ratio
edge cost percentile
scale jump magnitude
rotation-baseline ratio
retrieval edge disagreement
IMU prediction residual
```

这些指标可以驱动 `TRACKING / RELOC / DEGRADED / HOLD_SCALE` 等模式，而不是让所有帧都走同一条后端接受路径。

### 6. 求解器时间预算和优化强度自适应

VINS 会根据实时性需求限制 Ceres 求解时间，并在不同 marginalization 情况下设置不同预算。这个机制的重点不是“迭代越多越好”，而是让后端优化强度与当前窗口复杂度、实时约束和状态风险匹配。

当前 MASt3R-Fusion 的 GTSAM 内层 LM 迭代数较固定。可借鉴 VINS 的思路：普通帧采用轻量优化；检测到尺度跳变、retrieval 边冲突或纯旋转退化时，临时提高优化/诊断强度；若仍不能稳定下降，则拒绝该次状态更新或降级处理。

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
