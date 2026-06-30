# CoTracker Frontend Lifecycle Notes

本文记录一个设计方向：引入 CoTracker 作为轻量化 tracker frontend，为
MASt3R-Fusion 补充跨帧点轨迹和生命周期管理。这里的目标不是用 CoTracker
替代 MASt3R 或 PI3X 的几何前馈能力，而是把它作为高频、轻量、可维护状态的
tracking 层，让重模型在更合适的时机提供窗口重建和几何校正。

## 背景问题

MASt3R-Fusion 当前更偏向强匹配和前馈几何估计。MASt3R 可以在图像对之间给出
高质量 correspondence 和 point map，但它不是传统意义上的 tracker。系统缺少
一个显式维护点轨迹生命周期的 frontend，因此很难稳定回答这些问题：

- 一个点从哪一帧出生，已经被跟踪了多久；
- 这个点是否连续被观测，什么时候失效；
- 哪些点是长期稳定的几何约束；
- 哪些点只是某次前馈模型输出中的瞬时匹配；
- 当前窗口是否已经退化，是否需要调用 PI3X 或 MASt3R 重新建模。

在缺少 lifecycle 的情况下，跨帧一致性主要依赖 retrieval、pairwise matching
和模型本身的泛化能力。这种方式对局部匹配很强，但对长期状态管理不够直接。

## 为什么引入 CoTracker

CoTracker 适合作为轻量化 tracking frontend，因为它可以维护跨多帧的 2D
tracks。相比只做帧间 match，tracks 能提供更明确的时序状态：

- track id：同一个点在不同帧中的身份；
- age：轨迹寿命；
- visibility：每帧是否可见；
- confidence：跟踪置信度；
- death / lost：轨迹失效或丢失；
- birth：新轨迹初始化；
- spatial coverage：活跃点在图像上的覆盖情况。

这些状态可以成为后端窗口管理、稀疏点优化、关键帧选择和重模型触发的输入。

## 对 MASt3R-Fusion 的补充价值

### 1. 从 match 到 track

MASt3R 强在回答两帧之间的对应关系，但 pairwise match 不等价于持续跟踪。一个
SLAM 系统最终需要长期可维护的 landmark 和 observation，而不是一组彼此独立的
图像对匹配结果。

引入 CoTracker 后，可以把短期和中期的连续观测显式组织成 tracks。这样后端可以
知道哪些观测来自同一个潜在 landmark，哪些约束是长期稳定的，哪些只是临时匹配。

### 2. 用 lifecycle 驱动窗口化重建

如果没有 lifecycle，系统通常只能用固定帧间隔、retrieval 分数、匹配数量或局部
heuristic 来决定何时调用 PI3X / MASt3R 做窗口化重建。这类策略容易过粗，且很难
准确表达 tracking 是否退化。

有了 CoTracker 管理的 track lifecycle 后，可以用更自然的信号触发重模型：

- 活跃 track 数量快速下降；
- 大量长寿命 tracks 同时 lost；
- 新出生 tracks 比例过高；
- 长寿命 tracks 的图像覆盖区域不足；
- tracks 累计视差足够大，适合三角化或深度更新；
- 当前窗口内 reprojection residual 持续增大；
- 跟踪置信度整体下降。

这样 PI3X / MASt3R 可以从固定频率调用，变成由 tracking 状态驱动的低频校正和
窗口重建模块。

### 3. 支持稀疏点深度优化和过滤

MASt3R / PI3X 的前馈深度和 point map 可以作为很强的几何 prior，但不应该总是
被当成最终真值。引入 tracks 后，每个稀疏点可以维护更多可优化、可过滤的状态：

- 观测次数；
- 连续跟踪长度；
- 最近一次观测帧；
- 多帧 reprojection error；
- 深度估计方差；
- 是否频繁成为 outlier；
- 是否疑似动态点；
- 是否适合作为 local BA 或 window BA 约束。

这使得系统可以对稀疏点做更传统也更可控的几何处理：长寿命点可以进入深度优化，
短命点可以降权或剔除，前馈模型给出的深度可以作为 prior 被多帧几何约束修正。

### 4. 降低重模型调用频率

CoTracker 可以在相邻帧和短窗口内持续传播点的位置。MASt3R / PI3X 则只在必要时
提供更强的几何初始化或窗口校正，例如：

- 新窗口开始；
- 跟踪明显退化；
- 场景发生切换；
- 视差足够大，需要刷新深度；
- sparse backend 的不确定性升高；
- 局部地图点质量明显下降。

这种分工可以形成“轻 tracker 高频运行，重 reconstruction model 低频校正”的
结构，降低整体计算压力。

### 5. 提供时间一致性约束

Pairwise matching 可能出现 A-B 和 B-C 都合理，但 A-B-C 整体轨迹不一致的问题。
CoTracker 直接输出跨多帧轨迹，因此可以提供更直接的 temporal consistency。

这对于 SLAM 很重要，因为后端真正需要的是持续的观测链条，而不是孤立的两帧匹配。

### 6. 帮助识别动态点和不稳定点

CoTracker 本身不直接判断三维刚体运动，但 tracks 可以结合 pose、depth 和
reprojection residual 判断哪些点不符合静态场景假设：

- 某些 tracks 长期不符合相机运动模型；
- 局部区域的轨迹运动和背景不一致；
- 某些点的 residual 长期偏高；
- 某些 tracks 频繁 lost / reappeared。

这些点可以被降权、剔除，或不进入 map point 集合，从而提升后端鲁棒性。

## 推荐架构定位

推荐把 CoTracker 定位为 tracking frontend，而不是 reconstruction frontend：

- CoTracker：维护 2D tracks、track id、visibility、confidence 和 lifecycle；
- MASt3R / PI3X：提供点云、深度、强匹配、窗口几何初始化和困难场景校正；
- sparse backend：融合 track observations、depth prior、pose prior 和
  reprojection constraints；
- window manager：根据 track lifecycle 和 backend 质量决定何时开窗、重建或
  插入关键帧。

一个可能的数据流是：

```text
images
  -> CoTracker sparse / semi-dense tracks
  -> lifecycle manager
  -> keyframe / window decision
  -> PI3X or MASt3R window inference when needed
  -> sparse depth initialization / update
  -> backend optimization and point filtering
```

## 需要验证的风险

CoTracker 并不是完整替代方案，需要重点验证以下问题：

- CoTracker 输出是 2D tracks，不直接提供可靠 3D；
- 长序列 tracking 可能漂移，需要 MASt3R / PI3X 或几何后端周期性校正；
- 遮挡、动态物体、运动模糊仍然会导致错误 tracks；
- 点采样策略会影响几何约束质量，不能只追踪纹理强但几何分布差的区域；
- 运行开销需要实测，尤其是高分辨率或半稠密 tracking；
- CoTracker confidence 不能单独作为可靠性判断，需要结合几何误差；
- track lifecycle 与现有 backend factor、keyframe 和 retrieval 逻辑的接口需要谨慎设计。

## 初步实现方向

可以先做一个很小的实验性 frontend，不立即重构主流程：

1. 在单独模块中封装 CoTracker adapter，输入最近若干帧图像，输出 sparse tracks。
2. 为每个 track 维护 id、age、last_seen、visible_count、confidence 和状态。
3. 在现有 keyframe / frontend model 逻辑外侧记录 tracking 质量统计。
4. 先只用 lifecycle 指标做窗口触发分析，不直接改变 backend。
5. 再逐步把长寿命 tracks 转成 sparse observations，接入深度初始化和过滤逻辑。

这样可以先验证 CoTracker 是否能稳定提供有用的时序信号，再决定是否把它提升为
主 frontend 的正式组成部分。

## 后续实现草案

可以先把 CoTracker 接入 `sparse_vio_demo`，保持后端接口不变：

- 新增 CoTracker frontend adapter，输出现有 `FrontendResult.tracks`；
- 新增 lifecycle manager，维护 track cohort、age、last_seen、lost_count 和
  oldest cohort 消失事件；
- 让 `sparse_vio_demo/main.py` 支持 `--frontend cotracker`；
- 在配置中增加 CoTracker 和 lifecycle 参数；
- 增加一个独立的 PI3X rebuild scheduler，在 lifecycle 触发时收集最近窗口图像和
  当前 sparse backend 的 `T_wc` 作为 PI3X 外参先验；
- 增加 CoTracker 可选依赖声明。

PI3X 触发可以先分成两层：

```text
debug["pi3x_rebuild_request"] = 1.0
debug["pi3x_rebuild_reason_oldest_cohort_expired"] = 1.0
```

也就是说，当最老的一批 track cohort 全部失效时，frontend 会请求一次 PI3X
窗口重建。主循环随后通过 `Pi3XRebuildScheduler` 生成一个 window job，里面包含：

- 触发帧 id；
- 最近窗口的 frame ids；
- 最近窗口 RGB 图像；
- sparse backend 当前可用的 `T_wc` pose priors。

第一阶段可以不真正运行 PI3X 模型，也不修改 sparse backend 或主 MASt3R-Fusion
后端。下一步再让 scheduler 调用 PI3X window inference，并把 `pose_priors` 转换成
PI3X adapter 能消费的 camera pose prior。

## PI3X 重建和因子图融合的后续思考

后续比较合理的接入方式是增加一个独立调度器，而不是让 CoTracker frontend 直接
改写后端：

1. CoTracker lifecycle manager 发出 `pi3x_rebuild_request`。
2. 调度器收集最近一个 local window 的图像、时间戳和当前 SLAM pose。
3. 将当前 SLAM 轨迹转换为 PI3X 可消费的 camera pose prior。
4. 调用 PI3X window inference，得到 point maps、confidence 和 PI3X camera poses。
5. 将 PI3X 输出先作为 local geometry prior 缓存到 window 级别。
6. 后续再考虑把这些 prior 转成 factor graph 约束。

PI3X 输出加入因子图时，至少有几种可能形式：

- depth prior factor：对 CoTracker 长寿命 tracks 对应像素施加深度先验；
- reprojection factor：把 PI3X 点云投影到多帧 track observations 上；
- relative pose prior：使用 PI3X window poses 给相邻关键帧增加相对位姿先验；
- map point initialization：用 PI3X point map 初始化 sparse landmarks，然后由
  后端 BA 修正；
- outlier filter：用 PI3X confidence 和多帧 reprojection residual 共同过滤低质点。

这一部分当前只保留为设计思考，不在本分支的第一步实现中修改后端。

## 当前结论

这个方向是可行的，核心价值是用 CoTracker 补上 MASt3R-Fusion 缺失的时序状态
管理层。CoTracker 不负责替代 MASt3R / PI3X 的强几何能力，而是负责维护 tracks
和 lifecycle。系统因此可以更合理地决定何时重建、哪些点值得优化、哪些观测应该
剔除，以及如何把前馈模型的输出从一次性预测变成可被后端吸收和修正的几何 prior。
