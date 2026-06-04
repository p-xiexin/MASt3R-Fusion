# closeloop PLY 导出问题说明

之前观察到的 closeloop 点云 PLY 错位，主要是导出脚本的问题，不代表 realtime SLAM 主流程本身异常。

## 问题原因

`data.h5` 中保存的点云 `X` 不是最终可直接对比的世界系点云，而是每个 keyframe 的局部/canonical pointmap。官方 `evaluation/check_h5.py` 可视化时会做几步额外处理：

1. 用对应 keyframe 的 `T_WC` 位姿把局部点云变换到世界系。
2. 对 Sim3 位姿中的 scale 做特殊处理：先把 `X` 乘以 scale，再把位姿 scale 置为 `1.0` 后可视化。
3. 在 `use_calib=True` 时，不直接使用 H5 中的原始三维 `X`，而是用 `X[..., 2]` 深度和相机内参沿像素 ray 重新生成点云。
4. 使用 closeloop/global optimization 的结果文件时，pose 文件中的索引对应 H5 keyframe index，而不是普通图像原始 frame id。

旧的 PLY 导出逻辑没有完整复刻这些步骤，尤其漏掉了标定情况下的 ray projection，并且 pose 索引匹配也容易混淆。因此导出的 PLY 会和 `check_h5.py` 里看到的官方点云不一致，表现为明显错位。

## 为什么 realtime SLAM 是正常的

realtime SLAM 主流程并不依赖这个 PLY 导出脚本。实时跟踪、局部优化、回环因子、全局优化使用的是系统内部的 keyframe pointmap、位姿和视觉因子数据流。

也就是说，错误发生在“把 H5 数据离线导出成 PLY 用于人工对比”的后处理阶段，而不是发生在 realtime SLAM 的跟踪、建图或优化阶段。

`check_h5.py` 的官方可视化逻辑和实时可视化逻辑是一致的；修正后的 `evaluation/export_h5_pointcloud.py` 需要以 `check_h5.py` 为基准，复刻同样的 scale、标定和 pose 索引处理，导出的 PLY 才能用于和官方可视化做公平对比。

## 结论

之前的 closeloop PLY 错位更可能是导出逻辑问题，而不是方法本身或 realtime SLAM 结果错误。修正导出脚本后，应优先对比同一段 `check_h5.py --frame_id` 可视化窗口和对应导出的 PLY，再判断是否存在算法层面的点云质量问题。
