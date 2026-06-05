# PI3 目标数据集格式说明

本目录用于实验阶段的数据集格式自动转换工具开发。当前第一版目标格式是
PI3-ready 序列格式：它既可以被本地 validator 独立检查，也可以在后续通过
adapter 接到 PI3 官方训练分支的 `BaseDataset` 接口。

PI3 官方训练分支的数据样本通常以 view 字典表示，常见字段包括 `img`、
`depthmap`、`camera_intrinsics`、`camera_pose`、`dataset`、`label` 和
`instance`。本地转换格式会显式保存这些概念，同时保持文件结构简单，方便
后续批量转换脚本输出和校验。

## 目录结构

```text
converted_sequence/
  camera.json
  frames.csv
  images/
    000000.png
    000001.png
    ...
  depth/           # 可选
    000000.npy
    000001.npy
    ...
  poses.txt        # 可选
```

## `camera.json`

必需文件，用于保存转换后序列的相机模型和内参。

```json
{
  "model": "pinhole",
  "width": 1242,
  "height": 375,
  "K": [
    [718.856, 0.0, 607.1928],
    [0.0, 718.856, 185.2157],
    [0.0, 0.0, 1.0]
  ],
  "distortion": [0.0, 0.0, 0.0, 0.0]
}
```

字段说明：

- `model`：相机模型名，当前默认使用 `pinhole`。
- `width`、`height`：转换后的图像尺寸。
- `K`：3x3 相机内参矩阵，单位为像素。
- `distortion`：可选畸变参数。转换后的 PI3 数据通常建议已经完成去畸变，
  因此这里可以写全零。

## `frames.csv`

必需文件，用于保存有序帧索引。

```csv
frame_id,timestamp,image_path,depth_path
000000,0.000000,images/000000.png,depth/000000.npy
000001,0.100000,images/000001.png,depth/000001.npy
```

必需列：

- `timestamp`：帧时间戳，单位为秒。
- `image_path`：图像路径，相对于数据集根目录。

可选列：

- `frame_id`：稳定帧 id。如果省略，loader 会按行号生成零填充 id。
- `depth_path`：深度图路径，相对于数据集根目录。支持 `.npy`、`.npz`、
  `.png`、`.tif` 和 `.tiff`。

校验规则：

- `frames.csv` 必须包含表头。
- 时间戳必须是有限数值，并且单调不递减。
- 图像和深度路径必须位于数据集根目录内部。
- 图像文件必须存在，并使用支持的图像扩展名。
- 如果存在 `depth_path`，深度文件必须存在，并使用支持的深度扩展名。
- 使用 `--check-images` 时，loader 会读取图像和可选深度图，并检查尺寸是否
  与 `camera.json` 中的 `width`、`height` 一致。

## `poses.txt`

可选文件，用于保存每帧 4x4 相机位姿。每一行以 `frame_id` 开头，后面跟
16 个按行优先排列的矩阵数值。

```text
000000 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
000001 1 0 0 0.1 0 1 0 0 0 0 1 0 0 0 0 1
```

当源数据集提供可靠位姿时，建议保存 camera-to-world 形式的 4x4 矩阵。该字段
会映射到 PI3 view 中的 `camera_pose`。

## PI3 View 字段映射

本地 loader 保留转换检查所需的元信息，同时可以把每一帧导出为 PI3 风格的
view 字典：

| PI3 view 字段 | 本地格式来源 |
| --- | --- |
| `img` | 从 `frames.csv:image_path` 读取的图像 |
| `depthmap` | 从可选 `frames.csv:depth_path` 读取的深度图 |
| `camera_intrinsics` | `camera.json:K` |
| `camera_pose` | 可选 `poses.txt` 位姿矩阵 |
| `dataset` | 固定值 `converted_pi3` |
| `label` | 数据集根目录名 |
| `instance` | `frames.csv:frame_id` |

对转换结果校验来说，深度和位姿是可选的。对 PI3 训练来说，官方
`BaseDataset` 子类通常需要深度图，因为训练流程会基于 `depthmap` 和
`camera_intrinsics` 推导 `pts3d` 和 `valid_mask`。

## 校验转换后的数据集

快速结构检查：

```bash
python scripts/pi3_dataset_loader.py <converted_sequence>
```

完整图像和深度尺寸检查：

```bash
python scripts/pi3_dataset_loader.py <converted_sequence> --check-images
```

如果校验失败，命令会以退出码 `1` 结束，并输出具体错误信息，方便后续自动
转换脚本把它作为 post-conversion gate。

## Python 调用示例

```python
from scripts.pi3_dataset_loader import load_pi3_dataset

dataset = load_pi3_dataset("converted_sequence", load_images=True)
sample = dataset[0]
image = sample["img"]
timestamp = sample["timestamp"]
camera_intrinsics = sample["camera_intrinsics"]

view = dataset.as_pi3_view(0)
```
