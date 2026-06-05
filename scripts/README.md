# PI3 Target Dataset Format

This directory contains experimental utilities for dataset format conversion.
The first target format is a PI3-ready sequence format that can be validated
independently and later adapted to the official PI3 training `BaseDataset`
interface.

The official PI3 training branch represents samples as view dictionaries with
keys such as `img`, `depthmap`, `camera_intrinsics`, `camera_pose`, `dataset`,
`label`, and `instance`. This local format keeps those concepts explicit while
remaining simple enough for conversion scripts to validate.

## Directory Layout

```text
converted_sequence/
  camera.json
  frames.csv
  images/
    000000.png
    000001.png
    ...
  depth/           # optional
    000000.npy
    000001.npy
    ...
  poses.txt        # optional
```

## `camera.json`

Required. Stores the camera model and intrinsics for the converted sequence.

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

Fields:

- `model`: camera model name. `pinhole` is the current default.
- `width`, `height`: image size after conversion.
- `K`: 3x3 camera intrinsic matrix in pixel coordinates.
- `distortion`: optional distortion coefficients. Converted PI3 datasets should
  normally be undistorted already, so zeros are acceptable.

## `frames.csv`

Required. Stores the ordered frame index.

```csv
frame_id,timestamp,image_path,depth_path
000000,0.000000,images/000000.png,depth/000000.npy
000001,0.100000,images/000001.png,depth/000001.npy
```

Required columns:

- `timestamp`: frame timestamp in seconds.
- `image_path`: image path relative to the dataset root.

Optional columns:

- `frame_id`: stable frame id. If omitted, the loader assigns zero-padded ids
  from row order.
- `depth_path`: depth map path relative to the dataset root. Supported formats
  are `.npy`, `.npz`, `.png`, `.tif`, and `.tiff`.

Validation rules:

- `frames.csv` must contain a header row.
- Timestamps must be finite and monotonically non-decreasing.
- Image paths must stay inside the dataset root.
- Image files must exist and use a known image extension.
- Depth paths, when present, must stay inside the dataset root and use a known
  depth extension.
- With `--check-images`, every image must be readable and match `camera.json`
  width and height. Depth maps are also read and checked when `depth_path` is
  present.

## `poses.txt`

Optional. Stores one 4x4 camera pose per line. Each line starts with `frame_id`
followed by 16 row-major matrix values.

```text
000000 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
000001 1 0 0 0.1 0 1 0 0 0 0 1 0 0 0 0 1
```

Use camera-to-world 4x4 matrices when the source dataset provides a reliable
pose. This matches the key expected by PI3 training views: `camera_pose`.

## PI3 View Mapping

The local loader keeps the validation metadata but can expose each frame with
PI3-style keys:

| PI3 view key | Source in this format |
| --- | --- |
| `img` | image loaded from `frames.csv:image_path` |
| `depthmap` | optional depth loaded from `frames.csv:depth_path` |
| `camera_intrinsics` | `camera.json:K` |
| `camera_pose` | optional matrix from `poses.txt` |
| `dataset` | fixed value `converted_pi3` |
| `label` | dataset root directory name |
| `instance` | `frames.csv:frame_id` |

For conversion validation, depth and pose are optional. For PI3 training,
official `BaseDataset` subclasses generally expect depth to be present so that
`pts3d` and `valid_mask` can be derived from `depthmap` and
`camera_intrinsics`.

## Validate A Converted Dataset

Fast structural check:

```bash
python scripts/pi3_dataset_loader.py <converted_sequence>
```

Full image readability and shape check:

```bash
python scripts/pi3_dataset_loader.py <converted_sequence> --check-images
```

The command exits with code `1` on validation failure and prints actionable
messages, so future conversion scripts can call it as a post-conversion gate.

## Python Usage

```python
from scripts.pi3_dataset_loader import load_pi3_dataset

dataset = load_pi3_dataset("converted_sequence", load_images=True)
sample = dataset[0]
image = sample["img"]
timestamp = sample["timestamp"]
camera_intrinsics = sample["camera_intrinsics"]

view = dataset.as_pi3_view(0)
```
