import argparse
import io
import pathlib
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


FRAME_KEY_RE = re.compile(r"^frame_(\d+)$")
PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all keyframe pointmaps stored in data.h5 to a binary PLY."
    )
    parser.add_argument("--h5", required=True, help="Input H5 file produced by main.py --save_h5.")
    parser.add_argument("--output", required=True, help="Output .ply path.")
    parser.add_argument(
        "--pose-file",
        default=None,
        help=(
            "Optional result pose file. For repository result files, columns 1:9 "
            "are T_WC Sim3 and column 15 is the original frame id."
        ),
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=1.5,
        help="Keep points with average confidence C / N above this threshold.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=15.0,
        help=(
            "Drop points whose local z depth is above this value, matching "
            "check_h5.py's default visualization clamp. Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Drop points whose local z depth is below this value. Use < 0 to disable.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Optional pixel/point stride for lighter exports. Default exports every point.",
    )
    parser.add_argument(
        "--start-key",
        type=int,
        default=None,
        help="Optional minimum H5 key index, e.g. frame_100.",
    )
    parser.add_argument(
        "--end-key",
        type=int,
        default=None,
        help="Optional maximum H5 key index, inclusive.",
    )
    parser.add_argument(
        "--prefer-keyframe-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --pose-file has both tracking and keyframe rows for the same frame id, "
            "prefer rows whose final flag column is 1."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for command-line parity with check_h5.py; not needed for export.",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="Accepted for command-line parity with check_h5.py; not needed for export.",
    )
    return parser.parse_args()


def frame_keys(h5_file, start_key: Optional[int], end_key: Optional[int]) -> List[Tuple[int, str]]:
    keys = []
    for key in h5_file.keys():
        match = FRAME_KEY_RE.match(key)
        if match is None:
            continue
        idx = int(match.group(1))
        if start_key is not None and idx < start_key:
            continue
        if end_key is not None and idx > end_key:
            continue
        keys.append((idx, key))
    keys.sort(key=lambda item: item[0])
    return keys


def torch_load_h5_blob(blob):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to deserialize frame_* blobs from data.h5. "
            "Use the same environment that runs MASt3R-Fusion."
        ) from exc
    buffer = io.BytesIO(bytes(blob))
    try:
        return torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


def load_frame(h5_file, key: str):
    return torch_load_h5_blob(h5_file[key][()])


def to_numpy(value, dtype=None):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def frame_id_from_data(data, key_idx: int) -> int:
    if "id" not in data:
        return key_idx
    frame_id = data["id"]
    if hasattr(frame_id, "detach") and hasattr(frame_id, "cpu"):
        frame_id = frame_id.detach().cpu().numpy()
    return int(np.asarray(frame_id).reshape(-1)[0])


def normalize_pose(pose) -> np.ndarray:
    pose = to_numpy(pose, dtype=np.float64).reshape(-1)
    if pose.size < 7:
        raise ValueError(f"Pose must have at least 7 values, got {pose.size}.")
    if pose.size == 7:
        pose = np.concatenate([pose, np.array([1.0], dtype=np.float64)])
    else:
        pose = pose[:8].copy()
    q_norm = np.linalg.norm(pose[3:7])
    if q_norm <= 0:
        raise ValueError("Pose quaternion has zero norm.")
    pose[3:7] /= q_norm
    return pose


def load_pose_file(path: Optional[str], prefer_keyframe_poses: bool) -> Dict[int, np.ndarray]:
    if path is None:
        return {}
    pose_path = pathlib.Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file does not exist: {pose_path}")
    rows = np.loadtxt(pose_path)
    rows = np.atleast_2d(rows)
    pose_by_frame_id: Dict[int, np.ndarray] = {}
    is_keyframe_by_frame_id: Dict[int, bool] = {}
    for row in rows:
        if row.size < 16:
            raise ValueError(
                "--pose-file must use the repository result format with at least 16 columns."
            )
        frame_id = int(row[15])
        is_keyframe = bool(row.size >= 17 and row[16] == 1)
        if prefer_keyframe_poses:
            existing_is_keyframe = is_keyframe_by_frame_id.get(frame_id, False)
            if existing_is_keyframe and not is_keyframe:
                continue
            if (not existing_is_keyframe) or is_keyframe:
                pose_by_frame_id[frame_id] = normalize_pose(row[1:9])
                is_keyframe_by_frame_id[frame_id] = is_keyframe
        else:
            pose_by_frame_id[frame_id] = normalize_pose(row[1:9])
            is_keyframe_by_frame_id[frame_id] = is_keyframe
    return pose_by_frame_id


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = normalize_pose(pose)
    translation = pose[:3]
    rotation = quaternion_to_matrix(pose[3:7])
    scale = pose[7]
    # check_h5.py scales X by Sim3 scale and then visualizes with scale set to 1.
    return (points.astype(np.float64, copy=False) * scale) @ rotation.T + translation


def average_conf(data) -> np.ndarray:
    conf = to_numpy(data["C"], dtype=np.float32).reshape(-1)
    n_updates = data.get("N", 1)
    if hasattr(n_updates, "detach") and hasattr(n_updates, "cpu"):
        n_updates = int(n_updates.detach().cpu().reshape(-1)[0])
    else:
        n_updates = int(np.asarray(n_updates).reshape(-1)[0])
    n_updates = max(n_updates, 1)
    return conf / n_updates


def frame_arrays(data, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = to_numpy(data["X"], dtype=np.float32).reshape(-1, 3)
    conf = average_conf(data)
    colors = to_numpy(data["uimg"])
    if colors.dtype != np.uint8:
        colors = np.clip(colors.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    colors = colors.reshape(-1, 3)
    if points.shape[0] != conf.shape[0] or points.shape[0] != colors.shape[0]:
        raise ValueError(
            "Frame arrays have inconsistent lengths: "
            f"X={points.shape[0]}, C={conf.shape[0]}, uimg={colors.shape[0]}."
        )
    if stride > 1:
        points = points[::stride]
        colors = colors[::stride]
        conf = conf[::stride]
    return points, colors, conf


def valid_mask(
    points: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > conf_threshold)
    if min_depth >= 0:
        mask &= points[:, 2] >= min_depth
    if max_depth > 0:
        mask &= points[:, 2] <= max_depth
    return mask


def pack_vertices(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    vertices = np.empty(points.shape[0], dtype=PLY_DTYPE)
    if points.shape[0] > 0:
        vertices["x"], vertices["y"], vertices["z"] = points.astype(np.float32).T
        vertices["red"], vertices["green"], vertices["blue"] = colors.astype(np.uint8).T
    return vertices


def write_ply_header(fp, vertex_count: int) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    fp.write(header.encode("ascii"))


def resolve_pose(data, key_idx: int, pose_by_frame_id: Dict[int, np.ndarray]) -> np.ndarray:
    frame_id = frame_id_from_data(data, key_idx)
    if frame_id in pose_by_frame_id:
        return pose_by_frame_id[frame_id]
    if key_idx in pose_by_frame_id:
        return pose_by_frame_id[key_idx]
    return normalize_pose(data["T_WC"])


def count_frame_points(data, args) -> int:
    points, _, conf = frame_arrays(data, args.stride)
    mask = valid_mask(points, conf, args.conf_threshold, args.min_depth, args.max_depth)
    return int(mask.sum())


def transformed_frame_vertices(data, key_idx: int, pose_by_frame_id, args) -> np.ndarray:
    points, colors, conf = frame_arrays(data, args.stride)
    mask = valid_mask(points, conf, args.conf_threshold, args.min_depth, args.max_depth)
    points = points[mask]
    colors = colors[mask]
    pose = resolve_pose(data, key_idx, pose_by_frame_id)
    points_world = pose_to_world(points, pose)
    return pack_vertices(points_world, colors)


def iter_frame_data(h5_file, keys: Iterable[Tuple[int, str]]):
    for key_idx, key in keys:
        yield key_idx, key, load_frame(h5_file, key)


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read data.h5. Install it in the runtime environment "
            "or use the same environment that runs main.py --save_h5."
        ) from exc

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    pose_by_frame_id = load_pose_file(args.pose_file, args.prefer_keyframe_poses)

    with h5py.File(args.h5, "r") as h5_file:
        keys = frame_keys(h5_file, args.start_key, args.end_key)
        if not keys:
            raise ValueError(f"No frame_* datasets found in {args.h5}.")

        print(f"Found {len(keys)} H5 keyframes.")
        if pose_by_frame_id:
            print(f"Loaded {len(pose_by_frame_id)} poses from {args.pose_file}.")

        total_vertices = 0
        for _, _, data in iter_frame_data(h5_file, keys):
            total_vertices += count_frame_points(data, args)
        print(f"Writing {total_vertices} points to {output_path}.")

        with open(output_path, "wb") as fp:
            write_ply_header(fp, total_vertices)
            for key_idx, _, data in iter_frame_data(h5_file, keys):
                transformed_frame_vertices(data, key_idx, pose_by_frame_id, args).tofile(fp)

    print("Done.")


if __name__ == "__main__":
    main()
