import argparse
import io
import pathlib
import re
import struct
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
            "are T_WC Sim3 and column 15 is the pose index."
        ),
    )
    parser.add_argument(
        "--pose-index-mode",
        choices=["h5_key", "frame_id"],
        default="h5_key",
        help=(
            "How to interpret column 15 in --pose-file. Use h5_key for "
            "main_global_optimization.py outputs and frame_id for online main.py outputs."
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
        help="Config file. When supplied with --calib, export matches check_h5.py calibration handling.",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="Calibration file used to reconstruct calibrated pointmaps like check_h5.py.",
    )
    parser.add_argument(
        "--match-check-h5-window",
        action="store_true",
        help=(
            "Export the same local trajectory segment selected by check_h5.py: "
            "frames near --frame-id over a pose-distance window."
        ),
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=240,
        help="Reference H5 key used by --match-check-h5-window, matching check_h5.py.",
    )
    parser.add_argument(
        "--nearby-key-window",
        type=int,
        default=10,
        help="Half-window of H5 keys around --frame-id used by --match-check-h5-window.",
    )
    parser.add_argument(
        "--nearby-distance",
        type=float,
        default=30.0,
        help="Pose distance threshold used by --match-check-h5-window.",
    )
    parser.add_argument(
        "--match-surfelmap",
        action="store_true",
        help=(
            "Match the default check_h5.py surfelmap.glsl visibility filter: "
            "skip the 1-pixel image border before exporting points."
        ),
    )
    parser.add_argument(
        "--match-trianglemap",
        action="store_true",
        help=(
            "Match trianglemap.glsl visibility filtering: skip the 10-pixel border, "
            "require top-left quad confidence, and apply the slant threshold."
        ),
    )
    parser.add_argument(
        "--slant-threshold",
        type=float,
        default=0.1,
        help="Slant threshold used by --match-trianglemap, matching trianglemap.glsl.",
    )
    parser.add_argument(
        "--max-world-abs",
        type=float,
        default=0.0,
        help=(
            "Drop transformed world points whose absolute coordinate exceeds this value. "
            "Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--export-cameras",
        action="store_true",
        help="Also export camera frustums and trajectory tubes to <output_stem>_cameras.ply.",
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


def load_calibration_context(config_path: Optional[str], calib_path: Optional[str]) -> Optional[np.ndarray]:
    if config_path is None or calib_path is None:
        return None
    try:
        import yaml
        from mast3r_fusion.config import config, load_config
        from mast3r_fusion.dataloader import Intrinsics
    except ImportError as exc:
        raise ImportError(
            "--config/--calib export requires the MASt3R-Fusion runtime dependencies."
        ) from exc

    load_config(config_path)
    # check_h5.py forces calibrated visualization after loading the config.
    config["use_calib"] = True
    img_size = config.get("dataset", {}).get("img_size", 512)
    with open(calib_path, "r") as f:
        intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
    camera_intrinsics = Intrinsics.from_calib(
        img_size,
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["calibration"],
        False,
        intrinsics.get("model", "pinhole"),
        intrinsics.get("scale", 1),
        intrinsics.get("height_new", None),
    )
    return camera_intrinsics.K_frame.astype(np.float32)


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
    return points.astype(np.float64, copy=False) @ rotation.T + translation


def pose_rotation_translation(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pose = normalize_pose(pose)
    return quaternion_to_matrix(pose[3:7]), pose[:3].astype(np.float64, copy=False)


def pose_scale(pose: np.ndarray) -> float:
    return float(normalize_pose(pose)[7])


def average_conf(data) -> np.ndarray:
    conf = to_numpy(data["C"], dtype=np.float32).reshape(-1)
    n_updates = data.get("N", 1)
    if hasattr(n_updates, "detach") and hasattr(n_updates, "cpu"):
        n_updates = int(n_updates.detach().cpu().reshape(-1)[0])
    else:
        n_updates = int(np.asarray(n_updates).reshape(-1)[0])
    n_updates = max(n_updates, 1)
    return conf / n_updates


def calibrated_points_from_depth(points: np.ndarray, image_shape: Tuple[int, int], K: np.ndarray) -> np.ndarray:
    h, w = image_shape
    points = points.reshape(h, w, 3)
    y, x = np.indices((h, w), dtype=np.float32)
    z = points[..., 2:3]
    rays = np.empty((h, w, 3), dtype=np.float32)
    rays[..., 0] = (x - K[0, 2]) / K[0, 0]
    rays[..., 1] = (y - K[1, 2]) / K[1, 1]
    rays[..., 2] = 1.0
    return (z * rays).reshape(-1, 3)


def frame_arrays(data, K: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    points = to_numpy(data["X"], dtype=np.float32).reshape(-1, 3)
    colors = to_numpy(data["uimg"])
    h, w = colors.shape[:2]
    if K is not None:
        points = calibrated_points_from_depth(points, (h, w), K)
    conf = average_conf(data)
    if colors.dtype != np.uint8:
        colors = np.clip(colors.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    colors = colors.reshape(-1, 3)
    if points.shape[0] != conf.shape[0] or points.shape[0] != colors.shape[0]:
        raise ValueError(
            "Frame arrays have inconsistent lengths: "
            f"X={points.shape[0]}, C={conf.shape[0]}, uimg={colors.shape[0]}."
        )
    return points, colors, conf, (h, w)


def surfelmap_mask(image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    y, x = np.indices((h, w), dtype=np.int32)
    return (x >= 1) & (x < w - 1) & (y >= 1) & (y < h - 1)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def trianglemap_mask(
    points: np.ndarray,
    conf: np.ndarray,
    image_shape: Tuple[int, int],
    conf_threshold: float,
    slant_threshold: float,
) -> np.ndarray:
    h, w = image_shape
    point_grid = points.reshape(h, w, 3)
    conf_grid = conf.reshape(h, w)
    visible = np.zeros((h, w), dtype=bool)
    if h <= 20 or w <= 20:
        return visible

    tl = point_grid[:-1, :-1]
    tr = point_grid[:-1, 1:]
    bl = point_grid[1:, :-1]
    br = point_grid[1:, 1:]
    n1 = normalize_vectors(np.cross(bl - tl, tr - tl))
    n2 = normalize_vectors(np.cross(bl - tr, br - tr))
    ray1 = normalize_vectors(tl)
    ray2 = normalize_vectors(tr)

    finite_quad = (
        np.isfinite(tl).all(axis=-1)
        & np.isfinite(tr).all(axis=-1)
        & np.isfinite(bl).all(axis=-1)
        & np.isfinite(br).all(axis=-1)
    )
    valid_quad = (
        finite_quad
        & np.isfinite(conf_grid[:-1, :-1])
        & (conf_grid[:-1, :-1] >= conf_threshold)
        & (np.abs(np.sum(n1 * ray1, axis=-1)) >= slant_threshold)
        & (np.abs(np.sum(n2 * ray2, axis=-1)) >= slant_threshold)
    )
    border = np.zeros_like(valid_quad, dtype=bool)
    border[10 : h - 10, 10 : w - 10] = True
    valid_quad &= border

    visible[:-1, :-1] |= valid_quad
    visible[:-1, 1:] |= valid_quad
    visible[1:, :-1] |= valid_quad
    visible[1:, 1:] |= valid_quad
    return visible


def valid_mask(
    points: np.ndarray,
    conf: np.ndarray,
    image_shape: Tuple[int, int],
    conf_threshold: float,
    min_depth: float,
    max_depth: float,
    match_surfelmap: bool,
    match_trianglemap: bool,
    slant_threshold: float,
) -> np.ndarray:
    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > conf_threshold)
    if min_depth >= 0:
        mask &= points[:, 2] >= min_depth
    if max_depth > 0:
        mask &= points[:, 2] <= max_depth
    if match_surfelmap:
        mask &= surfelmap_mask(image_shape).reshape(-1)
    if match_trianglemap:
        mask &= trianglemap_mask(
            points, conf, image_shape, conf_threshold, slant_threshold
        ).reshape(-1)
    return mask


def pack_vertices(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    vertices = np.empty(points.shape[0], dtype=PLY_DTYPE)
    if points.shape[0] > 0:
        vertices["x"], vertices["y"], vertices["z"] = points.astype(np.float32).T
        vertices["red"], vertices["green"], vertices["blue"] = colors.astype(np.uint8).T
    return vertices


def pack_camera_vertices(points: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    colors = np.empty((points.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]
    return pack_vertices(points, colors)


def write_ply_header(fp, vertex_count: int, face_count: int = 0) -> None:
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
    )
    if face_count > 0:
        header += (
            f"element face {face_count}\n"
            "property list uchar int vertex_indices\n"
        )
    header += "end_header\n"
    fp.write(header.encode("ascii"))


def write_faces(fp, faces: List[Tuple[int, int, int]]) -> None:
    for face in faces:
        fp.write(struct.pack("<Biii", 3, face[0], face[1], face[2]))


def camera_output_path(output_path: pathlib.Path) -> pathlib.Path:
    return output_path.with_name(f"{output_path.stem}_cameras.ply")


def write_camera_mesh_ply(
    path: pathlib.Path,
    vertices: np.ndarray,
    faces: List[Tuple[int, int, int]],
) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    print(
        f"Writing {vertices.shape[0]} camera/trajectory vertices and "
        f"{len(faces)} faces to {path}."
    )
    with open(path, "wb") as fp:
        write_ply_header(fp, vertices.shape[0], len(faces))
        vertices.tofile(fp)
        write_faces(fp, faces)


def camera_local_vertices(image_shape: Tuple[int, int], scale: float) -> np.ndarray:
    h, w = image_shape
    aspect = max(float(w) / max(float(h), 1.0), 1e-6)
    half_w = scale * aspect
    half_h = scale
    depth = scale * 1.5
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )


def image_shape_from_data(data) -> Tuple[int, int]:
    if "uimg" in data:
        image = to_numpy(data["uimg"])
        return int(image.shape[0]), int(image.shape[1])
    if "img_shape" in data:
        shape = to_numpy(data["img_shape"]).reshape(-1)
        return int(shape[0]), int(shape[1])
    points = to_numpy(data["X"])
    if points.ndim >= 3:
        return int(points.shape[-3]), int(points.shape[-2])
    raise ValueError("Cannot infer image shape for camera frustum export.")


def auto_camera_scale(centers: np.ndarray) -> float:
    if centers.shape[0] == 0:
        return 0.05
    extent = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
    scale_from_extent = extent * 0.008 if np.isfinite(extent) else 0.0
    if centers.shape[0] > 1:
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        finite_steps = steps[np.isfinite(steps) & (steps > 0)]
        scale_from_step = np.median(finite_steps) * 0.2 if finite_steps.size else 0.0
    else:
        scale_from_step = 0.0
    return float(max(scale_from_extent, scale_from_step, 0.05))


def append_tube_segment(
    vertices: List[np.ndarray],
    faces: List[Tuple[int, int, int]],
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    color: Tuple[int, int, int],
    sides: int = 8,
) -> None:
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-12:
        return
    direction = axis / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring_offsets = radius * (
        np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :]
    )
    tube_points = np.concatenate([p0[None, :] + ring_offsets, p1[None, :] + ring_offsets])
    base = sum(chunk.shape[0] for chunk in vertices)
    vertices.append(pack_camera_vertices(tube_points, color))
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((base + i, base + j, base + sides + i))
        faces.append((base + j, base + sides + j, base + sides + i))


def append_camera_geometry(
    vertices: List[np.ndarray],
    faces: List[Tuple[int, int, int]],
    pose: np.ndarray,
    image_shape: Tuple[int, int],
    scale: float,
    tube_radius: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    rotation, translation = pose_rotation_translation(pose)
    local = camera_local_vertices(image_shape, scale)
    world = local @ rotation.T + translation
    for i, j in (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ):
        append_tube_segment(vertices, faces, world[i], world[j], tube_radius, color)
    return world[0]


def collect_camera_geometry(
    h5_file,
    keys: List[Tuple[int, str]],
    pose_by_index,
    args,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    frame_infos = []
    centers = []
    for key_idx, key in keys:
        data = load_frame(h5_file, key)
        pose = resolve_pose(data, key_idx, pose_by_index, args.pose_index_mode)
        image_shape = image_shape_from_data(data)
        _, center = pose_rotation_translation(pose)
        frame_infos.append((pose, image_shape))
        centers.append(center)

    centers_array = np.asarray(centers, dtype=np.float64)
    camera_scale = auto_camera_scale(centers_array)
    frustum_radius = camera_scale * 0.035
    trajectory_radius = camera_scale * 0.025
    vertices: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    previous_center: Optional[np.ndarray] = None
    for order, (pose, image_shape) in enumerate(frame_infos):
        color = (255, 128, 0) if order == 0 else (0, 180, 255)
        center = append_camera_geometry(
            vertices,
            faces,
            pose,
            image_shape,
            camera_scale,
            frustum_radius,
            color,
        )
        if previous_center is not None:
            append_tube_segment(
                vertices,
                faces,
                previous_center,
                center,
                trajectory_radius,
                (255, 64, 64),
            )
        previous_center = center
    vertex_count = sum(chunk.shape[0] for chunk in vertices)
    camera_vertices = np.concatenate(vertices) if vertices else np.empty(0, dtype=PLY_DTYPE)
    print(
        f"Prepared {vertex_count} camera/trajectory mesh vertices and {len(faces)} faces "
        f"(camera scale {camera_scale:.6g})."
    )
    return camera_vertices, faces


def resolve_pose(
    data,
    key_idx: int,
    pose_by_index: Dict[int, np.ndarray],
    pose_index_mode: str,
) -> np.ndarray:
    frame_id = frame_id_from_data(data, key_idx)
    primary_idx = key_idx if pose_index_mode == "h5_key" else frame_id
    fallback_idx = frame_id if pose_index_mode == "h5_key" else key_idx
    if primary_idx in pose_by_index:
        return pose_by_index[primary_idx]
    if fallback_idx in pose_by_index:
        return pose_by_index[fallback_idx]
    return normalize_pose(data["T_WC"])


def transformed_frame_arrays(
    data,
    key_idx: int,
    pose_by_index,
    args,
    K: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pose = resolve_pose(data, key_idx, pose_by_index, args.pose_index_mode)
    points, colors, conf, image_shape = frame_arrays(data, K)
    # check_h5.py multiplies X by the Sim3 scale, then sets pose scale to 1.0.
    points = points * pose_scale(pose)
    mask = valid_mask(
        points,
        conf,
        image_shape,
        args.conf_threshold,
        args.min_depth,
        args.max_depth,
        args.match_surfelmap,
        args.match_trianglemap,
        args.slant_threshold,
    )
    if args.stride > 1:
        points = points[:: args.stride]
        colors = colors[:: args.stride]
        mask = mask[:: args.stride]
    points = points[mask]
    colors = colors[mask]
    points_world = pose_to_world(points, pose)
    world_mask = np.isfinite(points_world).all(axis=1)
    if args.max_world_abs > 0:
        world_mask &= np.max(np.abs(points_world), axis=1) <= args.max_world_abs
    return points_world[world_mask], colors[world_mask]


def count_frame_points(data, key_idx: int, pose_by_index, args, K: Optional[np.ndarray]) -> int:
    points_world, _ = transformed_frame_arrays(data, key_idx, pose_by_index, args, K)
    return int(points_world.shape[0])


def transformed_frame_vertices(data, key_idx: int, pose_by_index, args, K: Optional[np.ndarray]) -> np.ndarray:
    points_world, colors = transformed_frame_arrays(data, key_idx, pose_by_index, args, K)
    return pack_vertices(points_world, colors)


def update_bounds(
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    points: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return bounds
    cur_min = points.min(axis=0)
    cur_max = points.max(axis=0)
    if bounds is None:
        return cur_min, cur_max
    return np.minimum(bounds[0], cur_min), np.maximum(bounds[1], cur_max)


def iter_frame_data(h5_file, keys: Iterable[Tuple[int, str]]):
    for key_idx, key in keys:
        yield key_idx, key, load_frame(h5_file, key)


def pose_by_key(h5_file, keys: Iterable[Tuple[int, str]], pose_by_index, args) -> Dict[int, np.ndarray]:
    poses = {}
    for key_idx, key, data in iter_frame_data(h5_file, keys):
        poses[key_idx] = resolve_pose(data, key_idx, pose_by_index, args.pose_index_mode)
    return poses


def filter_check_h5_window(
    keys: List[Tuple[int, str]],
    poses: Dict[int, np.ndarray],
    frame_id: int,
    nearby_key_window: int,
    nearby_distance: float,
) -> List[Tuple[int, str]]:
    ref_indices = [
        idx
        for idx in range(frame_id - nearby_key_window, frame_id + nearby_key_window)
        if idx in poses
    ]
    if not ref_indices:
        raise ValueError(
            "No reference poses found for check_h5 window "
            f"[{frame_id - nearby_key_window}, {frame_id + nearby_key_window})."
        )
    selected = []
    ref_positions = np.array([poses[idx][:3] for idx in ref_indices], dtype=np.float64)
    for key_idx, key in keys:
        if key_idx not in poses:
            continue
        distances = np.linalg.norm(ref_positions - poses[key_idx][:3], axis=1)
        if np.any(distances < nearby_distance):
            selected.append((key_idx, key))
    return selected


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.match_surfelmap and args.match_trianglemap:
        raise ValueError("--match-surfelmap and --match-trianglemap are mutually exclusive.")

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read data.h5. Install it in the runtime environment "
            "or use the same environment that runs main.py --save_h5."
        ) from exc

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    pose_by_index = load_pose_file(args.pose_file, args.prefer_keyframe_poses)
    K = load_calibration_context(args.config, args.calib)

    with h5py.File(args.h5, "r") as h5_file:
        keys = frame_keys(h5_file, args.start_key, args.end_key)
        if not keys:
            raise ValueError(f"No frame_* datasets found in {args.h5}.")
        poses = pose_by_key(h5_file, keys, pose_by_index, args)
        if args.match_check_h5_window:
            keys = filter_check_h5_window(
                keys,
                poses,
                args.frame_id,
                args.nearby_key_window,
                args.nearby_distance,
            )

        print(f"Found {len(keys)} H5 keyframes.")
        if pose_by_index:
            print(
                f"Loaded {len(pose_by_index)} poses from {args.pose_file} "
                f"using {args.pose_index_mode} indexing."
            )
        if K is not None:
            print("Using calibrated ray projection to match check_h5.py.")

        total_vertices = 0
        for key_idx, _, data in iter_frame_data(h5_file, keys):
            total_vertices += count_frame_points(data, key_idx, pose_by_index, args, K)

        camera_vertices = np.empty(0, dtype=PLY_DTYPE)
        camera_faces: List[Tuple[int, int, int]] = []
        if args.export_cameras:
            camera_vertices, camera_faces = collect_camera_geometry(
                h5_file,
                keys,
                pose_by_index,
                args,
            )

        print(f"Writing {total_vertices} points to {output_path}.")

        bounds = None
        with open(output_path, "wb") as fp:
            write_ply_header(fp, total_vertices)
            for key_idx, _, data in iter_frame_data(h5_file, keys):
                points_world, colors = transformed_frame_arrays(
                    data, key_idx, pose_by_index, args, K
                )
                bounds = update_bounds(bounds, points_world)
                pack_vertices(points_world, colors).tofile(fp)

        if bounds is None:
            print("No finite world points were exported.")
        else:
            print(
                "World bounds: "
                f"min=({bounds[0][0]:.6g}, {bounds[0][1]:.6g}, {bounds[0][2]:.6g}), "
                f"max=({bounds[1][0]:.6g}, {bounds[1][1]:.6g}, {bounds[1][2]:.6g})"
            )
        if args.export_cameras:
            write_camera_mesh_ply(
                camera_output_path(output_path),
                camera_vertices,
                camera_faces,
            )

    print("Done.")


if __name__ == "__main__":
    main()
