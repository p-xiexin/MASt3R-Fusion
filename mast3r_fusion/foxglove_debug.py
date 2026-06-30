import asyncio
import base64
import json
import logging
import queue
import time

import cv2
import lietorch
import numpy as np
import torch
import yaml
from foxglove_websocket.server import FoxgloveServer
from scipy.spatial.transform import Rotation

from mast3r_fusion.config import config, set_global_config
from mast3r_fusion.frame import Mode


POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "pose": {"$ref": "#/$defs/Pose"},
    },
    "required": ["timestamp", "frame_id", "pose"],
    "$defs": {
        "Time": {
            "type": "object",
            "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"},
            },
            "required": ["sec", "nsec"],
        },
        "Pose": {
            "type": "object",
            "properties": {
                "position": {"$ref": "#/$defs/Vector3"},
                "orientation": {"$ref": "#/$defs/Quaternion"},
            },
            "required": ["position", "orientation"],
        },
        "Vector3": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y", "z"],
        },
        "Quaternion": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
            "required": ["x", "y", "z", "w"],
        },
    },
}

POSES_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "poses": {"type": "array", "items": {"$ref": "#/$defs/Pose"}},
    },
    "required": ["timestamp", "frame_id", "poses"],
    "$defs": POSE_SCHEMA["$defs"],
}

POINT_CLOUD_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "pose": {"$ref": "#/$defs/Pose"},
        "point_stride": {"type": "integer"},
        "fields": {"type": "array", "items": {"$ref": "#/$defs/PackedElementField"}},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
    "required": ["timestamp", "frame_id", "pose", "point_stride", "fields", "data"],
    "$defs": {
        **POSE_SCHEMA["$defs"],
        "PackedElementField": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "offset": {"type": "integer"},
                "type": {"type": "integer"},
            },
            "required": ["name", "offset", "type"],
        },
    },
}

COMPRESSED_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "format": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
    "required": ["timestamp", "frame_id", "format", "data"],
    "$defs": POSE_SCHEMA["$defs"],
}

SCENE_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "deletions": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"$ref": "#/$defs/SceneEntity"}},
    },
    "required": ["deletions", "entities"],
    "$defs": {
        **POSE_SCHEMA["$defs"],
        "Color": {
            "type": "object",
            "properties": {
                "r": {"type": "number"},
                "g": {"type": "number"},
                "b": {"type": "number"},
                "a": {"type": "number"},
            },
            "required": ["r", "g", "b", "a"],
        },
        "Duration": {
            "type": "object",
            "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"},
            },
            "required": ["sec", "nsec"],
        },
        "KeyValuePair": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
        "LinePrimitive": {
            "type": "object",
            "properties": {
                "pose": {"$ref": "#/$defs/Pose"},
                "thickness": {"type": "number"},
                "scale_invariant": {"type": "boolean"},
                "points": {"type": "array", "items": {"$ref": "#/$defs/Vector3"}},
                "color": {"$ref": "#/$defs/Color"},
                "colors": {"type": "array", "items": {"$ref": "#/$defs/Color"}},
                "indices": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["pose", "thickness", "scale_invariant", "points", "color", "colors", "indices"],
        },
        "SceneEntity": {
            "type": "object",
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "frame_id": {"type": "string"},
                "id": {"type": "string"},
                "lifetime": {"$ref": "#/$defs/Duration"},
                "frame_locked": {"type": "boolean"},
                "metadata": {"type": "array", "items": {"$ref": "#/$defs/KeyValuePair"}},
                "arrows": {"type": "array", "items": {"type": "object"}},
                "cubes": {"type": "array", "items": {"type": "object"}},
                "spheres": {"type": "array", "items": {"type": "object"}},
                "cylinders": {"type": "array", "items": {"type": "object"}},
                "lines": {"type": "array", "items": {"$ref": "#/$defs/LinePrimitive"}},
                "triangles": {"type": "array", "items": {"type": "object"}},
                "texts": {"type": "array", "items": {"type": "object"}},
                "models": {"type": "array", "items": {"type": "object"}},
            },
            "required": [
                "timestamp",
                "frame_id",
                "id",
                "lifetime",
                "frame_locked",
                "metadata",
                "arrows",
                "cubes",
                "spheres",
                "cylinders",
                "lines",
                "triangles",
                "texts",
                "models",
            ],
        },
    },
}

FRAME_TRANSFORMS_SCHEMA = {
    "type": "object",
    "properties": {
        "transforms": {"type": "array", "items": {"$ref": "#/$defs/FrameTransform"}},
    },
    "required": ["transforms"],
    "$defs": {
        **POSE_SCHEMA["$defs"],
        "FrameTransform": {
            "type": "object",
            "properties": {
                "timestamp": {"$ref": "#/$defs/Time"},
                "parent_frame_id": {"type": "string"},
                "child_frame_id": {"type": "string"},
                "translation": {"$ref": "#/$defs/Vector3"},
                "rotation": {"$ref": "#/$defs/Quaternion"},
            },
            "required": ["timestamp", "parent_frame_id", "child_frame_id", "translation", "rotation"],
        },
    },
}

CHANNEL_DEFS = [
    ("tf", "/tf", "foxglove.FrameTransforms", FRAME_TRANSFORMS_SCHEMA),
    ("current_pose", "/current_pose", "foxglove.PoseInFrame", POSE_SCHEMA),
    ("trajectory", "/trajectory", "foxglove.PosesInFrame", POSES_SCHEMA),
    ("keyframe_points", "/keyframe_points", "foxglove.PointCloud", POINT_CLOUD_SCHEMA),
    ("current_image", "/current_image", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
    ("graph_edges", "/graph_edges", "foxglove.SceneUpdate", SCENE_UPDATE_SCHEMA),
]

# MASt3R-Fusion initializes its world close to x=right, y=forward, z=up.
# Foxglove/ROS displays are easier to inspect as x=forward, y=left, z=up.
R_FOXGLOVE_FROM_SLAM = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
T_FOXGLOVE_FROM_SLAM = np.eye(4, dtype=np.float64)
T_FOXGLOVE_FROM_SLAM[:3, :3] = R_FOXGLOVE_FROM_SLAM


def _time_msg(timestamp_ns):
    return {"sec": timestamp_ns // 1_000_000_000, "nsec": timestamp_ns % 1_000_000_000}


def _identity_pose():
    return {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _pose_from_sim3_data(data):
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(arr[3:7]).as_matrix()
    matrix[:3, 3] = arr[:3]
    return _pose_from_world_matrix(matrix)


def _pose_from_world_matrix(matrix):
    matrix = T_FOXGLOVE_FROM_SLAM @ np.asarray(matrix, dtype=np.float64)
    return _pose_from_matrix(matrix)


def _pose_from_relative_matrix(matrix):
    return _pose_from_matrix(np.asarray(matrix, dtype=np.float64))


def _pose_from_matrix(matrix):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": {
            "x": float(matrix[0, 3]),
            "y": float(matrix[1, 3]),
            "z": float(matrix[2, 3]),
        },
        "orientation": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def _frame_transforms_msg(timestamp_ns, transforms):
    return {
        "transforms": [
            {
                "timestamp": _time_msg(timestamp_ns),
                "parent_frame_id": parent_frame_id,
                "child_frame_id": child_frame_id,
                "translation": pose["position"],
                "rotation": pose["orientation"],
            }
            for parent_frame_id, child_frame_id, pose in transforms
        ]
    }


def _frame_matrix(frame):
    return frame.T_WC.matrix()[0].detach().cpu().numpy().astype(np.float32)


def _translation_from_sim3_data(data):
    matrix = lietorch.Sim3(data.reshape(1, -1)).matrix().detach().cpu().numpy()
    return R_FOXGLOVE_FROM_SLAM @ matrix[0, :3, 3]


def _camera_points(frame):
    if frame.X_canon is None or frame.C is None:
        return None

    if config.get("use_calib") and frame.K is not None:
        xyz = frame.X_canon.detach()
        shape = frame.img_shape.flatten()[:2].detach().cpu().numpy()
        height, width = int(shape[0]), int(shape[1])
        ys, xs = torch.meshgrid(
            torch.arange(height, device=xyz.device, dtype=xyz.dtype),
            torch.arange(width, device=xyz.device, dtype=xyz.dtype),
            indexing="ij",
        )
        K = frame.K.to(device=xyz.device, dtype=xyz.dtype)
        z = xyz[:, 2].reshape(height, width)
        x = (xs - K[0, 2]) / K[0, 0] * z
        y = (ys - K[1, 2]) / K[1, 1] * z
        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    else:
        points = frame.X_canon.detach().reshape(-1, 3)

    if getattr(frame, "N", 0) > 0:
        conf = frame.get_average_conf()
    else:
        conf = frame.C
    conf = conf.detach().reshape(-1) if conf is not None else frame.C.detach().reshape(-1)
    colors = frame.uimg.detach().cpu().numpy().reshape(-1, 3)
    return points.cpu().numpy(), conf.cpu().numpy(), colors


def _world_points(frame, conf_threshold, max_points):
    data = _camera_points(frame)
    if data is None:
        return None
    points, conf, colors = data
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf >= conf_threshold)
    points = points[valid]
    colors = colors[valid]
    if points.size == 0:
        return None

    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / max_points))
        points = points[::stride]
        colors = colors[::stride]

    points_h = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_w = (_frame_matrix(frame) @ points_h.T).T[:, :3]
    points_w = points_w @ R_FOXGLOVE_FROM_SLAM.T
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return points_w.astype(np.float32), colors_u8


def _point_cloud_msg(timestamp_ns, points, colors):
    rgb = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    rgba = (rgb << 8) | np.uint32(255)
    packed = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<u4"),
            ("rgba", "<u4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    packed["x"] = points[:, 0]
    packed["y"] = points[:, 1]
    packed["z"] = points[:, 2]
    packed["rgb"] = rgb
    packed["rgba"] = rgba
    packed["red"] = colors[:, 0]
    packed["green"] = colors[:, 1]
    packed["blue"] = colors[:, 2]
    packed["alpha"] = 255
    return {
        "timestamp": _time_msg(timestamp_ns),
        "frame_id": "world",
        "pose": _identity_pose(),
        "point_stride": 24,
        "fields": [
            {"name": "x", "offset": 0, "type": 7},
            {"name": "y", "offset": 4, "type": 7},
            {"name": "z", "offset": 8, "type": 7},
            {"name": "rgb", "offset": 12, "type": 5},
            {"name": "rgba", "offset": 16, "type": 5},
            {"name": "red", "offset": 20, "type": 1},
            {"name": "green", "offset": 21, "type": 1},
            {"name": "blue", "offset": 22, "type": 1},
            {"name": "alpha", "offset": 23, "type": 1},
        ],
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _tracking_overlay_from_states(states):
    with states.lock:
        prev = list(states.track_prev_xy)
        curr = list(states.track_curr_xy)
        ages = list(states.track_ages)
    n = min(len(prev), len(curr)) // 2
    n = min(n, len(ages))
    if n <= 0:
        return None
    prev_xy = np.asarray(prev[: 2 * n], dtype=np.float32).reshape(n, 2)
    curr_xy = np.asarray(curr[: 2 * n], dtype=np.float32).reshape(n, 2)
    ages = np.asarray(ages[:n], dtype=np.int32)
    return prev_xy, curr_xy, ages


def _age_heatmap_colors(ages):
    ages = np.asarray(ages, dtype=np.float32)
    if ages.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    max_age = max(1.0, float(np.percentile(ages, 95)))
    norm = np.clip(ages / max_age, 0.0, 1.0)
    values = np.rint(norm * 255.0).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)


def _draw_tracking_overlay(image_rgb, overlay, old_age_threshold):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if overlay is None:
        return image
    prev_xy, curr_xy, ages = overlay
    h, w = image.shape[:2]
    drawn = 0
    drawn_ages = []
    colors = _age_heatmap_colors(ages)
    for p0, p1, age, color in zip(prev_xy, curr_xy, ages, colors):
        p0 = np.rint(p0).astype(int)
        p1 = np.rint(p1).astype(int)
        if not (0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        color = tuple(int(c) for c in color)
        cv2.circle(image, tuple(p1), 2, color, -1, cv2.LINE_AA)
        if 0 <= p0[0] < w and 0 <= p0[1] < h:
            cv2.line(image, tuple(p0), tuple(p1), color, 1, cv2.LINE_AA)
        drawn += 1
        drawn_ages.append(int(age))
    if drawn:
        drawn_ages = np.asarray(drawn_ages, dtype=np.int32)
        old_count = int((drawn_ages >= old_age_threshold).sum())
        old_ratio = old_count / max(1, drawn)
        age_med = int(np.median(drawn_ages))
        text = f"tracks={drawn} old>={old_age_threshold}:{old_ratio:.0%} age_med={age_med}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.rectangle(image, (8, 8), (24 + text_size[0], 40), (0, 0, 0), -1)
        cv2.putText(image, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _image_msg(timestamp_ns, frame, jpeg_quality, tracking_overlay=None, old_age_threshold=5):
    image_rgb = np.clip(frame.uimg.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    image_bgr = _draw_tracking_overlay(image_rgb, tracking_overlay, old_age_threshold)
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        return None
    data = encoded.tobytes()
    return {
        "timestamp": _time_msg(timestamp_ns),
        "frame_id": "camera",
        "format": "jpeg",
        "data": base64.b64encode(data).decode("ascii"),
    }


def _snapshot_keyframes(keyframes):
    frames = []
    start = keyframes.rollup_sum.value
    for idx in range(start, start + len(keyframes)):
        try:
            frames.append(keyframes[idx])
        except Exception:
            continue
    return frames


async def _add_channels(server):
    channels = {}
    for name, topic, schema_name, schema in CHANNEL_DEFS:
        channels[name] = await server.add_channel(
            {
                "topic": topic,
                "encoding": "json",
                "schemaName": schema_name,
                "schemaEncoding": "jsonschema",
                "schema": json.dumps(schema, separators=(",", ":")),
            }
        )
    return channels


async def _send_json(server, channels, name, timestamp_ns, msg):
    await server.send_message(channels[name], timestamp_ns, json.dumps(msg, separators=(",", ":")).encode("utf8"))


async def _publish_snapshot(server, channels, states, keyframes, options, publish_points=False):
    timestamp_ns = time.time_ns()
    if states.get_mode() != Mode.INIT:
        try:
            current_frame = states.get_frame()
            current_T_WC = _frame_matrix(current_frame).astype(np.float64)
            current_T_WB = current_T_WC @ np.linalg.inv(options["Tic"])
            current_pose = _pose_from_world_matrix(current_T_WC)
            body_pose = _pose_from_world_matrix(current_T_WB)
            body_T_camera = _pose_from_relative_matrix(options["Tic"])
            await _send_json(
                server,
                channels,
                "tf",
                timestamp_ns,
                _frame_transforms_msg(
                    timestamp_ns,
                    [
                        ("world", "body", body_pose),
                        ("body", "camera", body_T_camera),
                    ],
                ),
            )
            await _send_json(
                server,
                channels,
                "current_pose",
                timestamp_ns,
                {
                    "timestamp": _time_msg(timestamp_ns),
                    "frame_id": "world",
                    "pose": current_pose,
                },
            )
            tracking_overlay = _tracking_overlay_from_states(states)
            image = _image_msg(
                timestamp_ns,
                current_frame,
                options["jpeg_quality"],
                tracking_overlay,
                options["old_track_age_threshold"],
            )
            if image is not None:
                await _send_json(server, channels, "current_image", timestamp_ns, image)
        except Exception as exc:
            print(f"[foxglove] current frame publish skipped: {exc}")

    frames = _snapshot_keyframes(keyframes)
    if frames:
        poses = [_pose_from_sim3_data(frame.T_WC.data.cpu().numpy()[0]) for frame in frames]
        await _send_json(
            server,
            channels,
            "trajectory",
            timestamp_ns,
            {"timestamp": _time_msg(timestamp_ns), "frame_id": "world", "poses": poses},
        )

        if publish_points:
            points_accum = []
            colors_accum = []
            per_kf_limit = max(1, options["max_points"] // min(len(frames), options["max_keyframes"]))
            for frame in frames[-options["max_keyframes"] :]:
                result = _world_points(frame, options["conf_threshold"], per_kf_limit)
                if result is None:
                    continue
                points, colors = result
                points_accum.append(points)
                colors_accum.append(colors)
            if points_accum:
                points = np.concatenate(points_accum, axis=0)
                colors = np.concatenate(colors_accum, axis=0)
                if points.shape[0] > options["max_points"]:
                    stride = int(np.ceil(points.shape[0] / options["max_points"]))
                    points = points[::stride]
                    colors = colors[::stride]
                await _send_json(server, channels, "keyframe_points", timestamp_ns, _point_cloud_msg(timestamp_ns, points, colors))
                print(
                    "[foxglove] published keyframe_points "
                    f"points={points.shape[0]} frames={len(frames[-options['max_keyframes']:])}"
                )
            else:
                print("[foxglove] keyframe_points skipped: no valid points")

    await _publish_graph_edges(server, channels, states, keyframes, timestamp_ns)


async def _publish_graph_edges(server, channels, states, keyframes, timestamp_ns):
    try:
        with states.lock:
            ii = list(states.edges_ii)
            jj = list(states.edges_jj)
        if not ii or not jj:
            return
        points = []
        rollup = keyframes.rollup_sum.value
        total = len(keyframes)
        for i, j in zip(ii, jj):
            local_i = i - rollup
            local_j = j - rollup
            if local_i < 0 or local_j < 0 or local_i >= total or local_j >= total:
                continue
            Ti = _translation_from_sim3_data(keyframes.T_WC[local_i, 0])
            Tj = _translation_from_sim3_data(keyframes.T_WC[local_j, 0])
            points.append({"x": float(Ti[0]), "y": float(Ti[1]), "z": float(Ti[2])})
            points.append({"x": float(Tj[0]), "y": float(Tj[1]), "z": float(Tj[2])})
        if not points:
            return
        await _send_json(
            server,
            channels,
            "graph_edges",
            timestamp_ns,
            {
                "deletions": [],
                "entities": [
                    {
                        "timestamp": _time_msg(timestamp_ns),
                        "frame_id": "world",
                        "id": "factor_graph_edges",
                        "lifetime": {"sec": 1, "nsec": 0},
                        "frame_locked": True,
                        "metadata": [],
                        "arrows": [],
                        "cubes": [],
                        "spheres": [],
                        "cylinders": [],
                        "lines": [
                            {
                                "pose": _identity_pose(),
                                "thickness": 0.02,
                                "scale_invariant": True,
                                "points": points,
                                "color": {"r": 0.0, "g": 1.0, "b": 0.2, "a": 1.0},
                                "colors": [],
                                "indices": [],
                            }
                        ],
                        "triangles": [],
                        "texts": [],
                        "models": [],
                    }
                ],
            },
        )
    except Exception as exc:
        print(f"[foxglove] graph edge publish skipped: {exc}")


async def _run_server(cfg, states, keyframes, host, port, publish_hz, options, event_queue):
    set_global_config(cfg)
    logger = logging.getLogger("MASt3R-Fusion Foxglove")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())

    async with FoxgloveServer(
        host,
        port,
        "MASt3R-Fusion debug",
        supported_encodings=["json"],
        logger=logger,
    ) as server:
        await _wait_opened(server)
        channels = await _add_channels(server)
        print(f"[foxglove] listening on ws://{host}:{port}")
        while states.get_mode() != Mode.TERMINATED:
            event = await asyncio.to_thread(event_queue.get)
            terminate, publish_points = _coalesce_events(event_queue, event)
            if terminate:
                break
            await _publish_snapshot(
                server,
                channels,
                states,
                keyframes,
                options,
                publish_points=publish_points,
            )


def _coalesce_events(event_queue, first_event):
    terminate = first_event == "terminate"
    publish_points = first_event == "pointcloud"
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        terminate = terminate or event == "terminate"
        publish_points = publish_points or event == "pointcloud"
    return terminate, publish_points


async def _wait_opened(server, timeout=5.0):
    deadline = time.time() + timeout
    while not server._opened.done():
        task = getattr(server, "_task", None)
        if task is not None and task.done():
            exc = task.exception()
            if exc is not None:
                raise exc
            raise RuntimeError("Foxglove server stopped before opening")
        if time.time() >= deadline:
            raise TimeoutError("Timed out waiting for Foxglove server to open")
        await asyncio.sleep(0.05)
    return await server.wait_opened()


def run_foxglove_publisher(
    cfg,
    states,
    keyframes,
    calib_path=None,
    host="127.0.0.1",
    port=8765,
    publish_hz=5.0,
    conf_threshold=1.5,
    max_points=20000,
    max_keyframes=8,
    jpeg_quality=70,
    event_queue=None,
):
    Tic = np.eye(4, dtype=np.float64)
    if calib_path:
        try:
            with open(calib_path, "r") as f:
                Tic = np.asarray(yaml.load(f, Loader=yaml.SafeLoader).get("Tic", Tic), dtype=np.float64)
        except Exception as exc:
            print(f"[foxglove] failed to load Tic from {calib_path}: {exc}")
    options = {
        "conf_threshold": float(conf_threshold),
        "max_points": int(max_points),
        "max_keyframes": int(max_keyframes),
        "jpeg_quality": int(jpeg_quality),
        "Tic": Tic,
        "old_track_age_threshold": int(config.get("tracking", {}).get("old_track_age_threshold", 5)),
    }
    if event_queue is None:
        raise ValueError("run_foxglove_publisher requires an event_queue for event-driven publishing.")
    try:
        asyncio.run(_run_server(cfg, states, keyframes, host, int(port), publish_hz, options, event_queue))
    finally:
        print("[foxglove] stopped")
