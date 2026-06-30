from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from foxglove_websocket.server import FoxgloveServer
from foxglove_websocket.types import ChannelWithoutId
from scipy.spatial.transform import Rotation

from ..types import BackendState, Camera, FrontendResult


TIME_DEF = {
    "type": "object",
    "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
    "required": ["sec", "nsec"],
}

POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "pose": {"$ref": "#/$defs/Pose"},
    },
    "required": ["timestamp", "frame_id", "pose"],
    "$defs": {
        "Time": TIME_DEF,
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
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
            "required": ["x", "y", "z"],
        },
        "Quaternion": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}},
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

COMPRESSED_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "format": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
    "required": ["timestamp", "frame_id", "format", "data"],
    "$defs": {"Time": TIME_DEF},
}

CAMERA_CALIBRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"$ref": "#/$defs/Time"},
        "frame_id": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "distortion_model": {"type": "string"},
        "D": {"type": "array", "items": {"type": "number"}},
        "K": {"type": "array", "items": {"type": "number"}},
        "R": {"type": "array", "items": {"type": "number"}},
        "P": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["timestamp", "frame_id", "width", "height", "distortion_model", "D", "K", "R", "P"],
    "$defs": {"Time": TIME_DEF},
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
            "properties": {"name": {"type": "string"}, "offset": {"type": "integer"}, "type": {"type": "integer"}},
            "required": ["name", "offset", "type"],
        },
    },
}


@dataclass
class FoxgloveConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    jpeg_quality: int = 85
    publish_every: int = 1
    max_points: int = 3000


class FoxglovePublisher:
    def __init__(self, camera: Camera, cfg: FoxgloveConfig):
        self.camera = camera
        self.cfg = cfg
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.frame_count = 0

    def start(self):
        self.thread.start()
        print(f"[sparse_vio][foxglove] ws://{self.cfg.host}:{self.cfg.port}")

    def publish(self, result: FrontendResult, state: BackendState):
        self.frame_count += 1
        if self.frame_count % max(1, self.cfg.publish_every) != 0:
            return
        item = (result, BackendState(dict(state.poses), dict(state.points), dict(state.track_to_point)))
        while self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put_nowait(item)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _thread_main(self):
        asyncio.run(self._run())

    async def _run(self):
        async with FoxgloveServer(
            self.cfg.host,
            self.cfg.port,
            "Sparse VIO demo",
            supported_encodings=["json"],
        ) as server:
            channels = {}
            for key, topic, schema_name, schema in [
                ("image", "/sparse_vio/image", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
                ("camera_info", "/sparse_vio/image/camera_info", "foxglove.CameraCalibration", CAMERA_CALIBRATION_SCHEMA),
                ("tracking_image", "/sparse_vio/tracking_image", "foxglove.CompressedImage", COMPRESSED_IMAGE_SCHEMA),
                ("tracking_camera_info", "/sparse_vio/tracking_image/camera_info", "foxglove.CameraCalibration", CAMERA_CALIBRATION_SCHEMA),
                ("tf", "/tf", "foxglove.FrameTransforms", FRAME_TRANSFORMS_SCHEMA),
                ("pose", "/sparse_vio/current_pose", "foxglove.PoseInFrame", POSE_SCHEMA),
                ("trajectory", "/sparse_vio/trajectory", "foxglove.PosesInFrame", POSES_SCHEMA),
                ("points", "/sparse_vio/points", "foxglove.PointCloud", POINT_CLOUD_SCHEMA),
                ("sparse_points", "/sparse_vio/sparse_points", "foxglove.PointCloud", POINT_CLOUD_SCHEMA),
            ]:
                channels[key] = await server.add_channel(
                    ChannelWithoutId(
                        topic=topic,
                        encoding="json",
                        schemaName=schema_name,
                        schema=json.dumps(schema),
                        schemaEncoding="jsonschema",
                    )
                )
            while not self.stop_event.is_set():
                try:
                    result, state = self.queue.get(timeout=0.05)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                timestamp_ns = int(time.time() * 1e9)
                await self._send_snapshot(server, channels, timestamp_ns, result, state)

    async def _send_snapshot(self, server, channels, timestamp_ns, result, state):
        image_msg = self._image_msg(timestamp_ns, result, state)
        camera_msg = self._camera_msg(timestamp_ns)
        points_msg = self._points_msg(timestamp_ns, state)
        for key, msg in {
            "image": image_msg,
            "camera_info": camera_msg,
            "tracking_image": image_msg,
            "tracking_camera_info": camera_msg,
            "tf": self._tf_msg(timestamp_ns, result.frame_id, state),
            "pose": self._pose_msg(timestamp_ns, result.frame_id, state),
            "trajectory": self._trajectory_msg(timestamp_ns, state),
            "points": points_msg,
            "sparse_points": points_msg,
        }.items():
            if msg is not None:
                await server.send_message(channels[key], timestamp_ns, json.dumps(msg).encode("utf8"))

    def _image_msg(self, timestamp_ns, result, state):
        image = self._draw_overlay(result, state)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
        if not ok:
            return None
        return {
            "timestamp": _time_msg(timestamp_ns),
            "frame_id": "camera",
            "format": "jpeg",
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    def _camera_msg(self, timestamp_ns):
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = self.camera.K
        return {
            "timestamp": _time_msg(timestamp_ns),
            "frame_id": "camera",
            "width": int(self.camera.width),
            "height": int(self.camera.height),
            "distortion_model": "plumb_bob",
            "D": [] if self.camera.distortion is None else np.asarray(self.camera.distortion, dtype=float).tolist(),
            "K": self.camera.K.reshape(-1).tolist(),
            "R": np.eye(3).reshape(-1).tolist(),
            "P": P.reshape(-1).tolist(),
        }

    def _pose_msg(self, timestamp_ns, frame_id, state):
        T_world_body = self._world_body_pose(frame_id, state)
        if T_world_body is None:
            return None
        return {"timestamp": _time_msg(timestamp_ns), "frame_id": "world", "pose": _pose(T_world_body)}

    def _trajectory_msg(self, timestamp_ns, state):
        T_body_camera_inv = np.linalg.inv(self._body_camera_pose())
        poses = [_pose(T_world_camera @ T_body_camera_inv) for _, T_world_camera in sorted(state.poses.items())]
        if not poses:
            return None
        return {"timestamp": _time_msg(timestamp_ns), "frame_id": "world", "poses": poses}

    def _tf_msg(self, timestamp_ns, frame_id, state):
        T_world_body = self._world_body_pose(frame_id, state)
        if T_world_body is None:
            return None
        T_body_camera = self._body_camera_pose()
        return {
            "transforms": [
                _transform(timestamp_ns, "world", "body", T_world_body),
                _transform(timestamp_ns, "body", "camera", T_body_camera),
            ]
        }

    def _points_msg(self, timestamp_ns, state):
        if not state.points:
            return None
        pts = np.asarray(list(state.points.values()), dtype=np.float32)
        if pts.shape[0] > self.cfg.max_points:
            stride = int(np.ceil(pts.shape[0] / self.cfg.max_points))
            pts = pts[::stride]
        rgb = np.tile(np.array([[80, 220, 80]], dtype=np.uint8), (pts.shape[0], 1))
        data = np.concatenate([pts.astype(np.float32).view(np.uint8).reshape(-1, 12), rgb], axis=1)
        return {
            "timestamp": _time_msg(timestamp_ns),
            "frame_id": "world",
            "pose": _identity_pose(),
            "point_stride": 15,
            "fields": [
                {"name": "x", "offset": 0, "type": 7},
                {"name": "y", "offset": 4, "type": 7},
                {"name": "z", "offset": 8, "type": 7},
                {"name": "red", "offset": 12, "type": 2},
                {"name": "green", "offset": 13, "type": 2},
                {"name": "blue", "offset": 14, "type": 2},
            ],
            "data": base64.b64encode(data.reshape(-1).tobytes()).decode("ascii"),
        }

    def _draw_overlay(self, result, state):
        image = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
        for track in result.tracks.values():
            color = _age_color(track.age)
            xy = np.rint(track.xy).astype(int)
            cv2.circle(image, tuple(xy), 2, color, -1, cv2.LINE_AA)
            if len(track.observations) >= 2:
                p0 = np.rint(track.observations[-2].xy).astype(int)
                cv2.line(image, tuple(p0), tuple(xy), color, 1, cv2.LINE_AA)
        text = (
            f"frame={result.frame_id} tracks={len(result.tracks)} "
            f"kf={int(result.new_keyframe)} poses={len(state.poses)} pts={len(state.points)}"
        )
        dbg = result.debug
        text2 = (
            f"age_med={dbg.get('age_median', 0):.0f} age>10={dbg.get('age_gt10', 0):.0f} "
            f"LK={dbg.get('lk_kept', 0):.0f}/{dbg.get('lk_before', 0):.0f} "
            f"F={dbg.get('f_kept', 0):.0f}/{dbg.get('f_before', 0):.0f} "
            f"PnP={dbg.get('pnp_inliers', 0):.0f} rmse={dbg.get('pnp_rmse', 0):.1f}"
        )
        text3 = (
            f"XFeat det={dbg.get('xfeat_detected', 0):.0f} "
            f"match={dbg.get('xfeat_raw_matches', 0):.0f} "
            f"geo={dbg.get('xfeat_geo_inliers', 0):.0f}/{dbg.get('xfeat_geo_ratio', 0):.2f} "
            f"tracked={dbg.get('xfeat_tracked', 0):.0f} spawn={dbg.get('xfeat_spawned', 0):.0f}"
        )
        cv2.rectangle(image, (8, 8), (720, 40), (0, 0, 0), -1)
        cv2.putText(image, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (8, 42), (880, 72), (0, 0, 0), -1)
        cv2.putText(image, text2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (8, 74), (880, 104), (0, 0, 0), -1)
        cv2.putText(image, text3, (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def _body_camera_pose(self):
        if self.camera.T_body_camera is None:
            return np.eye(4, dtype=np.float64)
        return np.asarray(self.camera.T_body_camera, dtype=np.float64)

    def _world_body_pose(self, frame_id, state):
        T_world_camera = state.poses.get(frame_id)
        if T_world_camera is None:
            return None
        return T_world_camera @ np.linalg.inv(self._body_camera_pose())


def _time_msg(timestamp_ns):
    return {"sec": timestamp_ns // 1_000_000_000, "nsec": timestamp_ns % 1_000_000_000}


def _identity_pose():
    return {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _pose(T):
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {
        "position": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])},
        "orientation": {"x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]), "w": float(quat[3])},
    }


def _transform(timestamp_ns, parent_frame_id, child_frame_id, T):
    pose = _pose(T)
    return {
        "timestamp": _time_msg(timestamp_ns),
        "parent_frame_id": parent_frame_id,
        "child_frame_id": child_frame_id,
        "translation": pose["position"],
        "rotation": pose["orientation"],
    }


def _age_color(age):
    t = min(1.0, age / 20.0)
    return (int(255 * (1 - t)), int(220 * t), 40)
