from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import lietorch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mast3r_fusion.sparse_flow_frontend import (
    SparseFlowFrontend,
    SparseFlowResult,
)


# This module follows pySLAM's monocular tracking/keyframe semantics: tracked
# map-point inliers drive keyframe insertion, while raw optical-flow statistics
# are used only for visualization. pySLAM is GPLv3 licensed.


@dataclass
class MapPoint:
    point_id: int
    anchor_keyframe_idx: int
    anchor_uv: np.ndarray
    point_camera: np.ndarray
    frozen_world: Optional[np.ndarray] = None
    observations: dict[int, np.ndarray] = field(default_factory=dict)
    visible_count: int = 1
    found_count: int = 1
    bad: bool = False


@dataclass
class KeyframeRecord:
    keyframe_idx: int
    frame_id: int
    gray: np.ndarray
    keypoints: np.ndarray
    pose_data: np.ndarray
    observed_point_ids: set[int] = field(default_factory=set)
    anchored_point_ids: list[int] = field(default_factory=list)


@dataclass
class SparseMapTrackingResult:
    flow: SparseFlowResult
    tracking_ok: bool
    reference_keyframe_idx: Optional[int]
    reference_tracked_points: int
    matched_inlier_map_points: int
    inlier_point_ids: np.ndarray
    inlier_image_points: np.ndarray

    @property
    def debug(self):
        return self.flow.debug


class SparseMap:
    """Sparse map and pose-only tracking for delayed PI3X."""

    def __init__(self, K, width, height, cfg):
        self.K = np.asarray(K, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.cfg = cfg
        self.flow = SparseFlowFrontend.from_config(self.K, width, height, cfg)

        self.keyframes: dict[int, KeyframeRecord] = {}
        self.map_points: dict[int, MapPoint] = {}
        self.recent_point_ids: set[int] = set()
        self.next_point_id = 0
        self.reference_keyframe_idx = None
        self.last_keyframe_idx = None
        self.last_tracking_result = None

        self.max_local_keyframes = int(cfg.get("local_keyframes", 80))
        self.max_points_per_keyframe = int(cfg.get("map_points_per_keyframe", 2000))
        self.map_confidence = float(cfg.get("map_point_confidence", 0.3))
        self.reprojection_error = float(cfg.get("reprojection_error", 3.0))
        self.min_tracking_inliers = int(cfg.get("min_tracking_inliers", 20))
        self.ref_ratio = float(cfg.get("keyframe_ref_ratio", 0.9))
        self.min_points_for_keyframe = int(cfg.get("min_tracked_points_for_keyframe", 15))
        self.max_keyframe_gap = int(cfg.get("max_keyframe_gap", 20))

    @classmethod
    def from_config(cls, K, width, height, cfg):
        return cls(K, width, height, cfg)

    def process_frame(self, frame, timestamp=None, frames_since_keyframe=None):
        flow_result = self.flow.process_frame(
            frame,
            timestamp=timestamp,
            frames_since_keyframe=frames_since_keyframe,
        )
        gray = self._gray_from_frame(frame)
        tracking = self._track_local_map(frame, gray)
        tracking.flow = flow_result
        flow_result.debug.update(
            {
                "map_tracking_ok": float(tracking.tracking_ok),
                "map_inlier_num": float(tracking.matched_inlier_map_points),
                "map_ref_point_num": float(tracking.reference_tracked_points),
                "map_ref_ratio": (
                    float(tracking.matched_inlier_map_points)
                    / float(tracking.reference_tracked_points)
                    if tracking.reference_tracked_points > 0
                    else 0.0
                ),
            }
        )
        self.last_tracking_result = tracking
        return tracking

    def draw_overlay(self, result):
        image = result.flow.image.copy()
        for point_id, uv in zip(
            result.inlier_point_ids, result.inlier_image_points
        ):
            if not np.isfinite(uv).all():
                continue
            map_point = self.map_points.get(int(point_id))
            if map_point is None or map_point.bad:
                continue
            color = (
                (0, 255, 0)
                if len(map_point.observations) > 2
                else (255, 0, 0)
            )
            point = tuple(np.rint(uv).astype(int))
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
        return image

    def need_new_keyframe(self, frame_id, last_keyframe_frame_id, result, local_mapping_idle=True):
        gap = int(frame_id) - int(last_keyframe_frame_id)
        if len(self.keyframes) < 2:
            return bool(
                result.tracking_ok
                and result.matched_inlier_map_points > self.min_points_for_keyframe
            )
        if not result.tracking_ok:
            return False

        enough_points = result.matched_inlier_map_points > self.min_points_for_keyframe
        tracking_weakened = (
            result.reference_tracked_points > 0
            and result.matched_inlier_map_points
            < result.reference_tracked_points * self.ref_ratio
        )
        max_interval_reached = gap >= self.max_keyframe_gap
        return bool(
            (max_interval_reached or local_mapping_idle)
            and tracking_weakened
            and enough_points
        )

    def register_keyframe(self, keyframe_idx, frame, result=None):
        gray = self._gray_from_frame(frame)
        keypoints, _ = self.flow.tracker.detectAndCompute(gray)
        keypoints = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
        observed_ids = set()
        if result is not None:
            observed_ids = {int(point_id) for point_id in result.inlier_point_ids.tolist()}

        record = KeyframeRecord(
            keyframe_idx=int(keyframe_idx),
            frame_id=int(frame.frame_id),
            gray=gray.copy(),
            keypoints=keypoints,
            pose_data=self._pose_data(frame),
            observed_point_ids=observed_ids,
        )
        self.keyframes[record.keyframe_idx] = record
        observations = (
            zip(result.inlier_point_ids, result.inlier_image_points)
            if result is not None
            else []
        )
        for point_id, uv in observations:
            point = self.map_points.get(int(point_id))
            if point is not None and not point.bad:
                point.observations[record.keyframe_idx] = np.asarray(
                    uv, dtype=np.float64
                ).copy()

        self.last_keyframe_idx = record.keyframe_idx
        self.reference_keyframe_idx = record.keyframe_idx
        self._cull_recent_map_points()

    def update_keyframe_map(self, keyframe_idx, frame):
        record = self.keyframes.get(int(keyframe_idx))
        if record is None:
            return
        record.pose_data = self._pose_data(frame)
        samples = self._sample_keyframe_points(frame, record.keypoints)
        if samples is None:
            return
        uvs, points_camera = samples

        if record.anchored_point_ids:
            by_uv = {
                tuple(np.rint(uv).astype(np.int32)): point
                for uv, point in zip(uvs, points_camera)
            }
            for point_id in record.anchored_point_ids:
                point = self.map_points.get(point_id)
                if point is None:
                    continue
                updated = by_uv.get(tuple(np.rint(point.anchor_uv).astype(np.int32)))
                if updated is not None:
                    point.point_camera = updated
            return

        for uv, point_camera in zip(uvs, points_camera):
            if self._is_observed_feature(record, uv):
                continue
            point_id = self.next_point_id
            self.next_point_id += 1
            point = MapPoint(
                point_id=point_id,
                anchor_keyframe_idx=record.keyframe_idx,
                anchor_uv=uv.copy(),
                point_camera=point_camera.copy(),
                observations={record.keyframe_idx: uv.copy()},
            )
            self.map_points[point_id] = point
            self.recent_point_ids.add(point_id)
            record.anchored_point_ids.append(point_id)
            record.observed_point_ids.add(point_id)

    def sync_keyframes(self, keyframes):
        first_idx = int(keyframes.rollup_sum.value)
        end_idx = first_idx + len(keyframes)
        self.prune_before(first_idx)
        for keyframe_idx in list(self.keyframes):
            if first_idx <= keyframe_idx < end_idx:
                self.update_keyframe_map(keyframe_idx, keyframes[keyframe_idx])

    def prune_before(self, first_keyframe_idx):
        first_keyframe_idx = int(first_keyframe_idx)
        stale_keyframes = {
            idx for idx in self.keyframes if idx < first_keyframe_idx
        }
        if not stale_keyframes:
            return

        remove_points = []
        for point_id, point in self.map_points.items():
            if point.anchor_keyframe_idx in stale_keyframes:
                point.frozen_world = self._point_world(point).copy()
            for keyframe_idx in stale_keyframes:
                point.observations.pop(keyframe_idx, None)
            if not point.observations:
                remove_points.append(point_id)

        for keyframe_idx in stale_keyframes:
            del self.keyframes[keyframe_idx]
        for point_id in remove_points:
            del self.map_points[point_id]
            self.recent_point_ids.discard(point_id)
        for record in self.keyframes.values():
            record.observed_point_ids.intersection_update(self.map_points)
            record.anchored_point_ids = [
                point_id
                for point_id in record.anchored_point_ids
                if point_id in self.map_points
            ]

        if self.last_keyframe_idx not in self.keyframes:
            self.last_keyframe_idx = max(self.keyframes, default=None)
        if self.reference_keyframe_idx not in self.keyframes:
            self.reference_keyframe_idx = self.last_keyframe_idx

    def _track_local_map(self, frame, gray):
        empty = SparseMapTrackingResult(
            flow=None,
            tracking_ok=False,
            reference_keyframe_idx=self.reference_keyframe_idx,
            reference_tracked_points=self._reference_point_count(),
            matched_inlier_map_points=0,
            inlier_point_ids=np.empty(0, dtype=np.int64),
            inlier_image_points=np.empty((0, 2), dtype=np.float64),
        )
        local_points = self._local_map_points()
        if len(local_points) < 6:
            return empty

        prior_pose = self._pose_data(frame)
        correspondences = self._match_map_points(gray, local_points, prior_pose)
        if correspondences is None:
            return empty
        world_points, image_points, point_ids = correspondences
        if world_points.shape[0] < 6:
            return empty

        optimized_pose, inlier_mask = self._optimize_pose(
            world_points, image_points, prior_pose
        )
        if (
            optimized_pose is None
            or np.count_nonzero(inlier_mask) < self.min_tracking_inliers
        ):
            return empty
        self._set_frame_pose(frame, optimized_pose, prior_pose[7])

        inlier_ids = point_ids[inlier_mask]
        for point_id in inlier_ids:
            point = self.map_points.get(int(point_id))
            if point is not None:
                point.found_count += 1

        self.reference_keyframe_idx = self._select_reference_keyframe(inlier_ids)
        return SparseMapTrackingResult(
            flow=None,
            tracking_ok=True,
            reference_keyframe_idx=self.reference_keyframe_idx,
            reference_tracked_points=self._reference_point_count(),
            matched_inlier_map_points=int(inlier_ids.shape[0]),
            inlier_point_ids=inlier_ids.astype(np.int64),
            inlier_image_points=image_points[inlier_mask].astype(np.float64),
        )

    def _match_map_points(self, gray, points, prior_pose):
        grouped = {}
        for point in points:
            source_keyframe_idx = max(
                idx for idx in point.observations if idx in self.keyframes
            )
            grouped.setdefault(source_keyframe_idx, []).append(
                (point, point.observations[source_keyframe_idx])
            )

        world_all = []
        image_all = []
        ids_all = []
        for source_idx, source_points in grouped.items():
            record = self.keyframes.get(source_idx)
            if record is None:
                continue
            map_points = [item[0] for item in source_points]
            world = np.asarray(
                [self._point_world(point) for point in map_points], dtype=np.float64
            )
            projected, valid_projection = self._project_world(world, prior_pose)
            if np.count_nonzero(valid_projection) < 1:
                continue
            selected = np.flatnonzero(valid_projection)
            for selected_idx in selected:
                map_points[selected_idx].visible_count += 1
            source_uv = np.asarray(
                [source_points[i][1] for i in selected], dtype=np.float32
            ).reshape(-1, 1, 2)
            initial_uv = projected[selected].astype(np.float32).reshape(-1, 1, 2)
            tracked_uv, status, _ = cv2.calcOpticalFlowPyrLK(
                record.gray,
                gray,
                source_uv,
                initial_uv,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            if tracked_uv is None or status is None:
                continue
            tracked_uv = tracked_uv.reshape(-1, 2)
            status = status.reshape(-1).astype(bool)
            in_bounds = (
                np.isfinite(tracked_uv).all(axis=1)
                & (tracked_uv[:, 0] >= 0)
                & (tracked_uv[:, 0] < self.width)
                & (tracked_uv[:, 1] >= 0)
                & (tracked_uv[:, 1] < self.height)
            )
            keep = status & in_bounds
            for local_i in np.flatnonzero(keep):
                point = map_points[selected[local_i]]
                world_all.append(world[selected[local_i]])
                image_all.append(tracked_uv[local_i])
                ids_all.append(point.point_id)

        if not world_all:
            return None
        return (
            np.asarray(world_all, dtype=np.float64),
            np.asarray(image_all, dtype=np.float64),
            np.asarray(ids_all, dtype=np.int64),
        )

    def _optimize_pose(self, world_points, image_points, prior_pose):
        T_wc = self._se3_from_pose_data(prior_pose)
        T_cw = np.linalg.inv(T_wc)
        rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
        tvec = T_cw[:3, 3:4].copy()
        success, rvec, tvec, ransac_inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            self.K,
            None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=self.reprojection_error,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or ransac_inliers is None or ransac_inliers.shape[0] < 6:
            return None, None
        inlier_idx = ransac_inliers.reshape(-1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                world_points[inlier_idx],
                image_points[inlier_idx],
                self.K,
                None,
                rvec,
                tvec,
            )
        projected, _ = cv2.projectPoints(world_points, rvec, tvec, self.K, None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        inlier_mask = np.isfinite(errors) & (errors <= self.reprojection_error)
        if np.count_nonzero(inlier_mask) < 6:
            return None, None

        R_cw, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4, dtype=np.float64)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = tvec.reshape(3)
        return np.linalg.inv(T_cw), inlier_mask

    def _sample_keyframe_points(self, frame, keypoints):
        if frame.X_canon is None or frame.C is None or int(frame.N) <= 0:
            return None
        shape = frame.img_shape.detach().cpu().numpy().reshape(-1, 2)[0]
        height, width = int(shape[0]), int(shape[1])
        depth = frame.X_canon.detach().cpu().numpy().reshape(height, width, 3)[..., 2]
        confidence = (
            frame.get_average_conf().detach().cpu().numpy().reshape(height, width)
        )
        pixels = np.rint(keypoints).astype(np.int32)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return None
        z = depth[pixels[indices, 1], pixels[indices, 0]]
        conf = confidence[pixels[indices, 1], pixels[indices, 0]]
        valid_depth = np.isfinite(z) & (z > 1e-6) & np.isfinite(conf) & (conf >= self.map_confidence)
        indices = indices[valid_depth]
        if indices.size == 0:
            return None
        conf = conf[valid_depth]
        if indices.size > self.max_points_per_keyframe:
            order = np.argsort(conf)[-self.max_points_per_keyframe :]
            indices = indices[order]
        uv = keypoints[indices].astype(np.float64)
        z = depth[pixels[indices, 1], pixels[indices, 0]].astype(np.float64)
        points = np.empty((indices.size, 3), dtype=np.float64)
        points[:, 0] = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0] * z
        points[:, 1] = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1] * z
        points[:, 2] = z
        return uv, points

    def _local_map_points(self):
        local_indices = self._local_keyframe_indices()
        point_ids = set()
        for idx in local_indices:
            point_ids.update(self.keyframes[idx].observed_point_ids)
        return [
            self.map_points[point_id]
            for point_id in point_ids
            if point_id in self.map_points and not self.map_points[point_id].bad
        ]

    def _local_keyframe_indices(self):
        root_idx = self.reference_keyframe_idx
        if root_idx not in self.keyframes:
            root_idx = self.last_keyframe_idx
        if root_idx not in self.keyframes:
            return []

        votes = {}
        for point_id in self.keyframes[root_idx].observed_point_ids:
            point = self.map_points.get(point_id)
            if point is None or point.bad:
                continue
            for keyframe_idx in point.observations:
                if keyframe_idx != root_idx and keyframe_idx in self.keyframes:
                    votes[keyframe_idx] = votes.get(keyframe_idx, 0) + 1
        neighbors = sorted(votes, key=votes.get, reverse=True)
        return [root_idx, *neighbors[: max(self.max_local_keyframes - 1, 0)]]

    def _reference_point_count(self):
        record = self.keyframes.get(self.reference_keyframe_idx)
        if record is None:
            return 0
        min_observations = 2 if len(self.keyframes) <= 2 else 3
        return sum(
            1
            for point_id in record.observed_point_ids
            if point_id in self.map_points
            and not self.map_points[point_id].bad
            and len(self.map_points[point_id].observations) >= min_observations
        )

    def _select_reference_keyframe(self, inlier_point_ids):
        votes = {}
        for point_id in inlier_point_ids:
            point = self.map_points.get(int(point_id))
            if point is None:
                continue
            for keyframe_idx in point.observations:
                if keyframe_idx in self.keyframes:
                    votes[keyframe_idx] = votes.get(keyframe_idx, 0) + 1
        if votes:
            return max(votes, key=votes.get)
        return self.reference_keyframe_idx

    def _cull_recent_map_points(self):
        if self.last_keyframe_idx is None:
            return
        stop_tracking = set()
        remove_points = set()
        for point_id in self.recent_point_ids:
            point = self.map_points.get(point_id)
            if point is None or point.bad:
                stop_tracking.add(point_id)
                continue
            age = self.last_keyframe_idx - point.anchor_keyframe_idx
            found_ratio = point.found_count / max(point.visible_count, 1)
            if found_ratio < 0.25:
                remove_points.add(point_id)
            elif age >= 2 and len(point.observations) <= 2:
                remove_points.add(point_id)
            elif age >= 3:
                stop_tracking.add(point_id)

        self.recent_point_ids.difference_update(stop_tracking | remove_points)
        for point_id in remove_points:
            self.map_points.pop(point_id, None)
            for record in self.keyframes.values():
                record.observed_point_ids.discard(point_id)
                if point_id in record.anchored_point_ids:
                    record.anchored_point_ids.remove(point_id)

    def _point_world(self, point):
        if point.frozen_world is not None:
            return point.frozen_world
        record = self.keyframes[point.anchor_keyframe_idx]
        data = record.pose_data
        rotation = Rotation.from_quat(data[3:7]).as_matrix()
        return rotation @ (point.point_camera * data[7]) + data[:3]

    def _is_observed_feature(self, record, uv):
        observed_uvs = []
        for point_id in record.observed_point_ids:
            point = self.map_points.get(point_id)
            if point is not None and record.keyframe_idx in point.observations:
                observed_uvs.append(point.observations[record.keyframe_idx])
        if not observed_uvs:
            return False
        distances = np.linalg.norm(np.asarray(observed_uvs) - uv, axis=1)
        return bool(np.min(distances) < 3.0)

    def _project_world(self, world_points, pose_data):
        T_wc = self._se3_from_pose_data(pose_data)
        points_camera = (world_points - T_wc[:3, 3]) @ T_wc[:3, :3]
        z = points_camera[:, 2]
        valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-6)
        safe_z = np.where(valid, z, 1.0)
        projected = np.zeros((world_points.shape[0], 2), dtype=np.float64)
        projected[:, 0] = self.K[0, 0] * points_camera[:, 0] / safe_z + self.K[0, 2]
        projected[:, 1] = self.K[1, 1] * points_camera[:, 1] / safe_z + self.K[1, 2]
        valid &= (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < self.width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < self.height)
        )
        return projected, valid

    def _set_frame_pose(self, frame, T_wc, scale):
        data = np.concatenate(
            [T_wc[:3, 3], Rotation.from_matrix(T_wc[:3, :3]).as_quat(), [scale]]
        )
        tensor = torch.as_tensor(
            data,
            device=frame.T_WC.data.device,
            dtype=frame.T_WC.data.dtype,
        ).reshape(1, 8)
        frame.T_WC = lietorch.Sim3(tensor)

    @staticmethod
    def _se3_from_pose_data(data):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(data[3:7]).as_matrix()
        T[:3, 3] = data[:3]
        return T

    @staticmethod
    def _pose_data(frame):
        return frame.T_WC.data.detach().cpu().numpy().reshape(-1, 8)[0].astype(np.float64)

    @staticmethod
    def _gray_from_frame(frame):
        image = np.clip(frame.uimg.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
