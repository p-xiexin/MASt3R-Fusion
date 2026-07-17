from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import lietorch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mast3r_fusion.sparse_flow_frontend import SparseFlowFrontend, SparseFlowResult


@dataclass
class MapPoint:
    point_id: int
    position_world: np.ndarray
    created_keyframe_idx: int
    observations: dict[int, np.ndarray] = field(default_factory=dict)
    visible_count: int = 2
    found_count: int = 2
    bad: bool = False


@dataclass
class KeyframeRecord:
    keyframe_idx: int
    frame_id: int
    gray: np.ndarray
    keypoints: np.ndarray
    pose_data: np.ndarray
    observed_point_ids: set[int] = field(default_factory=set)


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
    """Sparse LK map used before the delayed PI3X backend."""

    def __init__(self, K, width, height, cfg):
        self.K = np.asarray(K, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.flow = SparseFlowFrontend.from_config(self.K, width, height, cfg)

        self.keyframes: dict[int, KeyframeRecord] = {}
        self.map_points: dict[int, MapPoint] = {}
        self.recent_point_ids: set[int] = set()
        self.next_point_id = 0
        self.reference_keyframe_idx = None
        self.last_keyframe_idx = None
        self.last_tracking_result = None
        self.map_ready = False
        self.initialized = False
        self.parallax_since_keyframe = 0.0
        self.reference_flow_tracks = 0

        self.max_local_keyframes = int(cfg.get("local_keyframes", 80))
        self.max_points_per_keyframe = int(cfg.get("map_points_per_keyframe", 2000))
        self.visualization_downsample = max(
            1, int(cfg.get("visualization_downsample", 1))
        )
        self.reprojection_error = float(cfg.get("reprojection_error", 3.0))
        self.initializer_min_frame_gap = int(cfg.get("initializer_min_frame_gap", 2))
        self.initializer_min_features = int(cfg.get("initializer_min_features", 100))
        self.initializer_min_inliers = int(cfg.get("initializer_min_inliers", 5))
        self.min_tracking_inliers = int(cfg.get("min_tracking_inliers", 20))
        self.ref_ratio = float(cfg.get("keyframe_ref_ratio", 0.9))
        self.min_points_for_keyframe = int(cfg.get("min_tracked_points_for_keyframe", 15))
        self.min_keyframe_gap = int(cfg.get("min_keyframe_gap", 2))
        self.max_keyframe_gap = int(cfg.get("max_keyframe_gap", 20))
        self.force_keyframe_rotation = np.deg2rad(
            float(cfg.get("force_keyframe_gyro_deg", 30.0))
        )
        self.suppress_keyframe_rotation = np.deg2rad(
            float(cfg.get("suppress_keyframe_gyro_deg", 5.0))
        )
        self.suppress_keyframe_translation = float(
            cfg.get("suppress_keyframe_translation", 1.0)
        )
        self.suppress_keyframe_parallax = float(
            cfg.get("suppress_keyframe_parallax", 3.0)
        )
        self.min_triangulation_angle = np.deg2rad(
            float(cfg.get("min_triangulation_angle_deg", 0.5))
        )

    @classmethod
    def from_config(cls, K, width, height, cfg):
        return cls(K, width, height, cfg)

    def process_frame(
        self,
        frame,
        timestamp=None,
        gyro_R=None,
        frames_since_keyframe=None,
    ):
        flow_result = self.flow.process_frame(
            frame,
            timestamp=timestamp,
            gyro_R=gyro_R,
            frames_since_keyframe=frames_since_keyframe,
        )
        tracking = self._empty_tracking_result()
        self.parallax_since_keyframe += float(
            flow_result.debug.get("avg_parallax", 0.0)
        )
        tracking.flow = flow_result
        flow_inliers = int(
            flow_result.debug.get(
                "tracked_inlier_num", np.count_nonzero(flow_result.inlier_mask)
            )
        )
        flow_ref_ratio = (
            float(flow_inliers) / float(self.reference_flow_tracks)
            if self.reference_flow_tracks > 0
            else 1.0
        )
        flow_result.debug.update(
            {
                "map_ready": float(self.map_ready),
                "map_initialized": float(self.initialized),
                "map_tracking_ok": float(tracking.tracking_ok),
                "map_inlier_num": float(tracking.matched_inlier_map_points),
                "map_ref_point_num": float(tracking.reference_tracked_points),
                "map_ref_ratio": (
                    float(tracking.matched_inlier_map_points)
                    / float(tracking.reference_tracked_points)
                    if tracking.reference_tracked_points > 0
                    else 0.0
                ),
                "flow_ref_track_num": float(self.reference_flow_tracks),
                "flow_ref_ratio": flow_ref_ratio,
                "parallax_since_keyframe": self.parallax_since_keyframe,
            }
        )
        self.last_tracking_result = tracking
        return tracking

    def draw_overlay(self, result):
        image = result.flow.image.copy()
        valid = result.flow.inlier_mask & np.isfinite(result.flow.pts).all(axis=1)
        indices = np.flatnonzero(valid)[:: self.visualization_downsample]

        for index in indices:
            uv = result.flow.pts[index]
            age = result.flow.track_cnt[index]
            if not np.isfinite(uv).all():
                continue
            color = (255, 0, 0) if age >= 4 else (0, 0, 255)
            cv2.circle(
                image,
                tuple(np.rint(uv).astype(int)),
                2,
                color,
                -1,
                cv2.LINE_AA,
            )
        return image

    def export_point_cloud(self):
        points = []
        for map_point in self.map_points.values():
            if map_point.bad or not np.isfinite(map_point.position_world).all():
                continue
            points.append(map_point.position_world)
        if not points:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return points[:: self.visualization_downsample]

    def need_new_keyframe(
        self,
        frame_id,
        last_keyframe_frame_id,
        result,
        local_mapping_idle=True,
        relative_motion=None,
        gyro_R=None,
    ):
        del local_mapping_idle
        gap = int(frame_id) - int(last_keyframe_frame_id)
        flow_inliers = int(
            result.flow.debug.get(
                "tracked_inlier_num", np.count_nonzero(result.flow.inlier_mask)
            )
        )
        avg_parallax = float(result.flow.debug.get("avg_parallax", 0.0))
        rotation = 0.0
        translation = -1.0
        has_relative_motion = relative_motion is not None
        if relative_motion is not None:
            relative_motion = np.asarray(relative_motion, dtype=np.float64)
            rotation = float(
                Rotation.from_matrix(relative_motion[:3, :3]).magnitude()
            )
            translation = float(np.linalg.norm(relative_motion[:3, 3]))
        elif gyro_R is not None:
            gyro_R = np.asarray(gyro_R, dtype=np.float64)
            rotation = float(Rotation.from_matrix(gyro_R[:3, :3]).magnitude())

        reference_points = int(self.reference_flow_tracks)
        track_ratio = (
            float(flow_inliers) / float(reference_points)
            if reference_points > 0
            else 1.0
        )
        debug = result.flow.debug
        debug.update(
            {
                "keyframe_gap": float(gap),
                "keyframe_reference_tracks": float(reference_points),
                "keyframe_track_ratio": track_ratio,
                "keyframe_rotation_deg": float(np.rad2deg(rotation)),
                "keyframe_translation": translation,
            }
        )

        def decide(value, reason):
            debug["keyframe_selected"] = float(value)
            debug["keyframe_reason"] = reason
            return bool(value)

        if gap < self.min_keyframe_gap:
            return decide(False, "min_gap")
        if gap >= self.max_keyframe_gap:
            return decide(True, "max_gap")
        if rotation >= self.force_keyframe_rotation:
            return decide(True, "large_rotation")

        enough_points = flow_inliers >= self.min_points_for_keyframe
        tracking_weakened = (
            reference_points > 0 and track_ratio < self.ref_ratio
        )
        enough_motion = (
            self.parallax_since_keyframe >= self.flow.cfg.keyframe_parallax
        )
        flow_fallback = bool(
            result.flow.new_keyframe
            and flow_inliers >= self.initializer_min_inliers
        )
        candidate = bool(
            (tracking_weakened and enough_points)
            or (enough_motion and enough_points)
            or flow_fallback
        )
        if tracking_weakened and enough_points:
            reason = "track_ratio"
        elif enough_motion and enough_points:
            reason = "flow_motion"
        else:
            reason = "flow_fallback"

        low_rotation = rotation < self.suppress_keyframe_rotation
        low_translation = (
            not has_relative_motion
            or translation < self.suppress_keyframe_translation
        )
        low_parallax = avg_parallax < self.suppress_keyframe_parallax
        if candidate and low_rotation and low_translation and low_parallax:
            return decide(False, "low_motion")

        return decide(candidate, reason if candidate else "no_trigger")

    def register_keyframe(self, keyframe_idx, frame, result=None):
        previous_idx = self.last_keyframe_idx
        gray = self.flow.frame_gray(frame)
        keypoints = self.flow.detect_points(gray)

        observed_ids = set()
        if result is not None:
            observed_ids = {
                int(point_id) for point_id in result.inlier_point_ids.tolist()
            }
        record = KeyframeRecord(
            keyframe_idx=int(keyframe_idx),
            frame_id=int(frame.frame_id),
            gray=gray.copy(),
            keypoints=keypoints,
            pose_data=self._pose_data(frame),
            observed_point_ids=observed_ids,
        )
        self.keyframes[record.keyframe_idx] = record
        self._store_keyframe_observations(record, result)

        if previous_idx in self.keyframes:
            added = self._triangulate_new_points(previous_idx, record.keyframe_idx)
        else:
            added = 0

        self.last_keyframe_idx = record.keyframe_idx
        self.reference_keyframe_idx = record.keyframe_idx
        if result is not None and result.flow is not None:
            self.reference_flow_tracks = int(
                np.count_nonzero(result.flow.inlier_mask)
            )
        else:
            self.reference_flow_tracks = int(self.flow.ref_pts.shape[0])
            if self.reference_flow_tracks == 0:
                self.reference_flow_tracks = int(keypoints.shape[0])
        was_ready = self.map_ready
        self.map_ready = (
            self._reference_point_count() >= self.min_tracking_inliers
        )
        if not self.map_ready:
            self.initialized = False
        else:
            self.initialized = True
        if not was_ready or added > 0:
            print(
                f"[INFO] SparseMap keyframe={record.keyframe_idx} "
                f"triangulated={added} map_points={len(self.map_points)} "
                f"reference_points={self._reference_point_count()} "
                f"map_ready={self.map_ready} "
                f"initialized={self.initialized}"
            )
        if self.map_ready:
            self._cull_recent_map_points()
            self.map_ready = (
                self._reference_point_count() >= self.min_tracking_inliers
            )
            if not self.map_ready:
                self.initialized = False
            else:
                self.initialized = True
        self.parallax_since_keyframe = 0.0

    def optimize_local_bundle_adjustment(self):
        """Reserved for a future sparse visual-inertial BA stage."""
        return False

    def align_world_to_keyframe(self, keyframe_idx, frame):
        record = self.keyframes.get(int(keyframe_idx))
        if record is None:
            return False
        old_pose = record.pose_data
        new_pose = self._pose_data(frame)
        if np.allclose(old_pose, new_pose, rtol=1e-6, atol=1e-8):
            return False
        old_rotation = Rotation.from_quat(old_pose[3:7]).as_matrix()
        new_rotation = Rotation.from_quat(new_pose[3:7]).as_matrix()
        correction_rotation = new_rotation @ old_rotation.T
        correction_translation = (
            new_pose[:3] - correction_rotation @ old_pose[:3]
        )

        for point in self.map_points.values():
            point.position_world = (
                correction_translation
                + correction_rotation @ point.position_world
            )
        for keyframe in self.keyframes.values():
            pose = keyframe.pose_data
            rotation = Rotation.from_quat(pose[3:7]).as_matrix()
            transformed_rotation = correction_rotation @ rotation
            transformed_translation = (
                correction_translation + correction_rotation @ pose[:3]
            )
            keyframe.pose_data = np.concatenate(
                [
                    transformed_translation,
                    Rotation.from_matrix(transformed_rotation).as_quat(),
                    [pose[7]],
                ]
            )
        record.pose_data = new_pose
        return True

    def prune_before(self, first_keyframe_idx):
        first_keyframe_idx = int(first_keyframe_idx)
        stale_keyframes = {idx for idx in self.keyframes if idx < first_keyframe_idx}
        if not stale_keyframes:
            return

        remove_points = []
        for point_id, point in self.map_points.items():
            for keyframe_idx in stale_keyframes:
                point.observations.pop(keyframe_idx, None)
            if not point.observations:
                remove_points.append(point_id)

        for keyframe_idx in stale_keyframes:
            del self.keyframes[keyframe_idx]
        for point_id in remove_points:
            self.map_points.pop(point_id, None)
            self.recent_point_ids.discard(point_id)
        for record in self.keyframes.values():
            record.observed_point_ids.intersection_update(self.map_points)

        if self.last_keyframe_idx not in self.keyframes:
            self.last_keyframe_idx = max(self.keyframes, default=None)
        if self.reference_keyframe_idx not in self.keyframes:
            self.reference_keyframe_idx = self.last_keyframe_idx
        self.map_ready = (
            self._reference_point_count() >= self.min_tracking_inliers
        )
        if not self.map_ready:
            self.initialized = False

    def _store_keyframe_observations(self, record, result):
        if result is None:
            return
        for point_id, uv in zip(result.inlier_point_ids, result.inlier_image_points):
            point = self.map_points.get(int(point_id))
            if point is None or point.bad:
                continue
            point.observations[record.keyframe_idx] = np.asarray(
                uv, dtype=np.float64
            ).copy()

    def _triangulate_new_points(self, source_idx, target_idx):
        source = self.keyframes[source_idx]
        target = self.keyframes[target_idx]
        source_uv = self._unobserved_keypoints(source)
        if source_uv.shape[0] < 5:
            return 0

        _, source_matched, target_matched = self.flow.track_points(
            source.gray, target.gray, source_uv
        )
        if source_matched.shape[0] < 5:
            return 0

        valid = self._points_in_bounds(target_matched)
        valid &= ~self._points_near_observations(target, target_matched)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 5:
            return 0

        source_matched = source_matched[valid_indices]
        target_matched = target_matched[valid_indices]
        world_points, triangulated = self._triangulate(
            source.pose_data,
            target.pose_data,
            source_matched,
            target_matched,
        )
        valid_indices = np.flatnonzero(triangulated)
        if valid_indices.size > self.max_points_per_keyframe:
            valid_indices = valid_indices[: self.max_points_per_keyframe]

        for index in valid_indices:
            point_id = self.next_point_id
            self.next_point_id += 1
            point = MapPoint(
                point_id=point_id,
                position_world=world_points[index].copy(),
                created_keyframe_idx=target_idx,
                observations={
                    source_idx: source_matched[index].astype(np.float64),
                    target_idx: target_matched[index].astype(np.float64),
                },
            )
            self.map_points[point_id] = point
            self.recent_point_ids.add(point_id)
            source.observed_point_ids.add(point_id)
            target.observed_point_ids.add(point_id)
        return int(valid_indices.size)

    def _triangulate(self, source_pose, target_pose, source_uv, target_uv):
        source_T_wc = self._se3_from_pose_data(source_pose)
        target_T_wc = self._se3_from_pose_data(target_pose)
        count = source_uv.shape[0]
        empty_points = np.zeros((count, 3), dtype=np.float64)
        if np.linalg.norm(source_T_wc[:3, 3] - target_T_wc[:3, 3]) <= 1e-8:
            return empty_points, np.zeros(count, dtype=bool)

        source_norm = cv2.undistortPoints(
            np.ascontiguousarray(source_uv).reshape(-1, 1, 2), self.K, None
        ).reshape(-1, 2)
        target_norm = cv2.undistortPoints(
            np.ascontiguousarray(target_uv).reshape(-1, 1, 2), self.K, None
        ).reshape(-1, 2)
        source_T_cw = np.linalg.inv(source_T_wc)
        target_T_cw = np.linalg.inv(target_T_wc)
        homogeneous = cv2.triangulatePoints(
            source_T_cw[:3],
            target_T_cw[:3],
            source_norm.T,
            target_norm.T,
        ).T
        valid = np.isfinite(homogeneous).all(axis=1)
        valid &= np.abs(homogeneous[:, 3]) > 1e-10
        world_points = empty_points
        world_points[valid] = homogeneous[valid, :3] / homogeneous[valid, 3:4]

        source_camera = self._world_to_camera(world_points, source_T_wc)
        target_camera = self._world_to_camera(world_points, target_T_wc)
        valid &= source_camera[:, 2] > 1e-6
        valid &= target_camera[:, 2] > 1e-6

        source_projected, source_visible = self._project_world(
            world_points, source_pose
        )
        target_projected, target_visible = self._project_world(
            world_points, target_pose
        )
        source_error = np.linalg.norm(source_projected - source_uv, axis=1)
        target_error = np.linalg.norm(target_projected - target_uv, axis=1)
        valid &= source_visible & target_visible
        valid &= source_error <= self.reprojection_error
        valid &= target_error <= self.reprojection_error

        source_rays = world_points - source_T_wc[:3, 3]
        target_rays = world_points - target_T_wc[:3, 3]
        denominator = np.linalg.norm(source_rays, axis=1) * np.linalg.norm(
            target_rays, axis=1
        )
        cosine = np.ones(count, dtype=np.float64)
        nonzero = denominator > 1e-10
        cosine[nonzero] = np.sum(
            source_rays[nonzero] * target_rays[nonzero], axis=1
        ) / denominator[nonzero]
        valid &= cosine < np.cos(self.min_triangulation_angle)
        return world_points, valid

    def _track_local_map(self, frame, gray):
        empty = self._empty_tracking_result()
        local_points = self._local_map_points()
        if len(local_points) < 6:
            return empty

        correspondences = self._match_map_points(gray, local_points)
        if correspondences is None:
            return empty
        world_points, image_points, point_ids = correspondences
        if world_points.shape[0] < 6:
            return empty

        prior_pose = self._pose_data(frame)
        optimized_pose, inlier_mask = self._optimize_pose(
            world_points, image_points
        )
        if optimized_pose is None or np.count_nonzero(inlier_mask) < self.min_tracking_inliers:
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

    def _match_map_points(self, gray, points):
        grouped = {}
        for point in points:
            available = [idx for idx in point.observations if idx in self.keyframes]
            if not available:
                continue
            source_idx = max(available)
            grouped.setdefault(source_idx, []).append(
                (point, point.observations[source_idx])
            )

        world_all = []
        image_all = []
        ids_all = []
        for source_idx, source_points in grouped.items():
            record = self.keyframes[source_idx]
            source_uv = np.asarray(
                [item[1] for item in source_points], dtype=np.float32
            ).reshape(-1, 2)
            matched_indices, source_matched, target_matched = (
                self.flow.track_points(record.gray, gray, source_uv)
            )
            if matched_indices.size < 5:
                continue
            for index in matched_indices:
                source_points[index][0].visible_count += 1

            valid = self._points_in_bounds(target_matched)
            for local_index in np.flatnonzero(valid):
                point = source_points[matched_indices[local_index]][0]
                world_all.append(point.position_world)
                image_all.append(target_matched[local_index])
                ids_all.append(point.point_id)

        if not world_all:
            return None
        return (
            np.asarray(world_all, dtype=np.float64),
            np.asarray(image_all, dtype=np.float64),
            np.asarray(ids_all, dtype=np.int64),
        )

    def _optimize_pose(self, world_points, image_points):
        success, rvec, tvec, ransac_inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            self.K,
            None,
            iterationsCount=100,
            reprojectionError=self.reprojection_error,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
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

        rotation_cw, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4, dtype=np.float64)
        T_cw[:3, :3] = rotation_cw
        T_cw[:3, 3] = tvec.reshape(3)
        return np.linalg.inv(T_cw), inlier_mask

    def _empty_tracking_result(self):
        return SparseMapTrackingResult(
            flow=None,
            tracking_ok=False,
            reference_keyframe_idx=self.reference_keyframe_idx,
            reference_tracked_points=self._reference_point_count(),
            matched_inlier_map_points=0,
            inlier_point_ids=np.empty(0, dtype=np.int64),
            inlier_image_points=np.empty((0, 2), dtype=np.float64),
        )

    def _unobserved_keypoints(self, record):
        keypoints = record.keypoints
        if keypoints.shape[0] == 0:
            return keypoints
        observed = self._observed_uvs(record)
        if observed.shape[0] == 0:
            return keypoints
        distance_sq = np.sum(
            (keypoints[:, None, :] - observed[None, :, :]) ** 2, axis=2
        )
        return keypoints[np.min(distance_sq, axis=1) >= 9.0]

    def _points_near_observations(self, record, points):
        observed = self._observed_uvs(record)
        if observed.shape[0] == 0:
            return np.zeros(points.shape[0], dtype=bool)
        distance_sq = np.sum(
            (points[:, None, :] - observed[None, :, :]) ** 2, axis=2
        )
        return np.min(distance_sq, axis=1) < 9.0

    def _observed_uvs(self, record):
        observations = []
        for point_id in record.observed_point_ids:
            point = self.map_points.get(point_id)
            if point is not None and record.keyframe_idx in point.observations:
                observations.append(point.observations[record.keyframe_idx])
        if not observations:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(observations, dtype=np.float64)

    def _local_map_points(self):
        point_ids = set()
        for idx in self._local_keyframe_indices():
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
        min_observations = 2
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
        return max(votes, key=votes.get) if votes else self.reference_keyframe_idx

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
            age = self.last_keyframe_idx - point.created_keyframe_idx
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

    def _project_world(self, world_points, pose_data):
        T_wc = self._se3_from_pose_data(pose_data)
        points_camera = self._world_to_camera(world_points, T_wc)
        z = points_camera[:, 2]
        valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-6)
        safe_z = np.where(valid, z, 1.0)
        projected = np.zeros((world_points.shape[0], 2), dtype=np.float64)
        projected[:, 0] = self.K[0, 0] * points_camera[:, 0] / safe_z + self.K[0, 2]
        projected[:, 1] = self.K[1, 1] * points_camera[:, 1] / safe_z + self.K[1, 2]
        valid &= self._points_in_bounds(projected)
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

    def _points_in_bounds(self, points):
        return (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < self.width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < self.height)
        )

    @staticmethod
    def _world_to_camera(world_points, T_wc):
        return (world_points - T_wc[:3, 3]) @ T_wc[:3, :3]

    @staticmethod
    def _se3_from_pose_data(data):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(data[3:7]).as_matrix()
        T[:3, 3] = data[:3]
        return T

    @staticmethod
    def _pose_data(frame):
        return (
            frame.T_WC.data.detach()
            .cpu()
            .numpy()
            .reshape(-1, 8)[0]
            .astype(np.float64)
        )
