from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import gtsam
import numpy as np

from mast3r_fusion.vio_utils import VisualIMUAlignment

from ..geometry import compose, invert_pose, se3_from_rt, triangulate_pair
from ..imu import ImuBuffer
from ..types import BackendState, Camera, FrontendResult, Track


@dataclass
class BackendConfig:
    window_size: int = 8
    min_triangulation_parallax: float = 8.0
    max_ba_iters: int = 10
    max_ba_points: int = 80
    huber_loss: str = "huber"
    reproj_sigma: float = 2.0
    robust_reprojection: bool = False
    vi_init_min_keyframes: int = 8
    vi_init_min_points: int = 50
    vi_init_min_baseline: float = 0.05
    gravity: float = 9.81
    accel_noise_sigma: float = 0.08
    gyro_noise_sigma: float = 0.004
    accel_bias_rw_sigma: float = 0.0004
    gyro_bias_rw_sigma: float = 0.00002
    bias_prior_sigma: float = 0.1
    velocity_prior_sigma: float = 1.0
    marginal_pose_sigma: float = 0.03
    marginal_velocity_sigma: float = 0.3
    marginal_bias_sigma: float = 0.03
    marginal_point_sigma: float = 0.2
    final_global_ba: bool = True
    max_global_ba_points: int = 1600
    max_global_ba_iters: int = 8
    min_pnp_inliers: int = 30
    min_pnp_inlier_ratio: float = 0.2
    max_pnp_reproj_rmse: float = 3.5
    max_init_rotation_deg: float = 12.0
    max_init_translation_factor: float = 4.0
    max_init_translation_abs: float = 8.0
    min_track_observations: int = 3
    keyframe_only: bool = False
    use_odom_prior: bool = False
    vo_init_translation: float = 0.25
    vo_max_translation: float = 0.6
    vo_pose_prior_rot_sigma: float = 0.03
    vo_pose_prior_trans_sigma: float = 0.05
    odom_prior_rot_sigma: float = 0.05
    odom_prior_trans_sigma: float = 0.2
    vio_imu_mode: str = "gyro"
    gyro_factor_rot_sigma: float = 0.01
    gyro_factor_trans_sigma: float = 1.0e6
    vio_visual_pose_prior_rot_sigma: float = 0.02
    vio_visual_pose_prior_trans_sigma: float = 0.05
    vio_max_pose_update_trans: float = 0.25
    vio_max_pose_update_rot_deg: float = 5.0
    vio_use_imu_rotation: bool = True
    fix_vo_poses: bool = True
    fixed_pose_sigma: float = 1e-6
    max_landmark_depth: float = 80.0
    max_landmark_reproj_error: float = 4.0
    vi_init_disable_scale: bool = True
    vi_init_min_scale: float = 0.05
    vi_init_max_scale: float = 20.0
    vi_init_max_gyro_bias: float = 0.2


class SparseBackend:
    def __init__(self, camera: Camera, cfg: BackendConfig | None = None, imu: Optional[ImuBuffer] = None):
        self.camera = camera
        self.cfg = cfg or BackendConfig()
        self.imu = imu
        self.vi_initialized = False
        self.state = BackendState()
        self.keyframes: List[FrontendResult] = []
        self.init_keyframes: List[FrontendResult] = []
        self.frame_timestamps: Dict[int, float] = {}
        self.next_point_id = 0
        self.T_body_camera = np.asarray(camera.T_body_camera if camera.T_body_camera is not None else np.eye(4), dtype=np.float64)
        self.body_P_sensor = gtsam.Pose3(self.T_body_camera)
        if camera.distortion is not None and len(camera.distortion) >= 4:
            self.calib = gtsam.Cal3DS2(
                float(camera.K[0, 0]),
                float(camera.K[1, 1]),
                float(camera.K[0, 1]),
                float(camera.K[0, 2]),
                float(camera.K[1, 2]),
                float(camera.distortion[0]),
                float(camera.distortion[1]),
                float(camera.distortion[2]),
                float(camera.distortion[3]),
            )
            self.projection_factor_cls = gtsam.GenericProjectionFactorCal3DS2
        else:
            self.calib = gtsam.Cal3_S2(
                float(camera.K[0, 0]),
                float(camera.K[1, 1]),
                float(camera.K[0, 1]),
                float(camera.K[0, 2]),
                float(camera.K[1, 2]),
            )
            self.projection_factor_cls = gtsam.GenericProjectionFactorCal3_S2
        base_pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.cfg.reproj_sigma)
        self.pixel_noise = (
            gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_pixel_noise)
            if self.cfg.robust_reprojection
            else base_pixel_noise
        )
        self.pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], dtype=np.float64)
        )
        self.body_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2], dtype=np.float64)
        )
        self.velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.cfg.velocity_prior_sigma)
        self.bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.bias_prior_sigma)
        self.imu_params = self._make_imu_params()
        self.zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.biases: Dict[int, gtsam.imuBias.ConstantBias] = {}
        self.pose_priors: Dict[int, np.ndarray] = {}
        self.velocity_priors: Dict[int, np.ndarray] = {}
        self.bias_priors: Dict[int, gtsam.imuBias.ConstantBias] = {}
        self.point_priors: Dict[int, np.ndarray] = {}
        self.vo_pose_priors: Dict[int, np.ndarray] = {}
        self.relative_pose_priors: Dict[tuple[int, int], np.ndarray] = {}
        self.marginal_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.cfg.marginal_pose_sigma] * 3 + [self.cfg.marginal_pose_sigma] * 3, dtype=np.float64)
        )
        self.marginal_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.cfg.marginal_velocity_sigma)
        self.marginal_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.cfg.marginal_bias_sigma)
        self.marginal_point_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.cfg.marginal_point_sigma)
        self.odom_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.cfg.odom_prior_rot_sigma] * 3 + [self.cfg.odom_prior_trans_sigma] * 3, dtype=np.float64)
        )
        self.vo_pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.cfg.vo_pose_prior_rot_sigma] * 3 + [self.cfg.vo_pose_prior_trans_sigma] * 3, dtype=np.float64)
        )
        self.vio_visual_pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [self.cfg.vio_visual_pose_prior_rot_sigma] * 3 + [self.cfg.vio_visual_pose_prior_trans_sigma] * 3,
                dtype=np.float64,
            )
        )
        self.gyro_between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.cfg.gyro_factor_rot_sigma] * 3 + [self.cfg.gyro_factor_trans_sigma] * 3, dtype=np.float64)
        )
        self.fixed_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.cfg.fixed_pose_sigma] * 6, dtype=np.float64))

    def push(self, result: FrontendResult) -> BackendState:
        if self.cfg.keyframe_only and self.keyframes and not result.new_keyframe:
            return self.state
        self.keyframes.append(result)
        self.init_keyframes.append(result)
        self.frame_timestamps[result.frame_id] = result.timestamp
        if len(self.keyframes) == 1:
            self.state.poses[result.frame_id] = np.eye(4, dtype=np.float64)
            self.vo_pose_priors[result.frame_id] = self.state.poses[result.frame_id].copy()
            self.state.body_poses[result.frame_id] = self.state.poses[result.frame_id] @ np.linalg.inv(self.T_body_camera)
            self.state.velocities[result.frame_id] = np.zeros(3, dtype=np.float64)
            self.biases[result.frame_id] = self.zero_bias
            return self.state
        self._init_latest_pose()
        self._triangulate_new_points()
        if self.imu is not None and self.cfg.vio_imu_mode != "predict" and not self.vi_initialized and self._vi_init_ready():
            self._try_vi_initialize()
        self._optimize_window()
        if len(self.keyframes) > self.cfg.window_size:
            self._slide_window(result.new_keyframe)
        return self.state

    def _slide_window(self, keep_second_newest: bool):
        if keep_second_newest:
            self.keyframes = self.keyframes[-self.cfg.window_size :]
            return
        if len(self.keyframes) >= 2:
            self.keyframes.pop(-2)
        if len(self.keyframes) > self.cfg.window_size:
            self.keyframes = self.keyframes[-self.cfg.window_size :]

    def _init_latest_pose(self):
        prev = self.keyframes[-2]
        cur = self.keyframes[-1]
        T_pnp, pnp_debug = self._solve_pnp_camera_pose(cur)
        cur.debug.update(pnp_debug)
        cv_camera_prior = self._constant_velocity_camera_prior(prev.frame_id)
        imu_camera_prior = None
        if self.imu is not None and (self.vi_initialized or self.cfg.vio_imu_mode in {"predict", "predict_align"}):
            if self.cfg.vio_imu_mode == "combined":
                predicted = self._predict_with_imu(prev, cur)
                if predicted is not None:
                    T_world_body, velocity = predicted
                    self.biases[cur.frame_id] = self.biases.get(prev.frame_id, self.zero_bias)
                    self.state.velocities[cur.frame_id] = velocity
                    imu_camera_prior = T_world_body @ self.T_body_camera
            else:
                imu_camera_prior = self._predict_camera_with_gyro(prev, cur, cv_camera_prior)
        motion_prior = imu_camera_prior if imu_camera_prior is not None else cv_camera_prior
        if T_pnp is not None and self._accept_pose_candidate(prev.frame_id, T_pnp, motion_prior, cur.debug, "pnp"):
            T_pnp = self._apply_imu_rotation(T_pnp, imu_camera_prior)
            self._set_latest_pose_from_pnp(cur, T_pnp)
            return
        prev_pts, cur_pts = self._matched_points(prev.tracks, cur.tracks)
        if prev_pts.shape[0] < 8:
            self._set_latest_pose_from_prediction(prev.frame_id, cur.frame_id, motion_prior)
            return
        prev_norm = self._undistort_points(prev_pts)
        cur_norm = self._undistort_points(cur_pts)
        E, mask = cv2.findEssentialMat(
            prev_norm,
            cur_norm,
            np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0 / float((self.camera.K[0, 0] + self.camera.K[1, 1]) * 0.5),
        )
        if E is None:
            self._set_latest_pose_from_prediction(prev.frame_id, cur.frame_id, motion_prior)
            return
        inliers, R_cur_prev, t_cur_prev, _ = cv2.recoverPose(E, prev_norm, cur_norm, np.eye(3), mask=mask)
        t_norm = float(np.linalg.norm(t_cur_prev))
        if t_norm > 1e-8:
            if self._metric_scale_active():
                trans_scale = self._recent_translation_norm(prev.frame_id)
            else:
                trans_scale = min(self.cfg.vo_max_translation, max(1e-3, self._recent_translation_norm(prev.frame_id)))
                if not self.vi_initialized:
                    trans_scale = self.cfg.vo_init_translation
            t_cur_prev = t_cur_prev * (trans_scale / t_norm)
        T_cur_prev = se3_from_rt(R_cur_prev, t_cur_prev.reshape(3))
        T_prev_cur = invert_pose(T_cur_prev)
        T_essential = compose(self.state.poses[prev.frame_id], T_prev_cur)
        cur.debug["essential_inliers"] = float(inliers)
        if self._accept_pose_candidate(prev.frame_id, T_essential, motion_prior, cur.debug, "essential"):
            T_essential = self._apply_imu_rotation(T_essential, imu_camera_prior)
            self._set_latest_pose_from_camera(cur.frame_id, T_essential, prev.frame_id)
        else:
            self._set_latest_pose_from_prediction(prev.frame_id, cur.frame_id, motion_prior)

    def _apply_imu_rotation(self, T_visual: np.ndarray, T_imu_prior: Optional[np.ndarray]):
        if (
            T_imu_prior is None
            or self.imu is None
            or self.cfg.vio_imu_mode != "predict_align"
            or not self.cfg.vio_use_imu_rotation
            or not self.vi_initialized
        ):
            return T_visual
        T = T_visual.copy()
        T[:3, :3] = T_imu_prior[:3, :3]
        return T

    def _solve_pnp_camera_pose(self, cur: FrontendResult):
        object_points = []
        image_points = []
        for track_id, track in cur.tracks.items():
            point_id = self.state.track_to_point.get(track_id)
            if point_id is None or point_id not in self.state.points:
                continue
            object_points.append(self.state.points[point_id])
            image_points.append(track.xy)
        if len(object_points) < 12:
            return None, {"pnp_refs": float(len(object_points))}
        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera.K,
            self.camera.distortion,
            iterationsCount=100,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        debug = {"pnp_refs": float(len(object_points)), "pnp_inliers": 0.0, "pnp_rmse": 999.0}
        if not ok or inliers is None:
            return None, debug
        debug["pnp_inliers"] = float(len(inliers))
        debug["pnp_inlier_ratio"] = float(len(inliers) / max(1, len(object_points)))
        if len(inliers) < self.cfg.min_pnp_inliers or debug["pnp_inlier_ratio"] < self.cfg.min_pnp_inlier_ratio:
            return None, debug
        ok, rvec, tvec = cv2.solvePnP(
            object_points[inliers[:, 0]],
            image_points[inliers[:, 0]],
            self.camera.K,
            self.camera.distortion,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, debug
        proj, _ = cv2.projectPoints(object_points[inliers[:, 0]], rvec, tvec, self.camera.K, self.camera.distortion)
        err = np.linalg.norm(proj.reshape(-1, 2) - image_points[inliers[:, 0]], axis=1)
        debug["pnp_rmse"] = float(np.sqrt(np.mean(err * err))) if len(err) else 999.0
        if debug["pnp_rmse"] > self.cfg.max_pnp_reproj_rmse:
            return None, debug
        R_cw, _ = cv2.Rodrigues(rvec)
        T_cw = se3_from_rt(R_cw, tvec.reshape(3))
        return invert_pose(T_cw), debug

    def _set_latest_pose_from_pnp(self, cur: FrontendResult, T_world_camera: np.ndarray):
        self._set_latest_pose_from_camera(cur.frame_id, T_world_camera, self.keyframes[-2].frame_id)

    def _set_latest_pose_from_camera(self, frame_id: int, T_world_camera: np.ndarray, prev_id: int):
        self.state.poses[frame_id] = T_world_camera
        self.vo_pose_priors[frame_id] = self.state.poses[frame_id].copy()
        self.state.body_poses[frame_id] = self.state.poses[frame_id] @ np.linalg.inv(self.T_body_camera)
        self.relative_pose_priors[(prev_id, frame_id)] = np.linalg.inv(self.state.poses[prev_id]) @ self.state.poses[frame_id]
        dt = max(1e-6, self.frame_timestamps[frame_id] - self.frame_timestamps[prev_id])
        prev_body = self.state.body_poses.get(prev_id, self.state.body_poses[frame_id])
        self.state.velocities[frame_id] = (self.state.body_poses[frame_id][:3, 3] - prev_body[:3, 3]) / dt
        self.biases[frame_id] = self.biases.get(prev_id, self.zero_bias)

    def _set_latest_pose_from_prediction(self, prev_id: int, frame_id: int, predicted_camera: Optional[np.ndarray]):
        T = predicted_camera if predicted_camera is not None else self.state.poses[prev_id].copy()
        self._set_latest_pose_from_camera(frame_id, T, prev_id)

    def _constant_velocity_camera_prior(self, prev_id: int):
        ids = sorted(fid for fid in self.state.poses if fid < prev_id)
        if not ids:
            return None
        prev2_id = ids[-1]
        T_prev2_prev = np.linalg.inv(self.state.poses[prev2_id]) @ self.state.poses[prev_id]
        return self.state.poses[prev_id] @ T_prev2_prev

    def _accept_pose_candidate(self, prev_id: int, T_candidate: np.ndarray, T_prior: Optional[np.ndarray], debug: dict, prefix: str):
        T_prev_cur = np.linalg.inv(self.state.poses[prev_id]) @ T_candidate
        rot_deg = _rotation_deg(T_prev_cur[:3, :3])
        trans = float(np.linalg.norm(T_prev_cur[:3, 3]))
        recent = self._recent_translation_norm(prev_id)
        max_trans = min(self.cfg.max_init_translation_abs, max(0.2, self.cfg.max_init_translation_factor * max(recent, 1e-3)))
        if not self._metric_scale_active():
            max_trans = min(max_trans, self.cfg.vo_max_translation)
        bootstrap_essential = prefix == "essential" and recent < 1e-6
        debug[f"{prefix}_rot_deg"] = float(rot_deg)
        debug[f"{prefix}_trans"] = trans
        debug[f"{prefix}_max_trans"] = float(max_trans)
        if rot_deg > self.cfg.max_init_rotation_deg or (not bootstrap_essential and trans > max_trans):
            debug[f"{prefix}_accepted"] = 0.0
            return False
        if T_prior is not None:
            T_err = np.linalg.inv(T_prior) @ T_candidate
            prior_rot = _rotation_deg(T_err[:3, :3])
            prior_trans = float(np.linalg.norm(T_err[:3, 3]))
            debug[f"{prefix}_prior_rot_deg"] = float(prior_rot)
            debug[f"{prefix}_prior_trans"] = prior_trans
            if prior_rot > self.cfg.max_init_rotation_deg or (not bootstrap_essential and prior_trans > max_trans):
                debug[f"{prefix}_accepted"] = 0.0
                return False
        debug[f"{prefix}_accepted"] = 1.0
        return True

    def _metric_scale_active(self):
        return self.vi_initialized and self.imu is not None and self.cfg.vio_imu_mode == "combined" and not self.cfg.vi_init_disable_scale

    def _recent_translation_norm(self, prev_id: int):
        ids = sorted(fid for fid in self.state.poses if fid < prev_id)
        if not ids:
            return 1.0
        prev2_id = ids[-1]
        return float(np.linalg.norm(self.state.poses[prev_id][:3, 3] - self.state.poses[prev2_id][:3, 3]))


    def _triangulate_new_points(self):
        if len(self.keyframes) < 2:
            return
        for track_id, track in self.keyframes[-1].tracks.items():
            if track_id in self.state.track_to_point:
                continue
            obs = self._track_pose_observations(track)
            if len(obs) < self.cfg.min_track_observations:
                continue
            if self._max_observation_parallax(obs) < self.cfg.min_triangulation_parallax:
                continue
            point = self._triangulate_track(obs)
            if point is None:
                continue
            point_id = self.next_point_id
            self.next_point_id += 1
            self.state.points[point_id] = point
            self.state.track_to_point[track_id] = point_id

    def _track_pose_observations(self, track: Track):
        obs = []
        seen = set()
        for ob in track.observations:
            if ob.frame_id in seen or ob.frame_id not in self.state.poses:
                continue
            seen.add(ob.frame_id)
            obs.append((ob.frame_id, ob.xy.astype(np.float64)))
        return obs

    def _max_observation_parallax(self, obs):
        if len(obs) < 2:
            return 0.0
        pts = np.asarray([xy for _, xy in obs], dtype=np.float64)
        norm = self._undistort_points(pts)
        ref = norm[0]
        focal = float((self.camera.K[0, 0] + self.camera.K[1, 1]) * 0.5)
        return float(np.max(np.linalg.norm(norm[1:] - ref, axis=1)) * focal)

    def _triangulate_track(self, obs):
        A = []
        anchor_id = obs[0][0]
        anchor_Tcw = invert_pose(self.state.poses[anchor_id])[:3]
        for frame_id, xy in obs:
            ray = self._bearing(xy)
            P = invert_pose(self.state.poses[frame_id])[:3]
            A.append(ray[0] * P[2] - ray[2] * P[0])
            A.append(ray[1] * P[2] - ray[2] * P[1])
        A = np.asarray(A, dtype=np.float64)
        if A.shape[0] < 4:
            return None
        _, _, vh = np.linalg.svd(A)
        Xh = vh[-1]
        if abs(Xh[3]) < 1e-12:
            return None
        point = Xh[:3] / Xh[3]
        if not np.isfinite(point).all():
            return None
        depths = []
        for frame_id, _ in obs:
            T_cw = invert_pose(self.state.poses[frame_id])
            depths.append((T_cw[:3, :3] @ point + T_cw[:3, 3])[2])
        if min(depths) <= 0.1:
            return None
        if max(depths) > self.cfg.max_landmark_depth:
            return None
        anchor_depth = (anchor_Tcw[:3, :3] @ point + anchor_Tcw[:3, 3])[2]
        if anchor_depth <= 0.1:
            return None
        if self._max_reprojection_error(point, obs) > self.cfg.max_landmark_reproj_error:
            return None
        return point

    def _project_point(self, frame_id: int, point: np.ndarray):
        T_cw = invert_pose(self.state.poses[frame_id])
        pc = T_cw[:3, :3] @ point + T_cw[:3, 3]
        if pc[2] <= 0.1 or pc[2] > self.cfg.max_landmark_depth:
            return None
        uv, _ = cv2.projectPoints(
            pc.reshape(1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.camera.K,
            self.camera.distortion,
        )
        return uv.reshape(2)

    def _undistort_points(self, pts: np.ndarray):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.camera.distortion is None:
            K_inv = np.linalg.inv(self.camera.K)
            homog = np.c_[pts, np.ones(len(pts), dtype=np.float64)]
            rays = (K_inv @ homog.T).T
            return rays[:, :2] / rays[:, 2:3]
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.camera.K, self.camera.distortion).reshape(-1, 2)

    def _bearing(self, xy: np.ndarray):
        norm = self._undistort_points(np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
        return np.array([norm[0], norm[1], 1.0], dtype=np.float64)

    def _max_reprojection_error(self, point: np.ndarray, obs):
        errors = []
        for frame_id, xy in obs:
            uv = self._project_point(frame_id, point)
            if uv is None:
                return float("inf")
            errors.append(np.linalg.norm(uv - xy))
        return float(max(errors)) if errors else float("inf")

    def _prune_bad_points(self, point_ids, frame_ids):
        if not point_ids:
            return
        obs_by_point = {}
        for fid, pid, xy in self._collect_observations(frame_ids):
            if pid in point_ids:
                obs_by_point.setdefault(pid, []).append((fid, xy))
        removed = 0
        for pid in list(point_ids):
            point = self.state.points.get(pid)
            obs = obs_by_point.get(pid, [])
            if point is None or len(obs) < 2 or self._max_reprojection_error(point, obs) > self.cfg.max_landmark_reproj_error:
                self.state.points.pop(pid, None)
                removed += 1
        if removed:
            dead = {pid for pid in point_ids if pid not in self.state.points}
            for track_id, point_id in list(self.state.track_to_point.items()):
                if point_id in dead:
                    self.state.track_to_point.pop(track_id, None)

    def _optimize_window(self):
        frame_ids = [kf.frame_id for kf in self.keyframes[-self.cfg.window_size :]]
        if len(frame_ids) < 2:
            return
        if self.vi_initialized and self.imu is not None and self.cfg.vio_imu_mode not in {"predict", "predict_align"}:
            if self.cfg.vio_imu_mode == "combined":
                self._optimize_vio_window(frame_ids)
            else:
                self._optimize_gyro_vio_window(frame_ids)
            return
        obs = self._collect_observations(frame_ids)
        if len(obs) < 20:
            return
        obs_frame_ids = {o[0] for o in obs}
        anchored_ids = set(frame_ids[:2])
        frame_ids = [fid for fid in frame_ids if fid in obs_frame_ids or fid in anchored_ids]
        frame_set = set(frame_ids)
        obs = [o for o in obs if o[0] in frame_set]
        if len(frame_ids) < 2 or len(obs) < 20:
            return
        point_ids = sorted({o[1] for o in obs})
        if len(point_ids) > self.cfg.max_ba_points:
            counts = {pid: 0 for pid in point_ids}
            for _, pid, _ in obs:
                counts[pid] += 1
            point_ids = sorted(point_ids, key=lambda pid: -counts[pid])[: self.cfg.max_ba_points]
            point_set = set(point_ids)
            obs = [o for o in obs if o[1] in point_set]
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for fid in frame_ids:
            key = self._x(fid)
            initial.insert(key, self._pose3(self.state.poses[fid]))
        for pid in point_ids:
            initial.insert(self._l(pid), gtsam.Point3(*self.state.points[pid].tolist()))

        graph.add(gtsam.PriorFactorPose3(self._x(frame_ids[0]), self._pose3(self.state.poses[frame_ids[0]]), self.pose_prior_noise))
        if len(frame_ids) > 1:
            graph.add(gtsam.PriorFactorPose3(self._x(frame_ids[1]), self._pose3(self.state.poses[frame_ids[1]]), self.pose_prior_noise))
        if self.cfg.fix_vo_poses:
            for fid in frame_ids:
                graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(self.state.poses[fid]), self.fixed_pose_noise))
        if self.cfg.use_odom_prior:
            for fid in frame_ids[1:]:
                T_prior = self.vo_pose_priors.get(fid)
                if T_prior is not None:
                    graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(T_prior), self.vo_pose_prior_noise))
            for a, b in zip(frame_ids[:-1], frame_ids[1:]):
                T_ab = self.relative_pose_priors.get((a, b))
                if T_ab is not None:
                    graph.add(gtsam.BetweenFactorPose3(self._x(a), self._x(b), self._pose3(T_ab), self.odom_prior_noise))
        for fid, pid, xy in obs:
            graph.add(self._projection_factor(xy, self._x(fid), self._l(pid)))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_ba_iters)
        try:
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        except RuntimeError as exc:
            print(f"[sparse_ba] GTSAM optimize skipped: {exc}")
            return

        if not self.cfg.fix_vo_poses:
            for fid in frame_ids:
                self.state.poses[fid] = result.atPose3(self._x(fid)).matrix()
                self.state.body_poses[fid] = self.state.poses[fid] @ np.linalg.inv(self.T_body_camera)
        for pid in point_ids:
            self.state.points[pid] = np.asarray(result.atPoint3(self._l(pid)), dtype=np.float64)
        self._prune_bad_points(point_ids, frame_ids)

    def _matched_points(self, a: Dict[int, Track], b: Dict[int, Track]):
        ids = sorted(set(a.keys()) & set(b.keys()))
        return (
            np.array([a[i].xy for i in ids], dtype=np.float32),
            np.array([b[i].xy for i in ids], dtype=np.float32),
        )

    def _collect_observations(self, frame_ids):
        frame_set = set(frame_ids)
        latest_tracks = {}
        for kf in self.keyframes[-self.cfg.window_size :]:
            for track_id, track in kf.tracks.items():
                latest_tracks[track_id] = track
        obs = []
        for track_id, point_id in self.state.track_to_point.items():
            if point_id not in self.state.points:
                continue
            track = latest_tracks.get(track_id)
            if track is None:
                continue
            for ob in track.observations:
                if ob.frame_id in frame_set and ob.frame_id in self.state.poses:
                    obs.append((ob.frame_id, point_id, ob.xy.astype(np.float64)))
        return obs

    def _optimize_vio_window(self, frame_ids):
        obs = self._collect_observations(frame_ids)
        if len(obs) < 20:
            return
        point_ids = sorted({o[1] for o in obs})
        if len(point_ids) > self.cfg.max_ba_points:
            counts = {pid: 0 for pid in point_ids}
            for _, pid, _ in obs:
                counts[pid] += 1
            point_ids = sorted(point_ids, key=lambda pid: -counts[pid])[: self.cfg.max_ba_points]
            point_set = set(point_ids)
            obs = [o for o in obs if o[1] in point_set]

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for fid in frame_ids:
            initial.insert(self._x(fid), self._pose3(self.state.body_poses[fid]))
            initial.insert(self._v(fid), np.asarray(self.state.velocities[fid], dtype=np.float64))
            initial.insert(self._b(fid), self.biases.get(fid, self.zero_bias))
        for pid in point_ids:
            initial.insert(self._l(pid), gtsam.Point3(*self.state.points[pid].tolist()))

        first = frame_ids[0]
        graph.add(gtsam.PriorFactorPose3(self._x(first), self._pose3(self.state.body_poses[first]), self.body_prior_noise))
        graph.add(gtsam.PriorFactorVector(self._v(first), self.state.velocities[first], self.velocity_prior_noise))
        graph.add(gtsam.PriorFactorConstantBias(self._b(first), self.biases.get(first, self.zero_bias), self.bias_prior_noise))
        for fid in frame_ids[:-1]:
            if fid in self.pose_priors:
                graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(self.pose_priors[fid]), self.marginal_pose_noise))
            if fid in self.velocity_priors:
                graph.add(gtsam.PriorFactorVector(self._v(fid), self.velocity_priors[fid], self.marginal_velocity_noise))
            if fid in self.bias_priors:
                graph.add(gtsam.PriorFactorConstantBias(self._b(fid), self.bias_priors[fid], self.marginal_bias_noise))
        for pid in point_ids:
            if pid in self.point_priors:
                graph.add(gtsam.PriorFactorPoint3(self._l(pid), gtsam.Point3(*self.point_priors[pid].tolist()), self.marginal_point_noise))
        for fid in frame_ids:
            visual_prior = self.vo_pose_priors.get(fid)
            if visual_prior is not None:
                body_prior = visual_prior @ np.linalg.inv(self.T_body_camera)
                graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(body_prior), self.vio_visual_pose_prior_noise))
        for a, b in zip(frame_ids[:-1], frame_ids[1:]):
            pim = self._preintegrate(a, b, self.biases.get(a, self.zero_bias))
            graph.add(gtsam.CombinedImuFactor(self._x(a), self._v(a), self._x(b), self._v(b), self._b(a), self._b(b), pim))

        for fid, pid, xy in obs:
            graph.add(self._projection_factor(xy, self._x(fid), self._l(pid), self.body_P_sensor))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_ba_iters)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

        max_trans_update = 0.0
        max_rot_update = 0.0
        for fid in frame_ids:
            old_body = self.state.body_poses[fid]
            new_body = result.atPose3(self._x(fid)).matrix()
            delta = np.linalg.inv(old_body) @ new_body
            max_trans_update = max(max_trans_update, float(np.linalg.norm(delta[:3, 3])))
            max_rot_update = max(max_rot_update, _rotation_deg(delta[:3, :3]))
        if max_trans_update > self.cfg.vio_max_pose_update_trans or max_rot_update > self.cfg.vio_max_pose_update_rot_deg:
            print(
                "[sparse_vio] VIO window rejected: "
                f"dtrans={max_trans_update:.3f} drot={max_rot_update:.2f}"
            )
            return

        for fid in frame_ids:
            self.state.body_poses[fid] = result.atPose3(self._x(fid)).matrix()
            self.state.poses[fid] = self.state.body_poses[fid] @ self.T_body_camera
            self.state.velocities[fid] = np.asarray(result.atVector(self._v(fid)), dtype=np.float64)
            self.biases[fid] = result.atConstantBias(self._b(fid))
        for pid in point_ids:
            self.state.points[pid] = np.asarray(result.atPoint3(self._l(pid)), dtype=np.float64)
        self._update_marginal_priors(frame_ids, point_ids)

    def _optimize_gyro_vio_window(self, frame_ids):
        obs = self._collect_observations(frame_ids)
        if len(obs) < 20:
            return
        point_ids = sorted({o[1] for o in obs})
        if len(point_ids) > self.cfg.max_ba_points:
            counts = {pid: 0 for pid in point_ids}
            for _, pid, _ in obs:
                counts[pid] += 1
            point_ids = sorted(point_ids, key=lambda pid: -counts[pid])[: self.cfg.max_ba_points]
            point_set = set(point_ids)
            obs = [o for o in obs if o[1] in point_set]

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for fid in frame_ids:
            initial.insert(self._x(fid), self._pose3(self.state.body_poses[fid]))
        for pid in point_ids:
            initial.insert(self._l(pid), gtsam.Point3(*self.state.points[pid].tolist()))

        first = frame_ids[0]
        graph.add(gtsam.PriorFactorPose3(self._x(first), self._pose3(self.state.body_poses[first]), self.body_prior_noise))
        for fid in frame_ids:
            visual_prior = self.vo_pose_priors.get(fid)
            if visual_prior is not None:
                body_prior = visual_prior @ np.linalg.inv(self.T_body_camera)
                graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(body_prior), self.vio_visual_pose_prior_noise))
            if fid in self.pose_priors:
                graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(self.pose_priors[fid]), self.marginal_pose_noise))
        for pid in point_ids:
            if pid in self.point_priors:
                graph.add(gtsam.PriorFactorPoint3(self._l(pid), gtsam.Point3(*self.point_priors[pid].tolist()), self.marginal_point_noise))
        for a, b in zip(frame_ids[:-1], frame_ids[1:]):
            pim = self._preintegrate(a, b, self.biases.get(a, self.zero_bias))
            T_ab = np.eye(4, dtype=np.float64)
            T_ab[:3, :3] = np.asarray(pim.deltaRij().matrix(), dtype=np.float64)
            graph.add(gtsam.BetweenFactorPose3(self._x(a), self._x(b), self._pose3(T_ab), self.gyro_between_noise))

        for fid, pid, xy in obs:
            graph.add(self._projection_factor(xy, self._x(fid), self._l(pid), self.body_P_sensor))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_ba_iters)
        try:
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        except RuntimeError as exc:
            print(f"[sparse_vio] gyro VIO optimize skipped: {exc}")
            return

        max_trans_update = 0.0
        max_rot_update = 0.0
        for fid in frame_ids:
            old_body = self.state.body_poses[fid]
            new_body = result.atPose3(self._x(fid)).matrix()
            delta = np.linalg.inv(old_body) @ new_body
            max_trans_update = max(max_trans_update, float(np.linalg.norm(delta[:3, 3])))
            max_rot_update = max(max_rot_update, _rotation_deg(delta[:3, :3]))
        if max_trans_update > self.cfg.vio_max_pose_update_trans or max_rot_update > self.cfg.vio_max_pose_update_rot_deg:
            print(
                "[sparse_vio] gyro VIO window rejected: "
                f"dtrans={max_trans_update:.3f} drot={max_rot_update:.2f}"
            )
            return

        for fid in frame_ids:
            self.state.body_poses[fid] = result.atPose3(self._x(fid)).matrix()
            self.state.poses[fid] = self.state.body_poses[fid] @ self.T_body_camera
            prev_ids = [x for x in self.state.body_poses if x < fid]
            if prev_ids:
                prev_id = max(prev_ids)
                dt = max(1e-6, self.frame_timestamps[fid] - self.frame_timestamps[prev_id])
                self.state.velocities[fid] = (self.state.body_poses[fid][:3, 3] - self.state.body_poses[prev_id][:3, 3]) / dt
        for pid in point_ids:
            self.state.points[pid] = np.asarray(result.atPoint3(self._l(pid)), dtype=np.float64)
        self._prune_bad_points(point_ids, frame_ids)
        self._update_marginal_priors(frame_ids, point_ids)

    def _try_vi_initialize(self):
        frame_ids = [kf.frame_id for kf in self.init_keyframes]
        preintegrations = [self._preintegrate(a, b, self.zero_bias) for a, b in zip(frame_ids[:-1], frame_ids[1:])]
        wTcs = np.asarray([self.state.poses[fid] for fid in frame_ids], dtype=np.float64)
        try:
            vi = VisualIMUAlignment(
                self.T_body_camera,
                wTcs,
                preintegrations,
                ignore_lever=True,
                disable_scale=self.cfg.vi_init_disable_scale,
            )
        except np.linalg.LinAlgError as exc:
            print(f"[sparse_vio] VI init skipped: {exc}")
            return
        scale = float(vi["s"])
        bg_norm = float(np.linalg.norm(vi["bs"][0].gyroscope()))
        if not self.cfg.vi_init_disable_scale and (scale < self.cfg.vi_init_min_scale or scale > self.cfg.vi_init_max_scale):
            print(f"[sparse_vio] VI init rejected: scale={scale:.6f}")
            return
        if bg_norm > self.cfg.vi_init_max_gyro_bias:
            print(f"[sparse_vio] VI init rejected: gyro_bias_norm={bg_norm:.6f}")
            return
        old_first = self.state.poses[frame_ids[0]].copy()
        for fid, T_world_body, velocity, bias in zip(frame_ids, vi["wTbs"], vi["vs"], vi["bs"]):
            self.state.body_poses[fid] = np.asarray(T_world_body, dtype=np.float64)
            self.state.poses[fid] = self.state.body_poses[fid] @ self.T_body_camera
            self.state.velocities[fid] = np.asarray(velocity, dtype=np.float64)
            self.biases[fid] = bias
        new_first = self.state.poses[frame_ids[0]]
        R_align = new_first[:3, :3] @ old_first[:3, :3].T
        t_align = new_first[:3, 3] - R_align @ (scale * old_first[:3, 3])
        point_scale = 1.0 if self.cfg.vi_init_disable_scale else scale
        for pid, point in list(self.state.points.items()):
            self.state.points[pid] = R_align @ (point_scale * point) + t_align
        self.vi_initialized = True
        self._update_marginal_priors(frame_ids, list(self.state.points.keys()))
        print(f"[sparse_vio] VI initialized: scale={scale:.6f} keyframes={len(frame_ids)}")

    def _vi_init_ready(self):
        if len(self.init_keyframes) < self.cfg.vi_init_min_keyframes:
            return False
        if len(self.state.points) < self.cfg.vi_init_min_points:
            return False
        frame_ids = [kf.frame_id for kf in self.init_keyframes]
        centers = np.asarray([self.state.poses[fid][:3, 3] for fid in frame_ids if fid in self.state.poses], dtype=np.float64)
        if centers.shape[0] < self.cfg.vi_init_min_keyframes:
            return False
        baseline = float(np.linalg.norm(centers - centers[0], axis=1).max())
        if baseline < self.cfg.vi_init_min_baseline:
            return False
        return True

    def _update_marginal_priors(self, frame_ids, point_ids):
        for fid in frame_ids:
            if fid in self.state.body_poses:
                self.pose_priors[fid] = self.state.body_poses[fid].copy()
            if fid in self.state.velocities:
                self.velocity_priors[fid] = self.state.velocities[fid].copy()
            if fid in self.biases:
                self.bias_priors[fid] = self.biases[fid]
        for pid in point_ids:
            if pid in self.state.points:
                self.point_priors[pid] = self.state.points[pid].copy()

    def _predict_with_imu(self, prev: FrontendResult, cur: FrontendResult):
        if prev.frame_id not in self.state.body_poses:
            return None
        bias = self.biases.get(prev.frame_id, self.zero_bias)
        pim = self._preintegrate(prev.frame_id, cur.frame_id, bias)
        nav = gtsam.NavState(self._pose3(self.state.body_poses[prev.frame_id]), self.state.velocities[prev.frame_id])
        predicted = pim.predict(nav, bias)
        return predicted.pose().matrix(), np.asarray(predicted.velocity(), dtype=np.float64)

    def _predict_camera_with_gyro(self, prev: FrontendResult, cur: FrontendResult, cv_camera_prior: Optional[np.ndarray]):
        if prev.frame_id not in self.state.body_poses:
            return cv_camera_prior
        base_body = (
            cv_camera_prior @ np.linalg.inv(self.T_body_camera)
            if cv_camera_prior is not None
            else self.state.body_poses[prev.frame_id].copy()
        )
        pim = self._preintegrate(prev.frame_id, cur.frame_id, self.biases.get(prev.frame_id, self.zero_bias))
        base_body[:3, :3] = self.state.body_poses[prev.frame_id][:3, :3] @ np.asarray(pim.deltaRij().matrix(), dtype=np.float64)
        return base_body @ self.T_body_camera

    def _preintegrate(self, frame_i: int, frame_j: int, bias):
        pim = gtsam.PreintegratedCombinedMeasurements(self.imu_params, bias)
        for rec in self.imu.records(self.frame_timestamps[frame_i], self.frame_timestamps[frame_j]):
            pim.integrateMeasurement(rec.accel, rec.gyro, rec.t1 - rec.t0)
        return pim

    def _make_imu_params(self):
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.cfg.gravity)
        params.setAccelerometerCovariance(np.eye(3) * self.cfg.accel_noise_sigma**2)
        params.setGyroscopeCovariance(np.eye(3) * self.cfg.gyro_noise_sigma**2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)
        params.setBiasAccCovariance(np.eye(3) * self.cfg.accel_bias_rw_sigma**2)
        params.setBiasOmegaCovariance(np.eye(3) * self.cfg.gyro_bias_rw_sigma**2)
        params.setBiasAccOmegaInit(np.eye(6) * 1e-5)
        return params

    def _x(self, frame_id: int):
        return gtsam.symbol("x", int(frame_id))

    def _v(self, frame_id: int):
        return gtsam.symbol("v", int(frame_id))

    def _b(self, frame_id: int):
        return gtsam.symbol("b", int(frame_id))

    def _l(self, point_id: int):
        return gtsam.symbol("l", int(point_id))

    def _pose3(self, T_wc: np.ndarray):
        return gtsam.Pose3(np.asarray(T_wc, dtype=np.float64))

    def _projection_factor(self, xy, pose_key, point_key, body_P_sensor=None):
        args = (
            gtsam.Point2(float(xy[0]), float(xy[1])),
            self.pixel_noise,
            pose_key,
            point_key,
            self.calib,
        )
        if body_P_sensor is None:
            return self.projection_factor_cls(*args)
        return self.projection_factor_cls(*args, body_P_sensor)

    def finalize(self):
        if self.cfg.final_global_ba:
            self._optimize_global(include_imu=self.imu is not None and self.vi_initialized)

    def _optimize_global(self, include_imu: bool):
        frame_ids = [kf.frame_id for kf in self.init_keyframes if kf.frame_id in self.state.poses]
        if len(frame_ids) < 3:
            return
        obs = self._collect_observations_from_keyframes(self.init_keyframes, frame_ids)
        if len(obs) < 50:
            return
        point_ids = sorted({o[1] for o in obs if o[1] in self.state.points})
        if len(point_ids) > self.cfg.max_global_ba_points:
            counts = {pid: 0 for pid in point_ids}
            for _, pid, _ in obs:
                if pid in counts:
                    counts[pid] += 1
            point_ids = sorted(point_ids, key=lambda pid: -counts[pid])[: self.cfg.max_global_ba_points]
            point_set = set(point_ids)
            obs = [o for o in obs if o[1] in point_set]
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        if include_imu:
            for fid in frame_ids:
                initial.insert(self._x(fid), self._pose3(self.state.body_poses[fid]))
                initial.insert(self._v(fid), np.asarray(self.state.velocities[fid], dtype=np.float64))
                initial.insert(self._b(fid), self.biases.get(fid, self.zero_bias))
        else:
            for fid in frame_ids:
                initial.insert(self._x(fid), self._pose3(self.state.poses[fid]))
        for pid in point_ids:
            initial.insert(self._l(pid), gtsam.Point3(*self.state.points[pid].tolist()))

        if include_imu:
            first = frame_ids[0]
            graph.add(gtsam.PriorFactorPose3(self._x(first), self._pose3(self.state.body_poses[first]), self.body_prior_noise))
            graph.add(gtsam.PriorFactorVector(self._v(first), self.state.velocities[first], self.velocity_prior_noise))
            graph.add(gtsam.PriorFactorConstantBias(self._b(first), self.biases.get(first, self.zero_bias), self.bias_prior_noise))
            for fid in frame_ids[:-1]:
                if fid in self.pose_priors:
                    graph.add(gtsam.PriorFactorPose3(self._x(fid), self._pose3(self.pose_priors[fid]), self.marginal_pose_noise))
                if fid in self.velocity_priors:
                    graph.add(gtsam.PriorFactorVector(self._v(fid), self.velocity_priors[fid], self.marginal_velocity_noise))
                if fid in self.bias_priors:
                    graph.add(gtsam.PriorFactorConstantBias(self._b(fid), self.bias_priors[fid], self.marginal_bias_noise))
            for a, b in zip(frame_ids[:-1], frame_ids[1:]):
                pim = self._preintegrate(a, b, self.biases.get(a, self.zero_bias))
                graph.add(gtsam.CombinedImuFactor(self._x(a), self._v(a), self._x(b), self._v(b), self._b(a), self._b(b), pim))
        else:
            graph.add(gtsam.PriorFactorPose3(self._x(frame_ids[0]), self._pose3(self.state.poses[frame_ids[0]]), self.pose_prior_noise))
            graph.add(gtsam.PriorFactorPose3(self._x(frame_ids[1]), self._pose3(self.state.poses[frame_ids[1]]), self.pose_prior_noise))
        for fid, pid, xy in obs:
            if include_imu:
                graph.add(self._projection_factor(xy, self._x(fid), self._l(pid), self.body_P_sensor))
            else:
                graph.add(self._projection_factor(xy, self._x(fid), self._l(pid)))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_global_ba_iters)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        if include_imu:
            for fid in frame_ids:
                self.state.body_poses[fid] = result.atPose3(self._x(fid)).matrix()
                self.state.poses[fid] = self.state.body_poses[fid] @ self.T_body_camera
                self.state.velocities[fid] = np.asarray(result.atVector(self._v(fid)), dtype=np.float64)
                self.biases[fid] = result.atConstantBias(self._b(fid))
        else:
            for fid in frame_ids:
                self.state.poses[fid] = result.atPose3(self._x(fid)).matrix()
                self.state.body_poses[fid] = self.state.poses[fid] @ np.linalg.inv(self.T_body_camera)
        for pid in point_ids:
            self.state.points[pid] = np.asarray(result.atPoint3(self._l(pid)), dtype=np.float64)
        print(
            f"[sparse_ba] final global BA use_imu={int(include_imu)} frames={len(frame_ids)} "
            f"points={len(point_ids)} obs={len(obs)}"
        )

    def _collect_observations_from_keyframes(self, keyframes, frame_ids):
        frame_set = set(frame_ids)
        latest_tracks = {}
        for kf in keyframes:
            for track_id, track in kf.tracks.items():
                latest_tracks[track_id] = track
        obs = []
        for track_id, point_id in self.state.track_to_point.items():
            if point_id not in self.state.points:
                continue
            track = latest_tracks.get(track_id)
            if track is None:
                continue
            for ob in track.observations:
                if ob.frame_id in frame_set and ob.frame_id in self.state.poses:
                    obs.append((ob.frame_id, point_id, ob.xy.astype(np.float64)))
        return obs


def _rotation_deg(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))
