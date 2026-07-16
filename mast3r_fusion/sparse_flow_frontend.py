from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch


# The Shi-Tomasi/LK implementation below follows pySLAM commit
# 8a7b88b9e6be3f823d4330122ab6c7b3dcc89b8c. pySLAM is Copyright (C)
# 2016-present Luigi Freda and licensed under GPLv3.


class FeatureDetectorTypes(Enum):
    SHI_TOMASI = 1


class FeatureDescriptorTypes(Enum):
    NONE = 0


class FeatureTrackerTypes(Enum):
    LK = 0


class FeatureTrackingResult:
    def __init__(self):
        self.kps_ref = None
        self.kps_cur = None
        self.des_ref = None
        self.des_cur = None
        self.idxs_ref = None
        self.idxs_cur = None
        self.kps_ref_matched = None
        self.kps_cur_matched = None


class ShiTomasiDetector:
    def __init__(self, num_features=2000, quality_level=0.01, min_coner_distance=3):
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance
        self.blockSize = 5

    def setMaxFeatures(self, num_features):
        self.num_features = num_features

    def detect(self, frame, mask=None):
        pts = cv2.goodFeaturesToTrack(
            frame,
            self.num_features,
            self.quality_level,
            self.min_coner_distance,
            blockSize=self.blockSize,
            mask=mask,
        )
        if pts is not None:
            kps = [cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts]
        else:
            kps = []
        return kps


class ShiTomasiFeatureManager:
    def __init__(self, num_features, num_levels):
        self.num_features = num_features
        self.num_levels = num_levels
        self.detector = ShiTomasiDetector(num_features=num_features)

    def detect(self, frame, mask=None):
        return self.detector.detect(frame, mask)


class LkFeatureTracker:
    def __init__(
        self,
        num_features=2000,
        num_levels=3,
        detector_type=FeatureDetectorTypes.SHI_TOMASI,
        descriptor_type=FeatureDescriptorTypes.NONE,
        tracker_type=FeatureTrackerTypes.LK,
    ):
        if detector_type != FeatureDetectorTypes.SHI_TOMASI:
            raise ValueError("The pySLAM LK frontend only supports Shi-Tomasi.")
        if descriptor_type != FeatureDescriptorTypes.NONE:
            raise ValueError("The pySLAM LK frontend does not use descriptors.")
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type
        self.feature_manager = ShiTomasiFeatureManager(num_features, num_levels)

        optic_flow_num_levels = max(3, num_levels)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=optic_flow_num_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @property
    def num_features(self):
        return self.feature_manager.num_features

    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detect(frame, mask), None

    def track(self, image_ref, image_cur, kps_ref, des_ref=None, mask_ref=None, mask_cur=None):
        del des_ref, mask_ref, mask_cur
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(
            image_ref, image_cur, kps_ref, None, **self.lk_params
        )
        del err
        if kps_cur is None or st is None:
            res = FeatureTrackingResult()
            res.idxs_ref = []
            res.idxs_cur = []
            res.kps_ref_matched = np.empty((0, 2), dtype=np.float32)
            res.kps_cur_matched = np.empty((0, 2), dtype=np.float32)
            res.kps_ref = res.kps_ref_matched
            res.kps_cur = res.kps_cur_matched
            res.des_ref = None
            res.des_cur = None
            return res
        st = st.reshape(st.shape[0])
        res = FeatureTrackingResult()
        res.idxs_ref = [i for i, v in enumerate(st) if v == 1]
        res.idxs_cur = res.idxs_ref.copy()
        res.kps_ref_matched = kps_ref[res.idxs_ref]
        res.kps_cur_matched = kps_cur[res.idxs_cur]
        res.kps_ref = res.kps_ref_matched
        res.kps_cur = res.kps_cur_matched
        res.des_ref = None
        res.des_cur = None
        return res


@dataclass
class SparseFlowConfig:
    max_features: int = 2000
    keyframe_parallax: float = 20.0
    keyframe_gap: int = 5


@dataclass
class SparseFlowResult:
    frame_id: int
    image: np.ndarray
    pts: np.ndarray
    prev_pts: np.ndarray
    track_cnt: np.ndarray
    ids: np.ndarray
    inlier_mask: np.ndarray
    new_keyframe: bool
    debug: dict


class SparseFlowFrontend:
    """Adapter from MASt3R-Fusion frames to pySLAM's Shi-Tomasi/LK tracker."""

    def __init__(self, K: np.ndarray, width: int, height: int, cfg: Optional[SparseFlowConfig] = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.cfg = cfg or SparseFlowConfig()
        self.tracker = LkFeatureTracker(
            num_features=self.cfg.max_features,
            num_levels=3,
        )
        self.ref_gray = None
        self.ref_pts = np.empty((0, 2), dtype=np.float32)
        self.ref_ids = np.empty((0,), dtype=np.int64)
        self.ref_ages = np.empty((0,), dtype=np.int32)
        self.next_track_id = 0

    @classmethod
    def from_config(cls, K: np.ndarray, width: int, height: int, cfg_dict: dict):
        cfg = SparseFlowConfig(
            max_features=2000,
            keyframe_parallax=float(cfg_dict.get("keyframe_parallax", 20.0)),
            keyframe_gap=int(cfg_dict.get("keyframe_gap", 5)),
        )
        return cls(K, width, height, cfg)

    def process_frame(
        self,
        frame,
        timestamp=None,
        gyro_R: Optional[np.ndarray] = None,
        frames_since_keyframe=None,
    ) -> SparseFlowResult:
        del timestamp, gyro_R
        image = self._frame_rgb(frame)
        gray = self._gray(image)

        if self.ref_gray is None or self.ref_pts.shape[0] == 0:
            self._redetect(gray)
            prev_pts = np.full_like(self.ref_pts, np.nan)
            inlier_mask = np.ones(self.ref_pts.shape[0], dtype=bool)
            return self._result(frame.frame_id, image, self.ref_pts, prev_pts, self.ref_ages, self.ref_ids, inlier_mask, True, 0.0, frames_since_keyframe)

        tracked = self.tracker.track(self.ref_gray, gray, self.ref_pts)
        idxs_ref = np.asarray(tracked.idxs_ref, dtype=np.int64)
        prev_pts = tracked.kps_ref_matched.astype(np.float32)
        cur_pts = tracked.kps_cur_matched.astype(np.float32)
        ids = self.ref_ids[idxs_ref]
        ages = self.ref_ages[idxs_ref] + 1
        inlier_mask = self._estimate_pose_inliers(prev_pts, cur_pts)
        avg_parallax = (
            float(np.mean(np.abs(prev_pts - cur_pts)))
            if cur_pts.shape[0] > 0
            else 0.0
        )
        gap = 0 if frames_since_keyframe is None else int(frames_since_keyframe)
        new_keyframe = (
            cur_pts.shape[0] < 20
            or avg_parallax >= self.cfg.keyframe_parallax
            or gap >= self.cfg.keyframe_gap
        )
        result = self._result(
            frame.frame_id,
            image,
            cur_pts,
            prev_pts,
            ages,
            ids,
            inlier_mask,
            new_keyframe,
            avg_parallax,
            frames_since_keyframe,
        )

        # Match pySLAM visual_odometry.py: once tracks drop below the requested
        # count, redetect a complete reference set instead of merging points.
        if tracked.kps_cur.shape[0] < self.tracker.num_features:
            self._redetect(gray)
        else:
            self.ref_gray = gray
            self.ref_pts = cur_pts
            self.ref_ids = ids
            self.ref_ages = ages
        return result

    def draw_overlay(self, result: SparseFlowResult) -> np.ndarray:
        image = result.image.copy()
        for cur_xy, ref_xy, is_inlier in zip(
            result.pts, result.prev_pts, result.inlier_mask
        ):
            if not is_inlier:
                continue
            cur_pt = tuple(np.rint(cur_xy).astype(int))
            if np.all(np.isfinite(ref_xy)):
                ref_pt = tuple(np.rint(ref_xy).astype(int))
                cv2.line(image, ref_pt, cur_pt, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(image, ref_pt, 1, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(image, cur_pt, 1, (0, 255, 0), -1, cv2.LINE_AA)
        return image

    def _redetect(self, gray):
        keypoints, _ = self.tracker.detectAndCompute(gray)
        self.ref_gray = gray
        self.ref_pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
        count = self.ref_pts.shape[0]
        self.ref_ids = np.arange(self.next_track_id, self.next_track_id + count, dtype=np.int64)
        self.ref_ages = np.ones(count, dtype=np.int32)
        self.next_track_id += count

    def _estimate_pose_inliers(self, ref_pts, cur_pts):
        if ref_pts.shape[0] < 5:
            return np.zeros(ref_pts.shape[0], dtype=bool)
        ref_norm = cv2.undistortPoints(
            np.ascontiguousarray(ref_pts).reshape(-1, 1, 2), self.K, None
        ).reshape(-1, 2)
        cur_norm = cv2.undistortPoints(
            np.ascontiguousarray(cur_pts).reshape(-1, 1, 2), self.K, None
        ).reshape(-1, 2)
        try:
            essential, mask = cv2.findEssentialMat(
                cur_norm,
                ref_norm,
                focal=1,
                pp=(0.0, 0.0),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=0.0004,
            )
        except cv2.error:
            return np.zeros(ref_pts.shape[0], dtype=bool)
        essential = None if essential is None else np.asarray(essential)
        if essential is None or essential.size < 9 or mask is None:
            return np.zeros(ref_pts.shape[0], dtype=bool)
        try:
            cv2.recoverPose(
                np.ascontiguousarray(essential),
                cur_norm,
                ref_norm,
                focal=1,
                pp=(0.0, 0.0),
            )
        except cv2.error:
            return np.zeros(ref_pts.shape[0], dtype=bool)
        return mask.reshape(-1).astype(bool)

    def _result(
        self,
        frame_id,
        image,
        pts,
        prev_pts,
        ages,
        ids,
        inlier_mask,
        new_keyframe,
        avg_parallax,
        frames_since_keyframe,
    ):
        debug = {
            "track_num": float(pts.shape[0]),
            "last_track_num": float(pts.shape[0]),
            "long_track_num": float(np.count_nonzero(ages >= 4)),
            "inlier_num": float(np.count_nonzero(inlier_mask)),
            "inlier_fraction": float(np.mean(inlier_mask)) if inlier_mask.size else 0.0,
            "avg_parallax": avg_parallax,
            "age_median": float(np.median(ages)) if ages.size else 0.0,
        }
        if frames_since_keyframe is not None:
            debug["frames_since_keyframe"] = float(frames_since_keyframe)
        return SparseFlowResult(
            frame_id=frame_id,
            image=image,
            pts=pts.copy(),
            prev_pts=prev_pts.copy(),
            track_cnt=ages.copy(),
            ids=ids.copy(),
            inlier_mask=inlier_mask.copy(),
            new_keyframe=new_keyframe,
            debug=debug,
        )

    def _frame_rgb(self, frame) -> np.ndarray:
        image = frame.uimg.detach().cpu().numpy()
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def overlay_to_uimg_tensor(image: np.ndarray, dtype=torch.float32):
    return torch.from_numpy(image.copy()).to(dtype=dtype) / 255.0
