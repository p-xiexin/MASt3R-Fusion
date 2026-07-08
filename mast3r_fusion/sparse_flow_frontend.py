from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch


@dataclass
class SparseFlowConfig:
    max_features: int = 400
    min_distance: int = 15
    quality: float = 0.01
    lk_win: int = 21
    lk_levels: int = 3
    f_ransac_thresh: float = 1.5
    keyframe_parallax: float = 20.0
    keyframe_gap: int = 5
    equalize: bool = True
    use_fast: bool = False
    reject_with_f: bool = True
    border_size: int = 1
    use_imu_prior: bool = True


@dataclass
class SparseFlowResult:
    frame_id: int
    image: np.ndarray
    pts: np.ndarray
    prev_pts: np.ndarray
    track_cnt: np.ndarray
    ids: np.ndarray
    new_keyframe: bool
    debug: dict


class SparseFlowFrontend:
    def __init__(self, K: np.ndarray, width: int, height: int, cfg: Optional[SparseFlowConfig] = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.width = int(width)
        self.height = int(height)
        self.cfg = cfg or SparseFlowConfig()
        self.cur_gray = None
        self.cur_pts = np.empty((0, 2), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)
        self.track_cnt = np.empty((0,), dtype=np.int32)
        self.next_track_id = 0
        self.last_keyframe_id = 0
        self.debug = {}
        self.focal = float((self.K[0, 0] + self.K[1, 1]) * 0.5)

    @classmethod
    def from_config(cls, K: np.ndarray, width: int, height: int, cfg_dict: dict):
        cfg = SparseFlowConfig(
            max_features=int(cfg_dict.get("max_features", 400)),
            min_distance=int(cfg_dict.get("min_distance", 15)),
            quality=float(cfg_dict.get("quality", 0.01)),
            lk_win=int(cfg_dict.get("lk_win", 21)),
            lk_levels=int(cfg_dict.get("lk_levels", 3)),
            f_ransac_thresh=float(cfg_dict.get("f_ransac_thresh", 1.5)),
            keyframe_parallax=float(cfg_dict.get("keyframe_parallax", 20.0)),
            keyframe_gap=int(cfg_dict.get("keyframe_gap", 5)),
            equalize=bool(cfg_dict.get("equalize", True)),
            use_fast=bool(cfg_dict.get("use_fast", False)),
            reject_with_f=bool(cfg_dict.get("reject_with_f", True)),
            border_size=int(cfg_dict.get("border_size", 1)),
            use_imu_prior=bool(cfg_dict.get("use_imu_prior", True)),
        )
        return cls(K, width, height, cfg)

    def process_frame(self, frame, timestamp=None, gyro_R: Optional[np.ndarray] = None) -> SparseFlowResult:
        image = self._frame_rgb(frame)
        gray = self._gray(image)
        self.debug = {}

        if self.cur_gray is None:
            draw_prev_pts = np.empty((0, 2), dtype=np.float32)
            self._spawn_points(gray)
            self.cur_gray = gray
            return self._result(frame.frame_id, image, draw_prev_pts, True)

        draw_prev_pts = self.cur_pts.copy()
        self._track_by_lk(gray, draw_prev_pts, gyro_R)
        draw_prev_pts = getattr(self, "_last_draw_prev_pts", draw_prev_pts[: self.cur_pts.shape[0]])
        if self.cfg.reject_with_f:
            draw_prev_pts = self._reject_with_f(draw_prev_pts)
        self._set_mask()
        draw_prev_pts = self._reorder_draw_prev_pts(draw_prev_pts)
        self._spawn_points(gray)

        new_keyframe = self._is_keyframe(frame.frame_id, draw_prev_pts)
        if new_keyframe:
            self.last_keyframe_id = frame.frame_id
        self.cur_gray = gray
        return self._result(frame.frame_id, image, draw_prev_pts, new_keyframe)

    def draw_overlay(self, result: SparseFlowResult) -> np.ndarray:
        image = result.image.copy()
        for xy, prev_xy, age in zip(result.pts, result.prev_pts, result.track_cnt):
            color = self._age_color(int(age))
            p1 = tuple(np.rint(xy).astype(int))
            cv2.circle(image, p1, 2, color, -1, cv2.LINE_AA)
            if age > 1 and np.all(np.isfinite(prev_xy)):
                p0 = tuple(np.rint(prev_xy).astype(int))
                cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)
        return image

    def _track_by_lk(self, gray, draw_prev_pts, gyro_R):
        if self.cur_pts.shape[0] == 0:
            return
        before = self.cur_pts.shape[0]
        init = None
        flags = 0
        if self.cfg.use_imu_prior and gyro_R is not None:
            init = self._gyro_predict(self.cur_pts, gyro_R)
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        cur, st, _ = cv2.calcOpticalFlowPyrLK(
            self.cur_gray,
            gray,
            self.cur_pts.reshape(-1, 1, 2),
            None if init is None else np.ascontiguousarray(init.reshape(-1, 1, 2), dtype=np.float32),
            winSize=(self.cfg.lk_win, self.cfg.lk_win),
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=flags,
        )
        if cur is None or st is None:
            self._clear_tracks()
            return
        cur = cur.reshape(-1, 2)
        valid = st.reshape(-1).astype(bool) & self._inside(cur)

        self.cur_pts = cur[valid].astype(np.float32)
        self.ids = self.ids[valid]
        self.track_cnt = self.track_cnt[valid] + 1
        self._last_draw_prev_pts = draw_prev_pts[valid].astype(np.float32)
        self.debug["lk_before"] = float(before)
        self.debug["lk_kept"] = float(valid.sum())

    def _reject_with_f(self, draw_prev_pts):
        if self.cur_pts.shape[0] < 8:
            return draw_prev_pts[: self.cur_pts.shape[0]]

        aligned_prev = draw_prev_pts[: self.cur_pts.shape[0]]
        prev_virtual = self._virtual_pixel_points(aligned_prev)
        cur_virtual = self._virtual_pixel_points(self.cur_pts)
        _, mask = cv2.findFundamentalMat(
            prev_virtual,
            cur_virtual,
            cv2.FM_RANSAC,
            self.cfg.f_ransac_thresh,
            0.99,
        )
        if mask is None:
            return aligned_prev
        keep = mask.reshape(-1).astype(bool)
        before = self.cur_pts.shape[0]
        self.cur_pts = self.cur_pts[keep]
        self.ids = self.ids[keep]
        self.track_cnt = self.track_cnt[keep]
        aligned_prev = aligned_prev[keep]
        self.debug["f_before"] = float(before)
        self.debug["f_kept"] = float(keep.sum())
        return aligned_prev

    def _set_mask(self):
        mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        if self.cur_pts.shape[0] == 0:
            self.mask = mask
            self._sort_order = np.empty((0,), dtype=np.int64)
            return

        order = np.argsort(-self.track_cnt)
        kept = []
        for idx in order:
            x, y = np.rint(self.cur_pts[idx]).astype(int)
            if 0 <= x < self.width and 0 <= y < self.height and mask[y, x] == 255:
                kept.append(idx)
                cv2.circle(mask, (x, y), self.cfg.min_distance, 0, -1)
        kept = np.asarray(kept, dtype=np.int64)
        self.cur_pts = self.cur_pts[kept]
        self.ids = self.ids[kept]
        self.track_cnt = self.track_cnt[kept]
        self.mask = mask
        self._sort_order = kept

    def _reorder_draw_prev_pts(self, draw_prev_pts):
        aligned_prev = draw_prev_pts
        order = getattr(self, "_sort_order", np.arange(self.cur_pts.shape[0]))
        if order.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        if aligned_prev.shape[0] >= np.max(order) + 1:
            return aligned_prev[order].astype(np.float32)
        return aligned_prev[: self.cur_pts.shape[0]].astype(np.float32)

    def _spawn_points(self, gray):
        need = self.cfg.max_features - self.cur_pts.shape[0]
        if need <= 0:
            return
        if not hasattr(self, "mask"):
            self.mask = np.full(gray.shape, 255, dtype=np.uint8)
        pts = self._detect(gray, self.mask, need)
        if pts is None:
            self.debug["new_feature_num"] = 0.0
            return
        pts = pts.reshape(-1, 2).astype(np.float32)
        new_ids = np.arange(self.next_track_id, self.next_track_id + pts.shape[0], dtype=np.int64)
        self.next_track_id += pts.shape[0]
        self.cur_pts = np.vstack([self.cur_pts, pts]).astype(np.float32)
        self.ids = np.concatenate([self.ids, new_ids])
        self.track_cnt = np.concatenate([self.track_cnt, np.ones(pts.shape[0], dtype=np.int32)])
        self.debug["new_feature_num"] = float(pts.shape[0])

    def _detect(self, gray, mask, max_count):
        if self.cfg.use_fast:
            fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
            kps = fast.detect(gray, mask)
            kps = sorted(kps, key=lambda k: -k.response)[:max_count]
            if not kps:
                return None
            return np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_count,
            qualityLevel=self.cfg.quality,
            minDistance=self.cfg.min_distance,
            mask=mask,
            blockSize=7,
        )

    def _is_keyframe(self, frame_id, draw_prev_pts):
        tracked = self.track_cnt > 1
        last_track_num = int(np.count_nonzero(tracked))
        long_track_num = int(np.count_nonzero(self.track_cnt >= 4))
        parallax = []
        if draw_prev_pts.shape[0] > 0 and self.cur_pts.shape[0] > 0:
            n = min(draw_prev_pts.shape[0], self.cur_pts.shape[0])
            tracked_n = tracked[:n]
            if np.any(tracked_n):
                cur_norm = self._undistort_points(self.cur_pts[:n][tracked_n])
                prev_norm = self._undistort_points(draw_prev_pts[:n][tracked_n])
                parallax = np.linalg.norm(cur_norm - prev_norm, axis=1) * self.focal

        avg_parallax = float(np.mean(parallax)) if len(parallax) else 0.0
        self.debug["last_track_num"] = float(last_track_num)
        self.debug["long_track_num"] = float(long_track_num)
        self.debug["avg_parallax"] = avg_parallax

        if frame_id < 2 or last_track_num < 20 or long_track_num < 40:
            return True
        if self.debug.get("new_feature_num", 0.0) > 0.5 * max(1, last_track_num):
            return True
        if frame_id - self.last_keyframe_id >= self.cfg.keyframe_gap:
            return True
        return avg_parallax >= self.cfg.keyframe_parallax

    def _result(self, frame_id, image, draw_prev_pts, new_keyframe):
        if draw_prev_pts.shape[0] < self.cur_pts.shape[0]:
            pad = np.full((self.cur_pts.shape[0] - draw_prev_pts.shape[0], 2), np.nan, dtype=np.float32)
            draw_prev_pts = np.vstack([draw_prev_pts, pad])
        ages = self.track_cnt
        debug = dict(self.debug)
        debug["age_median"] = float(np.median(ages)) if ages.size else 0.0
        debug["age_gt10"] = float(np.count_nonzero(ages > 10))
        debug["track_num"] = float(self.cur_pts.shape[0])
        return SparseFlowResult(
            frame_id=frame_id,
            image=image,
            pts=self.cur_pts.copy(),
            prev_pts=draw_prev_pts[: self.cur_pts.shape[0]].copy(),
            track_cnt=self.track_cnt.copy(),
            ids=self.ids.copy(),
            new_keyframe=new_keyframe,
            debug=debug,
        )

    def _frame_rgb(self, frame) -> np.ndarray:
        image = frame.uimg.detach().cpu().numpy()
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.cfg.equalize:
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        return gray

    def _inside(self, pts):
        b = self.cfg.border_size
        return (
            (pts[:, 0] >= b)
            & (pts[:, 0] < self.width - b)
            & (pts[:, 1] >= b)
            & (pts[:, 1] < self.height - b)
        )

    def _undistort_points(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homog = np.c_[pts, np.ones(len(pts), dtype=np.float64)]
        rays = (self.K_inv @ homog.T).T
        return rays[:, :2] / np.maximum(rays[:, 2:3], 1e-12)

    def _virtual_pixel_points(self, pts):
        norm = self._undistort_points(pts)
        return (norm * self.focal + np.array([[self.width * 0.5, self.height * 0.5]])).astype(np.float32)

    def _gyro_predict(self, pts, R_cur_prev):
        R_cur_prev = np.asarray(R_cur_prev, dtype=np.float64)
        homog = np.c_[pts, np.ones(len(pts), dtype=np.float64)]
        rays = (self.K_inv @ homog.T).T
        rays = (R_cur_prev @ rays.T).T
        proj = (self.K @ rays.T).T
        return (proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)).astype(np.float32)

    def _age_color(self, age):
        t = min(1.0, age / 8.0)
        return (int(255 * (1 - t)), int(220 * t), 40)

    def _clear_tracks(self):
        self.cur_pts = np.empty((0, 2), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)
        self.track_cnt = np.empty((0,), dtype=np.int32)
        self._last_draw_prev_pts = np.empty((0, 2), dtype=np.float32)


def overlay_to_uimg_tensor(image: np.ndarray, dtype=torch.float32):
    return torch.from_numpy(image.copy()).to(dtype=dtype) / 255.0
