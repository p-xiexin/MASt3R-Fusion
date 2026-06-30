from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from ..types import Camera, FramePacket, FrontendResult, Observation, Track


@dataclass
class VinsFrontendConfig:
    max_features: int = 250
    min_distance: int = 30
    quality: float = 0.01
    lk_win: int = 21
    lk_levels: int = 3
    fb_thresh: float = 0.5
    f_ransac_thresh: float = 1.0
    keyframe_parallax: float = 20.0
    keyframe_gap: int = 5
    equalize: bool = True
    use_fast: bool = False
    reject_with_f: bool = False


class VinsFrontend:
    def __init__(self, camera: Camera, cfg: Optional[VinsFrontendConfig] = None):
        self.camera = camera
        self.cfg = cfg or VinsFrontendConfig()
        self.prev_gray = None
        self.prev_frame_id = -1
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.last_keyframe_id = 0
        self.debug = {}

    def process(self, packet: FramePacket, gyro_R: Optional[np.ndarray] = None) -> FrontendResult:
        gray = self._gray(packet.image)
        self.debug = {}
        if self.prev_gray is None:
            self._spawn_tracks(gray, packet.idx)
            self.prev_gray = gray
            self.prev_frame_id = packet.idx
            return self._result(packet, True)

        self._track_by_lk(gray, packet.idx, gyro_R)
        if self.cfg.reject_with_f:
            self._reject_with_f()
        self._spawn_tracks(gray, packet.idx)
        new_keyframe = self._is_keyframe(packet.idx)
        if new_keyframe:
            self.last_keyframe_id = packet.idx
        self.prev_gray = gray
        self.prev_frame_id = packet.idx
        return self._result(packet, new_keyframe)

    def _gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.cfg.equalize:
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        return gray

    def _track_by_lk(self, gray, frame_id, gyro_R):
        if not self.tracks:
            return
        ids = np.array(list(self.tracks.keys()), dtype=np.int64)
        before = len(ids)
        prev_pts = np.array([self.tracks[i].xy for i in ids], dtype=np.float32)
        init = self._gyro_predict(prev_pts, gyro_R) if gyro_R is not None else None
        cur, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_pts.reshape(-1, 1, 2),
            None if init is None else init.reshape(-1, 1, 2).astype(np.float32),
            winSize=(self.cfg.lk_win, self.cfg.lk_win),
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=0 if init is None else cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if cur is None or st is None:
            self.tracks.clear()
            return
        back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            cur,
            None,
            winSize=(self.cfg.lk_win, self.cfg.lk_win),
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        cur = cur.reshape(-1, 2)
        valid = st.reshape(-1).astype(bool) & self._inside(cur)
        if back is not None and st_back is not None:
            fb = np.linalg.norm(back.reshape(-1, 2) - prev_pts, axis=1)
            valid &= st_back.reshape(-1).astype(bool) & (fb <= self.cfg.fb_thresh)
            self.debug["fb_median"] = float(np.median(fb)) if len(fb) else 0.0
        kept = int(valid.sum())
        self.debug["lk_before"] = float(before)
        self.debug["lk_kept"] = float(kept)
        for track_id, xy, ok in zip(ids, cur, valid):
            if ok:
                self.tracks[int(track_id)].add_observation(frame_id, xy)
            else:
                self.tracks.pop(int(track_id), None)

    def _reject_with_f(self):
        if len(self.tracks) < 8:
            return
        ids = np.array(list(self.tracks.keys()), dtype=np.int64)
        prev = []
        cur = []
        for track_id in ids:
            obs = self.tracks[int(track_id)].observations
            if len(obs) < 2:
                prev.append(obs[-1].xy)
            else:
                prev.append(obs[-2].xy)
            cur.append(obs[-1].xy)
        prev = np.asarray(prev, dtype=np.float32)
        cur = np.asarray(cur, dtype=np.float32)
        _, mask = cv2.findFundamentalMat(prev, cur, cv2.FM_RANSAC, self.cfg.f_ransac_thresh, 0.99)
        if mask is None:
            return
        kept = mask.reshape(-1).astype(bool)
        self.debug["f_before"] = float(len(ids))
        self.debug["f_kept"] = float(kept.sum())
        for track_id, ok in zip(ids, mask.reshape(-1).astype(bool)):
            if not ok:
                self.tracks.pop(int(track_id), None)

    def _spawn_tracks(self, gray, frame_id):
        need = self.cfg.max_features - len(self.tracks)
        if need <= 0:
            return
        self._set_mask(gray)
        need = self.cfg.max_features - len(self.tracks)
        if need <= 0:
            return
        pts = self._detect(gray, self.mask, need)
        if pts is None:
            return
        for xy in pts.reshape(-1, 2):
            track = Track(self.next_track_id, xy.astype(np.float32), age=1)
            track.observations.append(Observation(frame_id, track.xy.copy()))
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

    def _set_mask(self, gray):
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        kept = {}
        for track in sorted(self.tracks.values(), key=lambda t: -t.age):
            x, y = np.rint(track.xy).astype(int)
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0] and mask[y, x] == 255:
                kept[track.track_id] = track
                cv2.circle(mask, (x, y), self.cfg.min_distance, 0, -1)
        self.tracks = kept
        self.mask = mask

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

    def _is_keyframe(self, frame_id):
        last_track_num = 0
        long_track_num = 0
        new_feature_num = 0
        parallax = []
        for track in self.tracks.values():
            if len(track.observations) <= 1:
                new_feature_num += 1
                continue
            last_track_num += 1
            if len(track.observations) >= 4:
                long_track_num += 1
            cur = track.observations[-1]
            prev = track.observations[-2]
            if cur.frame_id == frame_id and prev.frame_id == self.prev_frame_id:
                parallax.append(np.linalg.norm(cur.xy - prev.xy))

        self.debug["last_track_num"] = float(last_track_num)
        self.debug["long_track_num"] = float(long_track_num)
        self.debug["new_feature_num"] = float(new_feature_num)
        self.debug["avg_parallax"] = float(np.mean(parallax)) if parallax else 0.0

        if frame_id < 2 or last_track_num < 20 or long_track_num < 40:
            return True
        if new_feature_num > 0.5 * max(1, last_track_num):
            return True
        if frame_id - self.last_keyframe_id >= self.cfg.keyframe_gap:
            return True
        return bool(parallax) and float(np.mean(parallax)) >= self.cfg.keyframe_parallax

    def _gyro_predict(self, pts, R_cur_prev):
        K = self.camera.K
        K_inv = np.linalg.inv(K)
        homog = np.c_[pts, np.ones(len(pts))]
        rays = (K_inv @ homog.T).T
        rays = (R_cur_prev @ rays.T).T
        proj = (K @ rays.T).T
        return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)

    def _inside(self, pts):
        return (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < self.camera.width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < self.camera.height)
        )

    def _result(self, packet, new_keyframe):
        tracks = {}
        for track_id, track in self.tracks.items():
            clone = Track(track.track_id, track.xy.copy(), age=track.age, point_id=track.point_id)
            clone.observations = [Observation(ob.frame_id, ob.xy.copy()) for ob in track.observations]
            tracks[track_id] = clone
        ages = [track.age for track in tracks.values()]
        debug = dict(self.debug)
        debug["age_median"] = float(np.median(ages)) if ages else 0.0
        debug["age_gt10"] = float(sum(age > 10 for age in ages))
        return FrontendResult(
            frame_id=packet.idx,
            timestamp=packet.timestamp,
            image=packet.image,
            tracks=tracks,
            new_keyframe=new_keyframe,
            debug=debug,
        )
