from __future__ import annotations

import pathlib
import sys
from typing import Optional

import cv2
import numpy as np
import torch

from ..types import Camera, FramePacket, FrontendResult, Observation, Track
from .vins_frontend import VinsFrontend, VinsFrontendConfig


class XFeatFrontend(VinsFrontend):
    def __init__(
        self,
        camera: Camera,
        cfg: Optional[VinsFrontendConfig] = None,
        top_k: int = 1200,
        min_cossim: float = 0.82,
        refill_ratio: float = 0.95,
        geo_ransac_thresh: float = 1.0,
        min_geo_inliers: int = 60,
        strong_geo_inliers: int = 90,
        min_geo_ratio: float = 0.40,
    ):
        super().__init__(camera, cfg)
        self.top_k = top_k
        self.min_cossim = min_cossim
        self.refill_ratio = refill_ratio
        self.geo_ransac_thresh = geo_ransac_thresh
        self.min_geo_inliers = min_geo_inliers
        self.strong_geo_inliers = strong_geo_inliers
        self.min_geo_ratio = min_geo_ratio
        self.xfeat = self._load_xfeat()
        self.prev_pts = None
        self.prev_desc = None
        self.prev_track_ids = None
        self.last_keyframe_tracks = {}

    def process(self, packet: FramePacket, gyro_R: Optional[np.ndarray] = None) -> FrontendResult:
        gray = self._gray(packet.image)
        self.debug = {}
        features = self._detect_xfeat(gray)
        cur_pts, cur_desc = features
        cur_track_ids = np.full(len(cur_pts), -1, dtype=np.int64)

        if self.prev_pts is None:
            self._spawn_xfeat_tracks(features, cur_track_ids, gray, packet.idx)
            self._set_previous_features(cur_pts, cur_desc, cur_track_ids)
            self.prev_gray = gray
            self.prev_frame_id = packet.idx
            result = self._result(packet, True)
            self._remember_keyframe_tracks()
            return result

        matched_cur = self._extend_tracks_by_matching(cur_pts, cur_desc, cur_track_ids, packet.idx)
        refill_threshold = int(self.refill_ratio * self.cfg.max_features)
        if len(self.tracks) < refill_threshold:
            self._spawn_xfeat_tracks(features, cur_track_ids, gray, packet.idx, used_indices=matched_cur)
        else:
            self.debug["xfeat_spawned"] = 0.0

        new_keyframe = self._is_keyframe(packet.idx)
        if new_keyframe:
            self.last_keyframe_id = packet.idx
            self._remember_keyframe_tracks()
        self._set_previous_features(cur_pts, cur_desc, cur_track_ids)
        self.prev_gray = gray
        self.prev_frame_id = packet.idx
        return self._result(packet, new_keyframe)

    def _load_xfeat(self):
        root = pathlib.Path(__file__).resolve().parents[3] / "thirdparty" / "xfeat"
        weights = root / "weights" / "xfeat.pt"
        if not root.exists():
            raise FileNotFoundError(f"XFeat dependency missing: {root}")
        if not weights.exists():
            raise FileNotFoundError(f"XFeat weights missing: {weights}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from modules.xfeat import XFeat

        return XFeat(weights=str(weights), top_k=self.top_k)

    def _detect_xfeat(self, gray):
        image = torch.from_numpy(gray).float()[None, None] / 255.0
        with torch.inference_mode():
            out = self.xfeat.detectAndCompute(image, top_k=self.top_k)[0]
        pts = out["keypoints"].detach().cpu().numpy().astype(np.float32)
        desc = out["descriptors"].detach().cpu()
        self.debug["xfeat_detected"] = float(len(pts))
        return pts, desc

    def _extend_tracks_by_matching(self, cur_pts, cur_desc, cur_track_ids, frame_id):
        old_tracks = self.tracks
        self.tracks = {}
        if self.prev_desc is None or len(self.prev_desc) == 0 or len(cur_desc) == 0:
            self.debug["xfeat_raw_matches"] = 0.0
            self.debug["xfeat_geo_inliers"] = 0.0
            return set()

        prev_desc = self.prev_desc.to(self.xfeat.dev)
        cur_desc_dev = cur_desc.to(self.xfeat.dev)
        idx0, idx1 = self.xfeat.match(prev_desc, cur_desc_dev, min_cossim=self.min_cossim)
        if idx0.numel() == 0:
            self.debug["xfeat_raw_matches"] = 0.0
            self.debug["xfeat_geo_inliers"] = 0.0
            return set()

        prev_idx = idx0.detach().cpu().numpy().astype(np.int64)
        cur_idx = idx1.detach().cpu().numpy().astype(np.int64)
        track_ids = self.prev_track_ids[prev_idx]
        valid = track_ids >= 0
        prev_idx = prev_idx[valid]
        cur_idx = cur_idx[valid]
        track_ids = track_ids[valid]

        prev_m = self.prev_pts[prev_idx]
        cur_m = cur_pts[cur_idx]
        inside = self._inside(prev_m) & self._inside(cur_m)
        prev_idx = prev_idx[inside]
        cur_idx = cur_idx[inside]
        track_ids = track_ids[inside]
        prev_m = prev_m[inside]
        cur_m = cur_m[inside]

        self.debug["xfeat_raw_matches"] = float(len(cur_idx))
        if len(cur_idx) < 8:
            self.debug["xfeat_geo_inliers"] = 0.0
            return set()

        geo_mask = self._geometric_inlier_mask(prev_m, cur_m)
        inlier_count = int(geo_mask.sum())
        inlier_ratio = float(inlier_count / max(1, len(geo_mask)))
        self.debug["xfeat_geo_inliers"] = float(inlier_count)
        self.debug["xfeat_geo_ratio"] = inlier_ratio
        if inlier_count < self.min_geo_inliers:
            return set()
        if inlier_count < self.strong_geo_inliers and inlier_ratio < self.min_geo_ratio:
            return set()

        used_cur = set()
        for track_id, j, ok in zip(track_ids, cur_idx, geo_mask):
            if not ok or int(track_id) not in old_tracks or int(j) in used_cur:
                continue
            track = old_tracks[int(track_id)]
            track.add_observation(frame_id, cur_pts[int(j)].astype(np.float32))
            self.tracks[int(track_id)] = track
            cur_track_ids[int(j)] = int(track_id)
            used_cur.add(int(j))
        self.debug["xfeat_tracked"] = float(len(self.tracks))
        return used_cur

    def _geometric_inlier_mask(self, prev_pts, cur_pts):
        _, mask = cv2.findEssentialMat(
            prev_pts,
            cur_pts,
            self.camera.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.geo_ransac_thresh,
        )
        if mask is None:
            _, mask = cv2.findFundamentalMat(prev_pts, cur_pts, cv2.FM_RANSAC, self.geo_ransac_thresh, 0.999)
        if mask is None:
            return np.zeros(len(prev_pts), dtype=bool)
        return mask.reshape(-1).astype(bool)

    def _spawn_xfeat_tracks(self, features, cur_track_ids, gray, frame_id, used_indices=None):
        need = self.cfg.max_features - len(self.tracks)
        if need <= 0:
            self.debug["xfeat_spawned"] = 0.0
            return
        used_indices = used_indices or set()
        self._set_mask(gray)
        pts, _ = features
        spawned = 0
        for i, xy in enumerate(pts):
            if spawned >= need:
                break
            if i in used_indices or cur_track_ids[i] >= 0:
                continue
            if not self._inside(xy[None])[0]:
                continue
            x, y = np.rint(xy).astype(int)
            if self.mask[y, x] == 0:
                continue
            track = Track(self.next_track_id, xy.astype(np.float32), age=1)
            track.observations.append(Observation(frame_id, track.xy.copy()))
            self.tracks[self.next_track_id] = track
            cur_track_ids[i] = self.next_track_id
            cv2.circle(self.mask, (x, y), self.cfg.min_distance, 0, -1)
            self.next_track_id += 1
            spawned += 1
        self.debug["xfeat_spawned"] = float(spawned)

    def _set_previous_features(self, pts, desc, track_ids):
        valid = track_ids >= 0
        self.prev_pts = pts[valid].copy()
        self.prev_desc = desc[valid].detach().cpu()
        self.prev_track_ids = track_ids[valid].copy()

    def _is_keyframe(self, frame_id):
        if frame_id == 0:
            return True
        tracked = int(self.debug.get("xfeat_tracked", 0.0))
        if tracked < self.min_geo_inliers:
            return True
        if frame_id - self.last_keyframe_id < self.cfg.keyframe_gap:
            return False
        parallax = []
        for track_id, track in self.tracks.items():
            ref = self.last_keyframe_tracks.get(track_id)
            if ref is not None:
                parallax.append(np.linalg.norm(track.xy - ref))
        avg_parallax = float(np.mean(parallax)) if parallax else 0.0
        self.debug["kf_parallax"] = avg_parallax
        return avg_parallax >= self.cfg.keyframe_parallax

    def _remember_keyframe_tracks(self):
        self.last_keyframe_tracks = {track_id: track.xy.copy() for track_id, track in self.tracks.items()}

    def _result(self, packet: FramePacket, new_keyframe: bool):
        result = super()._result(packet, new_keyframe)
        result.debug_image = None
        return result
