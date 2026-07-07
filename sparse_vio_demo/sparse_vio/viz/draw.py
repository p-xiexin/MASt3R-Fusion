from __future__ import annotations

import pathlib

import cv2
import numpy as np

from ..types import BackendState, FrontendResult


class DebugVisualizer:
    def __init__(self, out_dir: str, write_video: bool = True, fps: float = 10.0):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.write_video = write_video
        self.fps = fps
        self.writer = None

    def draw(self, result: FrontendResult, state: BackendState):
        image = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
        for track in result.tracks.values():
            color = self._age_color(track.age)
            x, y = np.rint(track.xy).astype(int)
            cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
            if len(track.observations) >= 2:
                p0 = np.rint(track.observations[-2].xy).astype(int)
                p1 = np.rint(track.observations[-1].xy).astype(int)
                cv2.line(image, tuple(p0), tuple(p1), color, 1, cv2.LINE_AA)
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
        cv2.rectangle(image, (8, 8), (720, 40), (0, 0, 0), -1)
        cv2.putText(image, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (8, 42), (880, 72), (0, 0, 0), -1)
        cv2.putText(image, text2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        if self.write_video:
            self._write(image)
        if result.new_keyframe:
            cv2.imwrite(str(self.out_dir / f"frame_{result.frame_id:06d}.jpg"), image)
        return image

    def save_trajectory(self, state: BackendState):
        canvas = np.full((800, 800, 3), 255, dtype=np.uint8)
        if len(state.poses) < 2:
            cv2.imwrite(str(self.out_dir / "trajectory.jpg"), canvas)
            return
        pts = np.array([[T[0, 3], T[2, 3]] for _, T in sorted(state.poses.items())], dtype=np.float64)
        pts -= pts.mean(axis=0)
        scale = 0.8 * min(canvas.shape[:2]) / max(1e-6, np.ptp(pts, axis=0).max())
        uv = pts * scale + np.array([400.0, 400.0])
        uv = np.rint(uv).astype(int)
        for a, b in zip(uv[:-1], uv[1:]):
            cv2.line(canvas, tuple(a), tuple(b), (30, 80, 220), 2, cv2.LINE_AA)
        cv2.imwrite(str(self.out_dir / "trajectory.jpg"), canvas)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _write(self, image):
        if self.writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.out_dir / "debug.mp4"), fourcc, self.fps, (w, h))
        self.writer.write(image)

    def _age_color(self, age):
        t = min(1.0, age / 20.0)
        return (int(255 * (1 - t)), int(220 * t), 40)

    @staticmethod
    def age_color(age):
        t = min(1.0, age / 20.0)
        return (int(255 * (1 - t)), int(220 * t), 40)
