from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from mast3r_fusion.config import config


@dataclass
class LifecycleEvent:
    trigger_pi3x: bool = False
    edge_indices: List[Tuple[int, int]] = field(default_factory=list)
    reason: str = ""


class CoTrackerLifecycleManager:
    """VINS-style feature lifecycle gate for PI3X reconstruction refreshes."""

    def __init__(self, enabled: bool = False):
        cfg = config.get("lifecycle", {})
        cotracker_cfg = config.get("cotracker", {})
        self.enabled = bool(enabled and cfg.get("enabled", True))
        self.window_size = int(cfg.get("window_size", config.get("pi3x", {}).get("window_size", 4)))
        self.overlap_size = int(cfg.get("overlap_size", cfg.get("overlap", self.window_size // 2)))
        if self.overlap_size < 0 or self.overlap_size >= self.window_size:
            raise ValueError(
                "lifecycle.overlap_size must satisfy 0 <= overlap_size < window_size, "
                f"got overlap_size={self.overlap_size}, window_size={self.window_size}"
            )
        self.visibility_threshold = float(cfg.get("oldest_visibility_threshold", 0.25))
        self.min_trigger_gap = int(cfg.get("min_trigger_gap", self.window_size))
        self.grid_size = int(cotracker_cfg.get("grid_size", 32))
        self._keyframes = []
        self._keyframe_indices = []
        self._last_trigger_frame_id: Optional[int] = None

    def observe_keyframe(self, frame, tracker_model=None, keyframe_idx=None) -> LifecycleEvent:
        if not self.enabled:
            return LifecycleEvent()
        self._keyframes.append(frame)
        self._keyframe_indices.append(keyframe_idx)
        if len(self._keyframes) > self.window_size:
            self._keyframes = self._keyframes[-self.window_size :]
            self._keyframe_indices = self._keyframe_indices[-self.window_size :]
        if len(self._keyframes) < self.window_size:
            return LifecycleEvent()

        newest_id = self._keyframes[-1].frame_id
        if (
            self._last_trigger_frame_id is not None
            and newest_id - self._last_trigger_frame_id < self.min_trigger_gap
        ):
            print(
                "[lifecycle] skip visibility check "
                f"newest_frame={newest_id} last_trigger={self._last_trigger_frame_id} "
                f"min_gap={self.min_trigger_gap}"
            )
            return LifecycleEvent()

        oldest_visibility = self._oldest_visibility(tracker_model)
        print(
            "[lifecycle] visibility "
            f"newest_frame={newest_id} oldest={oldest_visibility} "
            f"threshold={self.visibility_threshold} window={len(self._keyframes)} "
            f"overlap={self.overlap_size}"
        )
        if oldest_visibility is None or oldest_visibility > self.visibility_threshold:
            return LifecycleEvent()

        self._last_trigger_frame_id = newest_id
        edges = [(i, len(self._keyframes) - 1) for i in range(len(self._keyframes) - 1)]
        return LifecycleEvent(
            trigger_pi3x=True,
            edge_indices=edges,
            reason=f"oldest_visibility={oldest_visibility:.3f}",
        )

    def mark_submitted(self):
        if self.overlap_size == 0:
            self._keyframes = []
            self._keyframe_indices = []
            return
        self._keyframes = self._keyframes[-self.overlap_size :]
        self._keyframe_indices = self._keyframe_indices[-self.overlap_size :]

    def active_window(self) -> Sequence:
        return tuple(self._keyframes)

    def active_window_from_keyframes(self, keyframes) -> Sequence:
        if any(idx is None for idx in self._keyframe_indices):
            return self.active_window()
        return tuple(keyframes[idx] for idx in self._keyframe_indices)

    def active_window_start(self, keyframes) -> int:
        if not self._keyframe_indices or self._keyframe_indices[0] is None:
            return keyframes.rollup_sum.value + len(keyframes) - len(self._keyframes)
        return int(self._keyframe_indices[0])

    @torch.inference_mode()
    def _oldest_visibility(self, tracker_model) -> Optional[float]:
        if tracker_model is None or len(self._keyframes) < 2:
            return 0.0
        try:
            video = torch.stack([self._cotracker_image(frame) for frame in self._keyframes], dim=0)
            video = video.unsqueeze(0)
            _, visibility = tracker_model(video, grid_size=self.grid_size, grid_query_frame=0)
        except Exception as exc:
            print(f"[lifecycle] CoTracker visibility check failed: {exc}")
            return None

        visibility = visibility[0]
        if visibility.ndim != 2:
            return None
        if visibility.shape[0] == len(self._keyframes):
            oldest_visible = visibility[-1]
        else:
            oldest_visible = visibility[:, -1]
        return float(oldest_visible.float().mean().item())

    @staticmethod
    def _cotracker_image(frame):
        image = frame.uimg.to(device=frame.img.device, dtype=torch.float32)
        return (image.permute(2, 0, 1).clamp(0, 1) * 255.0).contiguous()


def maybe_apply_pi3x_lifecycle_event(
    model,
    factor_graph,
    keyframes,
    window,
    event: LifecycleEvent,
    global_start=None,
):
    if not event.trigger_pi3x:
        return False
    window = list(window)
    if len(window) < 2:
        return False

    print(f"[lifecycle] PI3X window refresh: {event.reason}")
    result = model.infer_window(window)
    if result.pointmaps is None or result.confidences is None:
        return False

    if global_start is None:
        global_start = keyframes.rollup_sum.value + len(keyframes) - len(window)
    local_start = global_start - keyframes.rollup_sum.value
    for local_idx, frame in enumerate(window):
        old_c_mean = float(frame.C.mean().item()) if frame.C is not None else float("nan")
        old_updates = int(getattr(frame, "N_updates", 0))
        frame.X_canon = result.pointmaps[local_idx : local_idx + 1].clone()
        frame.C = result.confidences[local_idx : local_idx + 1].clone()
        frame.N = 1
        frame.N_updates = old_updates + 1
        keyframes[local_start + local_idx] = frame
        print(
            "[lifecycle] wrote PI3X pointmap "
            f"global_kf={global_start + local_idx} frame_id={frame.frame_id} "
            f"old_C_mean={old_c_mean:.4f} new_C_mean={float(frame.C.mean().item()):.4f} "
            f"points={frame.X_canon.shape[1]}"
        )

    ii = [global_start + src for src, _ in event.edge_indices]
    jj = [global_start + dst for _, dst in event.edge_indices]
    pi3x_relative_poses = _relative_poses_from_window_result(result, event.edge_indices)
    factor_graph.add_sparse_pose_edges(zip(ii, jj), relative_poses=pi3x_relative_poses)
    factor_graph.add_factors(ii, jj, config["local_opt"]["min_match_frac"])
    return True


def _relative_poses_from_window_result(result, edge_indices):
    metadata = result.metadata or {}
    camera_poses = metadata.get("camera_poses")
    if camera_poses is None:
        return None
    camera_poses = torch.as_tensor(camera_poses).detach().cpu().numpy()
    if camera_poses.ndim == 4:
        camera_poses = camera_poses[0]
    if camera_poses.ndim != 3 or camera_poses.shape[-2:] != (4, 4):
        raise ValueError(f"PI3X camera poses must have shape (N, 4, 4), got {camera_poses.shape}")
    relative_poses = []
    for src, dst in edge_indices:
        if src >= camera_poses.shape[0] or dst >= camera_poses.shape[0]:
            raise ValueError(
                "Lifecycle edge index exceeds PI3X camera pose count: "
                f"edge=({src}, {dst}), poses={camera_poses.shape[0]}"
            )
        T_src = camera_poses[src]
        T_dst = camera_poses[dst]
        T_src_dst = np.linalg.inv(T_src) @ T_dst
        relative_poses.append(T_src_dst)
    return relative_poses
