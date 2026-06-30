from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Camera:
    K: np.ndarray
    width: int
    height: int
    distortion: Optional[np.ndarray] = None
    T_body_camera: Optional[np.ndarray] = None


@dataclass
class FramePacket:
    idx: int
    timestamp: float
    image: np.ndarray


@dataclass
class Observation:
    frame_id: int
    xy: np.ndarray


@dataclass
class Track:
    track_id: int
    xy: np.ndarray
    age: int = 1
    observations: List[Observation] = field(default_factory=list)
    point_id: Optional[int] = None

    def add_observation(self, frame_id: int, xy: np.ndarray):
        self.xy = xy.astype(np.float32)
        self.age += 1
        self.observations.append(Observation(frame_id, self.xy.copy()))


@dataclass
class FrontendResult:
    frame_id: int
    timestamp: float
    image: np.ndarray
    tracks: Dict[int, Track]
    new_keyframe: bool
    debug_image: Optional[np.ndarray] = None
    debug: Dict[str, float] = field(default_factory=dict)


@dataclass
class BackendState:
    poses: Dict[int, np.ndarray] = field(default_factory=dict)  # T_wc
    points: Dict[int, np.ndarray] = field(default_factory=dict)
    track_to_point: Dict[int, int] = field(default_factory=dict)
    body_poses: Dict[int, np.ndarray] = field(default_factory=dict)  # T_wb
    velocities: Dict[int, np.ndarray] = field(default_factory=dict)
