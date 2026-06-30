from __future__ import annotations

import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from .types import BackendState


def save_tum_trajectory(path: str, state: BackendState, timestamps: dict[int, float], frame: str = "body"):
    rows = []
    poses = state.body_poses if frame == "body" and state.body_poses else state.poses
    for frame_id, T in sorted(poses.items()):
        if frame_id not in timestamps:
            continue
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        rows.append([timestamps[frame_id], T[0, 3], T[1, 3], T[2, 3], quat[0], quat[1], quat[2], quat[3]])
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.asarray(rows, dtype=np.float64), fmt="%.9f")
