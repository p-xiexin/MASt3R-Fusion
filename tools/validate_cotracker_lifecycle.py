import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mast3r_fusion.config import config
from mast3r_fusion.lifecycle import (
    LifecycleEvent,
    _relative_poses_from_window_result,
    maybe_apply_pi3x_lifecycle_event,
)


class _Value:
    def __init__(self, value):
        self.value = value


class _Frame:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.X_canon = None
        self.C = None

    def update_pointmap(self, X, C):
        self.X_canon = X.clone()
        self.C = C.clone()


class _Keyframes:
    def __init__(self, frames, rollup_sum=0):
        self.frames = list(frames)
        self.rollup_sum = _Value(rollup_sum)
        self.writes = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx - self.rollup_sum.value]

    def __setitem__(self, idx, frame):
        self.writes.append(idx)
        self.frames[idx] = frame


class _Model:
    def infer_window(self, frames):
        n = len(frames)
        pointmaps = torch.arange(n * 4 * 3, dtype=torch.float32).reshape(n, 4, 3)
        confidences = torch.ones(n, 4, 1)
        poses = torch.eye(4).repeat(n, 1, 1)
        for i in range(n):
            poses[i, 0, 3] = float(i)
        return SimpleNamespace(
            frames=frames,
            pointmaps=pointmaps,
            confidences=confidences,
            metadata={"camera_poses": poses},
        )


class _FactorGraph:
    def __init__(self):
        self.sparse_edges = None
        self.relative_poses = None
        self.factor_edges = None

    def add_sparse_pose_edges(self, edges, relative_poses=None):
        self.sparse_edges = list(edges)
        self.relative_poses = list(relative_poses)

    def add_factors(self, ii, jj, min_match_frac):
        self.factor_edges = (list(ii), list(jj), min_match_frac)
        return True


def main():
    config.clear()
    config.update({"local_opt": {"min_match_frac": 0.001}})

    keyframes = _Keyframes([_Frame(i) for i in range(5)], rollup_sum=10)
    window = [keyframes[i] for i in (12, 13, 14)]
    event = LifecycleEvent(
        trigger_pi3x=True,
        edge_indices=[(0, 2), (1, 2)],
        reason="validation",
    )
    graph = _FactorGraph()

    assert maybe_apply_pi3x_lifecycle_event(
        _Model(),
        graph,
        keyframes,
        window,
        event,
        global_start=12,
    )

    assert keyframes.writes == [2, 3, 4]
    assert graph.sparse_edges == [(12, 14), (13, 14)]
    assert graph.factor_edges == ([12, 13], [14, 14], 0.001)
    np.testing.assert_allclose(graph.relative_poses[0], _translation_x(2.0))
    np.testing.assert_allclose(graph.relative_poses[1], _translation_x(1.0))
    _validate_bad_pose_shapes()
    print("cotracker lifecycle validation ok")


def _translation_x(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


def _validate_bad_pose_shapes():
    bad_shape = SimpleNamespace(metadata={"camera_poses": torch.zeros(3, 3)})
    try:
        _relative_poses_from_window_result(bad_shape, [(0, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("bad PI3X camera pose shape was accepted")

    too_few = SimpleNamespace(metadata={"camera_poses": torch.eye(4).repeat(1, 1, 1)})
    try:
        _relative_poses_from_window_result(too_few, [(0, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range lifecycle edge was accepted")


if __name__ == "__main__":
    main()
