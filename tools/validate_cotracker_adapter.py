import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mast3r_fusion.frontend_model.cotracker_adapter import CoTrackerPI3XAdapter


class _FakeCoTracker:
    def __call__(self, video, grid_size=0, grid_query_frame=0):
        assert video.shape == (1, 2, 3, 4, 5)
        assert grid_query_frame == 0
        tracks = torch.tensor(
            [[[[1.0, 1.0], [3.0, 2.0], [4.0, 3.0]], [[2.0, 1.0], [1.0, 2.0], [0.0, 0.0]]]]
        )
        visibility = torch.tensor([[[True, True, True], [True, False, True]]])
        return tracks, visibility


def main():
    adapter = CoTrackerPI3XAdapter.__new__(CoTrackerPI3XAdapter)
    adapter.cotracker = _FakeCoTracker()
    adapter.grid_size = 2
    adapter.max_points = 1200
    adapter.device = "cpu"

    keyframe = _frame()
    frame = _frame()
    idx_i2j, valid = adapter._track_keyframe_to_frame(keyframe, frame)

    assert idx_i2j.shape == (1, 20)
    assert valid.shape == (1, 20, 1)
    assert valid.sum().item() == 2
    assert idx_i2j[0, 6].item() == 7
    assert idx_i2j[0, 19].item() == 0

    tracks = torch.zeros(1, 3, 4, 2)
    visibility = torch.zeros(1, 3, 4, dtype=torch.bool)
    try:
        adapter._extract_two_frame_tracks(tracks, visibility)
    except ValueError:
        pass
    else:
        raise AssertionError("bad CoTracker frame count was accepted")

    try:
        adapter._load_cotracker(checkpoint="/tmp/definitely_missing_cotracker.pth", device="cpu")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing CoTracker checkpoint was accepted")

    print("cotracker adapter validation ok")


def _frame():
    return SimpleNamespace(
        uimg=torch.zeros(4, 5, 3),
        img=torch.zeros(3, 4, 5),
    )


if __name__ == "__main__":
    main()
