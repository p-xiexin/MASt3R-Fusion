from pathlib import Path

import torch

from mast3r_fusion.config import config
from mast3r_fusion.frontend_model.base import PairMatchResult
from mast3r_fusion.frontend_model.pi3_adapter import PI3Adapter


class CoTrackerPI3XAdapter(PI3Adapter):
    """CoTracker sparse frontend with PI3X dense reconstruction support."""

    name = "cotracker"

    def __init__(self, model, cotracker=None, grid_size=32, max_points=1200, device="cuda"):
        super().__init__(model)
        self.cotracker = cotracker
        self.grid_size = int(grid_size)
        self.max_points = int(max_points)
        self.device = device

    @classmethod
    def load(cls, path=None, device="cuda", **kwargs):
        pi3 = PI3Adapter.load(path=path, device=device)
        cfg = config.get("cotracker", {})
        cotracker = cls._load_cotracker(
            checkpoint=cfg.get("checkpoint"),
            device=device,
        )
        return cls(
            pi3.model,
            cotracker=cotracker,
            grid_size=cfg.get("grid_size", 32),
            max_points=cfg.get("max_points", 1200),
            device=device,
        )

    @staticmethod
    def _load_cotracker(checkpoint=None, device="cuda"):
        try:
            from cotracker.predictor import CoTrackerPredictor
        except ImportError as exc:
            raise ImportError(
                "CoTracker is not installed. Install the cotracker package or use "
                "--frontend-model pi3x/mast3r."
            ) from exc

        if checkpoint:
            if not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"CoTracker checkpoint not found: {checkpoint}. "
                    "Set cotracker.checkpoint or --cotracker-checkpoint to a local pretrained checkpoint, "
                    "or leave it unset to use torch.hub."
                )
            model = CoTrackerPredictor(checkpoint=checkpoint)
        else:
            try:
                model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            except Exception as exc:
                raise RuntimeError(
                    "CoTracker checkpoint was not provided and torch.hub loading failed. "
                    "Set cotracker.checkpoint in the config."
                ) from exc
        return model.to(device).eval()

    def share_memory(self):
        super().share_memory()
        if self.cotracker is not None and hasattr(self.cotracker, "share_memory"):
            self.cotracker.share_memory()
        return self

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        if symmetric:
            return super().match_pair(frame_i, frame_j, init=init, symmetric=True, **kwargs)

        dense = super().match_pair(frame_i, frame_j, init=init, symmetric=False, **kwargs)
        idx_i2j, valid_match_j = self._track_keyframe_to_frame(frame_j, frame_i)
        return PairMatchResult(
            idx_i2j=idx_i2j,
            valid_match_j=valid_match_j,
            Xii=dense.Xii,
            Cii=dense.Cii,
            Qii=dense.Qii,
            Xji=dense.Xji,
            Cji=dense.Cji,
            Qji=dense.Qji,
        )

    @torch.inference_mode()
    def _track_keyframe_to_frame(self, keyframe, frame):
        if self.cotracker is None:
            raise RuntimeError("CoTracker model is not loaded.")

        video = torch.stack(
            (
                self._cotracker_image(keyframe),
                self._cotracker_image(frame),
            ),
            dim=0,
        ).unsqueeze(0)
        pred_tracks, pred_visibility = self.cotracker(
            video,
            grid_size=self.grid_size,
            grid_query_frame=0,
        )
        xy0, xy1, vis = self._extract_two_frame_tracks(pred_tracks, pred_visibility)

        h, w = keyframe.uimg.shape[:2]
        idx_i2j = torch.zeros((1, h * w), device=frame.img.device, dtype=torch.long)
        valid = torch.zeros((1, h * w, 1), device=frame.img.device, dtype=torch.bool)

        xy0 = xy0.to(device=frame.img.device)
        xy1 = xy1.to(device=frame.img.device)
        vis = vis.to(device=frame.img.device, dtype=torch.bool)
        if xy0.shape[0] > self.max_points:
            step = max(1, xy0.shape[0] // self.max_points)
            xy0 = xy0[::step][: self.max_points]
            xy1 = xy1[::step][: self.max_points]
            vis = vis[::step][: self.max_points]

        src_x = torch.round(xy0[:, 0]).long().clamp(0, w - 1)
        src_y = torch.round(xy0[:, 1]).long().clamp(0, h - 1)
        dst_x = torch.round(xy1[:, 0]).long().clamp(0, w - 1)
        dst_y = torch.round(xy1[:, 1]).long().clamp(0, h - 1)
        src_lin = src_y * w + src_x
        dst_lin = dst_y * w + dst_x

        valid_points = vis & torch.isfinite(xy0).all(dim=-1) & torch.isfinite(xy1).all(dim=-1)
        idx_i2j[0, src_lin[valid_points]] = dst_lin[valid_points]
        valid[0, src_lin[valid_points], 0] = True
        return idx_i2j, valid

    @staticmethod
    def _extract_two_frame_tracks(pred_tracks, pred_visibility):
        tracks = pred_tracks[0]
        visibility = pred_visibility[0]
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"CoTracker tracks must have shape (T, N, 2), got {tracks.shape}")
        if visibility.ndim != 2:
            raise ValueError(f"CoTracker visibility must have shape (T, N), got {visibility.shape}")
        if tracks.shape[0] == 2 and visibility.shape[0] == 2:
            return tracks[0], tracks[1], visibility[1]
        if tracks.shape[1] == 2 and visibility.shape[1] == 2:
            return tracks[:, 0], tracks[:, 1], visibility[:, 1]
        raise ValueError(
            "CoTracker two-frame tracking expected T=2 in tracks/visibility, "
            f"got tracks={tracks.shape}, visibility={visibility.shape}"
        )

    def _cotracker_image(self, frame):
        image = frame.uimg.to(device=self.device, dtype=torch.float32)
        return (image.permute(2, 0, 1).clamp(0, 1) * 255.0).contiguous()
