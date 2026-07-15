import torch

from mast3r_fusion.frontend_model.base import (
    FeatureSpec,
    FeedForwardFrontend,
    PairMatchResult,
    WindowInferenceResult,
)
from mast3r_fusion.frontend_model.pi3x_utils import (
    encode_frame_pi3x,
    load_pi3x,
    pi3x_decode_symmetric_batch,
    pi3x_inference_mono,
    pi3x_inference_window,
    pi3x_match_window_edges,
    pi3x_match_asymmetric,
    pi3x_match_symmetric,
)


class PI3Adapter(FeedForwardFrontend):
    name = "pi3x"

    def __init__(self, model, weights_path=None, device="cuda"):
        self.model = model
        self.weights_path = weights_path
        self.device = device

    def __getattr__(self, name):
        return getattr(self.model, name)

    @classmethod
    def load(cls, path=None, device="cuda", **kwargs):
        model = load_pi3x(path=path, device=device)
        return cls(model=model, weights_path=path, device=device)

    def get_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(feat_dim=1024, patch_size=14)

    def encode_frame(self, frame):
        return encode_frame_pi3x(self.model, frame)

    def infer_single(self, frame):
        return pi3x_inference_mono(self.model, frame)

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        if symmetric:
            self.encode_frame(frame_i)
            self.encode_frame(frame_j)
            result = pi3x_match_symmetric(
                self.model,
                frame_i.feat,
                frame_i.pos,
                frame_j.feat,
                frame_j.pos,
                [frame_i.img_true_shape],
                [frame_j.img_true_shape],
                kwargs.get("subpixel_factor", 1),
                frames_i=[frame_i],
                frames_j=[frame_j],
            )
            return PairMatchResult(*result)

        result = pi3x_match_asymmetric(
            self.model,
            frame_i,
            frame_j,
            idx_i2j_init=init,
            init_relative_pose=kwargs.get("init_relative_pose"),
        )
        return PairMatchResult(
            idx_i2j=result[0],
            valid_match_j=result[1],
            Xii=result[2],
            Cii=result[3],
            Qii=result[4],
            Xji=result[5],
            Cji=result[6],
            Qji=result[7],
        )

    def infer_window(self, frames):
        pointmaps, confidences, poses = pi3x_inference_window(self.model, frames)
        return WindowInferenceResult(
            frames=frames,
            pointmaps=pointmaps,
            confidences=confidences,
            metadata={"source": "pi3x-window-inference", "poses": poses},
        )

    def build_pair_constraints_from_window(self, frames, edges, **kwargs):
        return pi3x_match_window_edges(
            self.model,
            frames,
            list(edges),
            kwargs.get("subpixel_factor", 1),
            preserve_anchor=kwargs.get("preserve_anchor", False),
        )

    def match_symmetric_batch(
        self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor=1, **kwargs
    ):
        return pi3x_match_symmetric(
            self.model,
            feat_i,
            pos_i,
            feat_j,
            pos_j,
            shape_i,
            shape_j,
            subpixel_factor,
            frames_i=kwargs.get("frames_i"),
            frames_j=kwargs.get("frames_j"),
        )

    def decode_symmetric_batch(self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, **kwargs):
        return pi3x_decode_symmetric_batch(
            self.model,
            feat_i,
            pos_i,
            feat_j,
            pos_j,
            shape_i,
            shape_j,
            frames_i=kwargs.get("frames_i"),
            frames_j=kwargs.get("frames_j"),
        )
