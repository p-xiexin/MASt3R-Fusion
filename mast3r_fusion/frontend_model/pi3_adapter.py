import torch

from mast3r_fusion.frontend_model.base import (
    FeatureSpec,
    FeedForwardFrontend,
    PairMatchResult,
    WindowInferenceResult,
)
from mast3r_fusion.pi3x_utils import (
    encode_frame_image,
    load_pi3x,
    pi3x_decode_symmetric_batch,
    pi3x_inference_mono,
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
        # Store RGB pixels as the per-frame feature so batch loop matching can
        # reconstruct pair images even though PI3X does not expose reusable
        # MASt3R-style encoded tokens.
        return FeatureSpec(feat_dim=3, patch_size=1)

    def encode_frame(self, frame):
        return encode_frame_image(frame)

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
        pointmaps = []
        confidences = []
        for frame in frames:
            X, C = self.infer_single(frame)
            pointmaps.append(X)
            confidences.append(C)
        return WindowInferenceResult(
            frames=frames,
            pointmaps=torch.stack(pointmaps),
            confidences=torch.stack(confidences),
            metadata={"source": "pi3x-pair-self-inference"},
        )

    def build_pair_constraints_from_window(self, frames, edges, **kwargs):
        # PI3X is currently used through pair inference. No separate window-level
        # constraint graph is exposed here.
        raise NotImplementedError(
            "PI3Adapter does not expose native window-level pair constraints yet."
        )

    def match_symmetric_batch(
        self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor=1
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
        )

    def decode_symmetric_batch(self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
        return pi3x_decode_symmetric_batch(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )
