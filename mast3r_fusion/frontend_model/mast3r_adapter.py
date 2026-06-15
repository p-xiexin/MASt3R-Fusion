import torch

from mast3r.model import AsymmetricMASt3R
from mast3r_fusion.frontend_model.base import (
    FeedForwardFrontend,
    PairMatchResult,
    WindowInferenceResult,
)
from mast3r_fusion.mast3r_utils import (
    mast3r_asymmetric_inference,
    mast3r_decode_symmetric_batch,
    mast3r_inference_mono,
    mast3r_match_asymmetric,
    mast3r_match_symmetric,
)


class MASt3RAdapter(FeedForwardFrontend):
    name = "mast3r"

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @classmethod
    def load(cls, path=None, device="cuda"):
        weights_path = (
            "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            if path is None
            else path
        )
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        return cls(model)

    @torch.inference_mode
    def encode_frame(self, frame):
        if frame.feat is None:
            frame.feat, frame.pos, _ = self.model._encode_image(
                frame.img, frame.img_true_shape
            )
        return frame.feat, frame.pos

    def infer_single(self, frame):
        return mast3r_inference_mono(self.model, frame)

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        if symmetric:
            result = mast3r_match_symmetric(
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

        result = mast3r_match_asymmetric(
            self.model, frame_i, frame_j, idx_i2j_init=init
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
            metadata={"source": "mast3r-single-frame-fallback"},
        )

    def match_symmetric_batch(
        self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor=1, **kwargs
    ):
        return mast3r_match_symmetric(
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
        return mast3r_decode_symmetric_batch(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

    def asymmetric_inference(self, frame_i, frame_j):
        return mast3r_asymmetric_inference(self.model, frame_i, frame_j)
