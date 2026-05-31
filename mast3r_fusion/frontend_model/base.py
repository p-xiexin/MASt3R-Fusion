import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    feat_dim: int = 1024
    patch_size: int = 16

    def num_patches(self, h: int, w: int) -> int:
        return h * w // (self.patch_size * self.patch_size)


@dataclasses.dataclass
class PairMatchResult:
    idx_i2j: torch.Tensor
    idx_j2i: Optional[torch.Tensor] = None
    valid_match_j: Optional[torch.Tensor] = None
    valid_match_i: Optional[torch.Tensor] = None
    Qii: Optional[torch.Tensor] = None
    Qjj: Optional[torch.Tensor] = None
    Qji: Optional[torch.Tensor] = None
    Qij: Optional[torch.Tensor] = None
    Xii: Optional[torch.Tensor] = None
    Cii: Optional[torch.Tensor] = None
    Xji: Optional[torch.Tensor] = None
    Cji: Optional[torch.Tensor] = None


@dataclasses.dataclass
class WindowInferenceResult:
    frames: Sequence[Any]
    pointmaps: Optional[torch.Tensor] = None
    confidences: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    descriptor_confidences: Optional[torch.Tensor] = None
    pair_constraints: Optional[Dict[Tuple[int, int], PairMatchResult]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedForwardFrontend:
    name = "base"

    def share_memory(self):
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "share_memory"):
            return model.share_memory()
        return self

    def get_feature_spec(self) -> FeatureSpec:
        return FeatureSpec()

    def encode_frame(self, frame):
        raise NotImplementedError

    def infer_single(self, frame):
        raise NotImplementedError

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        raise NotImplementedError

    def infer_window(self, frames: Sequence[Any]) -> WindowInferenceResult:
        raise NotImplementedError

    def build_pair_constraints_from_window(
        self, frames: Sequence[Any], edges: Iterable[Tuple[int, int]], **kwargs
    ) -> Dict[Tuple[int, int], PairMatchResult]:
        window_result = self.infer_window(frames)
        if window_result.pair_constraints is None:
            raise NotImplementedError(
                f"{self.name} does not expose window-level pair constraints"
            )
        return {
            edge: window_result.pair_constraints[edge]
            for edge in edges
            if edge in window_result.pair_constraints
        }
