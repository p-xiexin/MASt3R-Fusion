import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """Shared-memory feature layout required by Frame/SharedKeyframes.

    `feat_dim` is the final dimension stored in `frame.feat`; `patch_size`
    converts image size to the number of feature tokens. MASt3R uses patch
    tokens, while PI3X currently stores RGB pixels as features with
    `patch_size=1`.
    """

    feat_dim: int = 1024
    patch_size: int = 16

    def num_patches(self, h: int, w: int) -> int:
        return h * w // (self.patch_size * self.patch_size)


@dataclasses.dataclass
class PairMatchResult:
    """Pairwise frontend outputs consumed by tracker and factor graph.

    Required for tracking:
    - `idx_i2j`: dense linear indices into frame i, indexed by frame j pixels,
      shape `(B, H*W)` or `(B, H*W*subpixel_factor^2)` for subpixel matching.
    - `valid_match_j`: frame-j validity mask for `idx_i2j`, shape `(B, H*W, 1)`.
    - `Xii`, `Cii`, `Qii`: frame-i point map, confidence, and match confidence
      flattened to `(B, H*W, 3)`, `(B, H*W, 1)`, `(B, H*W, 1)`.
    - `Xji`, `Cji`, `Qji`: frame-j or cross-view point map in frame-i matching
      convention with the same flattened shapes.

    Required for symmetric batch factors:
    - `idx_j2i`, `valid_match_i`, `Qjj`, `Qij`.
    """

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
    """Optional multiview frontend output.

    The current backend is still pairwise. Multiview models can either expose
    `pair_constraints` directly, keyed by `(i, j)`, or fall back to pairwise
    adapter methods. `metadata` can record model-specific debug details.
    """

    frames: Sequence[Any]
    pointmaps: Optional[torch.Tensor] = None
    confidences: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    descriptor_confidences: Optional[torch.Tensor] = None
    pair_constraints: Optional[Dict[Tuple[int, int], PairMatchResult]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedForwardFrontend:
    """Abstract frontend model contract.

    Runtime code should depend on this interface instead of calling MASt3R or
    PI3X utility functions directly. Model-specific private APIs belong inside
    each adapter implementation.
    """

    name = "base"

    def share_memory(self):
        """Move model parameters to shared memory when the backend supports it."""
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "share_memory"):
            return model.share_memory()
        return self

    def get_feature_spec(self) -> FeatureSpec:
        """Return the feature tensor layout stored in shared frame buffers."""
        return FeatureSpec()

    def encode_frame(self, frame):
        """Populate and return `frame.feat` and `frame.pos`.

        Implementations must write tensors compatible with `get_feature_spec()`.
        This method is responsible for caching encoded features on the frame.
        """
        raise NotImplementedError

    def infer_single(self, frame):
        """Infer a single-frame point map.

        Returns:
            `(X, C)` where `X` has shape `(B, H*W, 3)` and `C` has shape
            `(B, H*W, 1)`. The caller writes these into `frame.update_pointmap()`.
        """
        raise NotImplementedError

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        """Infer pairwise correspondences and point maps.

        Args:
            frame_i: current/query frame.
            frame_j: reference/keyframe.
            init: optional previous dense correspondence indices for iterative
                projection refinement.
            symmetric: when true, return the fields needed by symmetric factor
                construction as well.

        Returns:
            `PairMatchResult`.
        """
        raise NotImplementedError

    def match_symmetric_batch(
        self, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor=1
    ):
        """Infer batched symmetric matches for factor graph construction.

        Returns the tuple consumed by `FactorGraph.add_factors()`:
        `idx_i2j`, `idx_j2i`, `valid_match_j`, `valid_match_i`, `Qii`, `Qjj`,
        `Qji`, and `Qij`.
        """
        raise NotImplementedError

    def infer_window(self, frames: Sequence[Any]) -> WindowInferenceResult:
        """Optional multiframe inference hook.

        Pair-only models may implement this by running `infer_single()` for each
        frame. True multiview models can return window point maps and optional
        pair constraints.
        """
        raise NotImplementedError

    def build_pair_constraints_from_window(
        self, frames: Sequence[Any], edges: Iterable[Tuple[int, int]], **kwargs
    ) -> Dict[Tuple[int, int], PairMatchResult]:
        """Convert a window result into pair constraints for selected edges."""
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
