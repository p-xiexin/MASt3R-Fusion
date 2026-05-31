from mast3r_fusion.frontend_model.base import FeedForwardFrontend


class PI3Adapter(FeedForwardFrontend):
    name = "pi3"

    def __init__(self, model=None, weights_path=None, device="cuda"):
        self.model = model
        self.weights_path = weights_path
        self.device = device

    @classmethod
    def load(cls, path=None, device="cuda", **kwargs):
        return cls(weights_path=path, device=device)

    def _missing_backend(self):
        raise NotImplementedError(
            "PI3Adapter is the multiview frontend extension point. "
            "Install PI3 and implement load/infer_window/build_pair_constraints_from_window "
            "to emit the same pairwise constraints consumed by the existing backend."
        )

    def encode_frame(self, frame):
        return self._missing_backend()

    def infer_single(self, frame):
        return self._missing_backend()

    def match_pair(self, frame_i, frame_j, init=None, symmetric=False, **kwargs):
        return self._missing_backend()

    def infer_window(self, frames):
        return self._missing_backend()

    def build_pair_constraints_from_window(self, frames, edges, **kwargs):
        return self._missing_backend()
