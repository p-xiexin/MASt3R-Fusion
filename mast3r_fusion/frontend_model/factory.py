from mast3r_fusion.config import config
from mast3r_fusion.frontend_model.cotracker_adapter import CoTrackerPI3XAdapter
from mast3r_fusion.frontend_model.mast3r_adapter import MASt3RAdapter
from mast3r_fusion.frontend_model.pi3_adapter import PI3Adapter


def load_frontend_model(name=None, path=None, device="cuda", **kwargs):
    frontend_cfg = config.get("frontend_model", {})
    name = (name or frontend_cfg.get("name", "mast3r")).lower()
    path = path or frontend_cfg.get("weights")

    if name == "mast3r":
        return MASt3RAdapter.load(path=path, device=device, **kwargs)
    if name in ("pi3", "pi3x"):
        return PI3Adapter.load(path=path, device=device, **kwargs)
    if name == "cotracker":
        return CoTrackerPI3XAdapter.load(path=path, device=device, **kwargs)

    raise ValueError(f"Unsupported frontend model: {name}")
