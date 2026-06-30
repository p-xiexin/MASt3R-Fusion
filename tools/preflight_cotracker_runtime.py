import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mast3r_fusion.config import config, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_kitti360_cotracker.yaml")
    parser.add_argument("--dataset-root", default="data/KITTI-360")
    parser.add_argument("--seq", default="0000")
    parser.add_argument("--pi3x-weights", default=None)
    parser.add_argument("--cotracker-checkpoint", default=None)
    parser.add_argument("--require-cotracker-checkpoint", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    frontend_name = config.get("frontend_model", {}).get("name")
    if frontend_name != "cotracker":
        raise SystemExit(f"Expected frontend_model.name=cotracker, got {frontend_name!r}")

    pi3x_weights = args.pi3x_weights or config.get("frontend_model", {}).get(
        "weights",
        "checkpoints/pi3x/model.safetensors",
    )
    _require_file(pi3x_weights, "PI3X checkpoint")

    cotracker_checkpoint = args.cotracker_checkpoint or config.get("cotracker", {}).get("checkpoint")
    if cotracker_checkpoint:
        _require_file(cotracker_checkpoint, "CoTracker checkpoint")
    elif args.require_cotracker_checkpoint:
        raise SystemExit(
            "CoTracker checkpoint is required for this run. Set --cotracker-checkpoint "
            "or COTRACKER_CHECKPOINT, or rerun with torchhub fallback explicitly allowed."
        )
    else:
        print("[preflight] CoTracker checkpoint not set; runtime will use torch.hub fallback.")

    seq_root = Path(args.dataset_root) / f"2013_05_28_drive_{args.seq}_sync"
    _require_dir(seq_root / "image_00" / "data_rgb", "KITTI360 RGB directory")
    _require_file(seq_root / "imu.txt", "KITTI360 IMU file")
    _require_file(seq_root / "camstamp.txt", "KITTI360 camstamp file")

    _require_import("torch")
    _require_import("gtsam")
    _require_import("mast3r_fusion_backends", import_torch_first=True)
    _require_import("pi3.models.pi3x")
    _require_import("cotracker.predictor")

    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"[preflight] torch={torch.__version__} cuda_available={cuda_ok}")
    if args.require_cuda and not cuda_ok:
        raise SystemExit("CUDA is required for this runtime path but is not available.")

    print("[preflight] cotracker runtime preflight ok")


def _require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    print(f"[preflight] {label}: {path}")


def _require_dir(path, label):
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")
    print(f"[preflight] {label}: {path}")


def _require_import(module_name, import_torch_first=False):
    if import_torch_first:
        importlib.import_module("torch")
    importlib.import_module(module_name)
    print(f"[preflight] import ok: {module_name}")


if __name__ == "__main__":
    main()
