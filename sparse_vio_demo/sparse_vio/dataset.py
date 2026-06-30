from __future__ import annotations

import pathlib
import re
from typing import Iterator, Optional

import cv2
import numpy as np
import yaml

from .types import Camera, FramePacket


def _sorted_images(path: pathlib.Path):
    files = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
    return sorted(files, key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])


def load_camera(calib_path: str) -> Camera:
    with open(calib_path, "r") as f:
        cfg = yaml.safe_load(f)
    calib = np.asarray(cfg["calibration"], dtype=np.float64)
    fx, fy, cx, cy = calib[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = calib[4:] if calib.shape[0] > 4 else None
    T_body_camera = np.asarray(cfg.get("Tic", np.eye(4)), dtype=np.float64)
    return Camera(
        K=K,
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        distortion=D,
        T_body_camera=T_body_camera,
    )


class ImageDataset:
    def __init__(
        self,
        image_dir: str,
        stamp_path: Optional[str] = None,
        start: int = 0,
        end: int = -1,
        subsample: int = 1,
        resize: Optional[tuple[int, int]] = None,
    ):
        self.image_dir = pathlib.Path(image_dir)
        files = _sorted_images(self.image_dir)
        if end < 0:
            end = len(files)
        self.files = files[start:end:subsample]
        if stamp_path:
            raw = np.genfromtxt(stamp_path, dtype=str)
            stamps = raw[:, 0].astype(np.float64) if raw.ndim > 1 else raw.astype(np.float64)
            self.timestamps = stamps[start:end:subsample]
        else:
            self.timestamps = np.arange(len(self.files), dtype=np.float64) / 30.0
        self.resize = resize

    def __len__(self):
        return len(self.files)

    def __iter__(self) -> Iterator[FramePacket]:
        for idx, path in enumerate(self.files):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                image = cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)
            yield FramePacket(idx=idx, timestamp=float(self.timestamps[idx]), image=image)
