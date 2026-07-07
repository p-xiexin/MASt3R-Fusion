from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class ImuRecord:
    t0: float
    t1: float
    gyro: np.ndarray
    accel: np.ndarray


class ImuBuffer:
    def __init__(self, path: str, dt: float = 0.0, gyro_unit: str = "deg"):
        data = np.loadtxt(path)
        if data.shape[1] != 7:
            raise ValueError(f"Expected imu rows as t gx gy gz ax ay az, got {data.shape[1]} columns.")
        if data[0, 0] > 1e12:
            data[:, 0] /= 1e9
        self.time = data[:, 0].astype(np.float64) + float(dt)
        gyro = data[:, 1:4].astype(np.float64)
        if gyro_unit == "deg":
            gyro *= math.pi / 180.0
        self.gyro = gyro
        self.accel = data[:, 4:7].astype(np.float64)

    def records(self, t0: float, t1: float) -> List[ImuRecord]:
        out = []
        start = max(0, bisect.bisect_right(self.time, float(t0)) - 1)
        stop = min(len(self.time) - 1, bisect.bisect_left(self.time, float(t1)) + 1)
        for idx in range(start, stop):
            seg_t0 = max(float(t0), float(self.time[idx]))
            seg_t1 = min(float(t1), float(self.time[idx + 1]))
            if seg_t1 > seg_t0:
                out.append(ImuRecord(seg_t0, seg_t1, self.gyro[idx].copy(), self.accel[idx].copy()))
        return out

    def delta_rotation(self, t0: float, t1: float) -> np.ndarray:
        R = np.eye(3, dtype=np.float64)
        for rec in self.records(t0, t1):
            omega_dt = rec.gyro * (rec.t1 - rec.t0)
            dR, _ = cv2.Rodrigues(omega_dt.reshape(3, 1))
            R = R @ dR
        return R
