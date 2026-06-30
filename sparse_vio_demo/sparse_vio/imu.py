from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List

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
        cur_t = float(t0)
        while cur_t < t1 - 1e-9:
            idx = bisect.bisect(self.time, cur_t + 1e-3)
            if idx >= len(self.time):
                break
            next_t = min(float(self.time[idx]), float(t1))
            if next_t > cur_t:
                out.append(ImuRecord(cur_t, next_t, self.gyro[idx].copy(), self.accel[idx].copy()))
            cur_t = next_t
        return out
