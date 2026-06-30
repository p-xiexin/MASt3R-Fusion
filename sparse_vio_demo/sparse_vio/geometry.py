from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def compose(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    return T_ab @ T_bc


def pose_to_vec(T_wc: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(T_wc[:3, :3]).as_rotvec(), T_wc[:3, 3]]


def vec_to_pose(x: np.ndarray) -> np.ndarray:
    return se3_from_rt(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])


def project(K: np.ndarray, T_wc: np.ndarray, Pw: np.ndarray) -> np.ndarray:
    T_cw = invert_pose(T_wc)
    Pc = (T_cw[:3, :3] @ Pw.T + T_cw[:3, 3:4]).T
    z = np.maximum(Pc[:, 2:3], 1e-8)
    uv = Pc[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def triangulate_pair(K: np.ndarray, T_w0: np.ndarray, T_w1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray):
    P0 = K @ invert_pose(T_w0)[:3]
    P1 = K @ invert_pose(T_w1)[:3]
    X_h = cv2.triangulatePoints(P0, P1, pts0.T.astype(np.float64), pts1.T.astype(np.float64)).T
    good = np.abs(X_h[:, 3]) > 1e-12
    X = X_h[:, :3] / X_h[:, 3:4]
    z0 = (invert_pose(T_w0)[:3, :3] @ X.T + invert_pose(T_w0)[:3, 3:4]).T[:, 2]
    z1 = (invert_pose(T_w1)[:3, :3] @ X.T + invert_pose(T_w1)[:3, 3:4]).T[:, 2]
    good &= np.isfinite(X).all(axis=1) & (z0 > 0.1) & (z1 > 0.1)
    return X, good

