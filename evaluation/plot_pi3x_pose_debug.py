import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_pose_debug(path, align_pi3x=True):
    prior_by_frame = {}
    pi3x_by_frame = {}
    windows = {}

    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = (row["record_time"], row["window_start"], row["window_end"])
            windows.setdefault(key, []).append(row)

    for rows in windows.values():
        rows = sorted(rows, key=lambda row: int(row["local_idx"]))
        frame_ids = [int(row["frame_id"]) for row in rows]
        prior = np.stack([matrix_from_row(row, "prior") for row in rows], axis=0)
        pi3x = np.stack([matrix_from_row(row, "pi3x") for row in rows], axis=0)
        if align_pi3x and len(prior) > 0:
            pi3x = np.einsum("ij,njk->nik", prior[0] @ np.linalg.inv(pi3x[0]), pi3x)
        for frame_id, prior_T, pi3x_T in zip(frame_ids, prior, pi3x):
            prior_by_frame[frame_id] = prior_T[:3, 3]
            pi3x_by_frame[frame_id] = pi3x_T[:3, 3]

    return dict_to_trajectory(prior_by_frame), dict_to_trajectory(pi3x_by_frame)


def matrix_from_row(row, prefix):
    return np.asarray([float(row[f"{prefix}_{idx}"]) for idx in range(16)], dtype=np.float64).reshape(4, 4)


def dict_to_trajectory(values):
    if not values:
        return np.empty((0, 4), dtype=np.float64)
    rows = [[frame_id, *values[frame_id]] for frame_id in sorted(values)]
    return np.asarray(rows, dtype=np.float64)


def load_slam_result(path, keyframes_only=False):
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 16:
        raise ValueError(f"Result file must have at least 16 columns, got {data.shape[1]}.")

    if keyframes_only and data.shape[1] >= 17:
        data = data[data[:, 16] > 0.5]

    by_frame = {}
    for row in data:
        frame_id = int(row[15])
        by_frame[frame_id] = row[1:4].astype(np.float64)
    return dict_to_trajectory(by_frame)


def set_equal_axes(ax, trajectories):
    points = [traj[:, 1:4] for traj in trajectories if traj.size]
    if not points:
        return
    points = np.concatenate(points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.5, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_traj(ax, traj, label, color, marker=None):
    if traj.size == 0:
        print(f"[WARN] empty trajectory skipped: {label}")
        return
    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color=color, linewidth=1.2, label=label)
    if marker is not None:
        ax.scatter(traj[:, 1], traj[:, 2], traj[:, 3], color=color, s=8, marker=marker)


def main():
    parser = argparse.ArgumentParser(description="Plot PI3X pose-prior debug trajectories.")
    parser.add_argument("--debug-csv", required=True, help="Path to *.pi3x_pose_debug.csv.")
    parser.add_argument("--result", required=True, help="SLAM result.txt path.")
    parser.add_argument("--output", default="pi3x_pose_debug_trajectory.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-align-pi3x", action="store_true", help="Plot raw PI3X output poses without per-window first-pose alignment.")
    parser.add_argument("--slam-keyframes-only", action="store_true", help="Plot only result rows with keyframe flag == 1.")
    args = parser.parse_args()

    prior_traj, pi3x_traj = load_pose_debug(args.debug_csv, align_pi3x=not args.no_align_pi3x)
    slam_traj = load_slam_result(args.result, keyframes_only=args.slam_keyframes_only)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_traj(ax, prior_traj, "PI3X input prior / IMU preintegration", "tab:orange", marker="o")
    plot_traj(ax, pi3x_traj, "PI3X output pose", "tab:green", marker="^")
    plot_traj(ax, slam_traj, "SLAM result", "tab:blue")
    set_equal_axes(ax, [prior_traj, pi3x_traj, slam_traj])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"[INFO] saved {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
