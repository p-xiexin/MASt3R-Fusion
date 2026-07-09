import argparse
import csv
import io
from pathlib import Path

import numpy as np


def matrix_from_row(row, prefix):
    values = [float(row[f"{prefix}_{idx}"]) for idx in range(16)]
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def trajectory_from_map(positions_by_frame):
    if not positions_by_frame:
        return np.empty((0, 4), dtype=np.float64)
    rows = [[frame_id, *positions_by_frame[frame_id]] for frame_id in sorted(positions_by_frame)]
    return np.asarray(rows, dtype=np.float64)


def load_pi3x_debug(csv_path):
    windows = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            key = (row["record_time"], row["window_start"], row["window_end"])
            windows.setdefault(key, []).append(row)

    prior_by_frame = {}
    pi3x_by_frame = {}
    window_start_frame_ids = []

    for rows in windows.values():
        rows = sorted(rows, key=lambda row: int(row["local_idx"]))
        frame_ids = [int(row["frame_id"]) for row in rows]
        if frame_ids:
            window_start_frame_ids.append(frame_ids[0])

        prior = np.stack([matrix_from_row(row, "prior") for row in rows], axis=0)
        pi3x = np.stack([matrix_from_row(row, "pi3x") for row in rows], axis=0)
        pi3x_aligned = align_window_to_prior(prior, pi3x)

        for frame_id, prior_T, pi3x_T in zip(frame_ids, prior, pi3x_aligned):
            prior_by_frame[frame_id] = prior_T[:3, 3]
            pi3x_by_frame[frame_id] = pi3x_T[:3, 3]

    return (
        trajectory_from_map(prior_by_frame),
        trajectory_from_map(pi3x_by_frame),
        sorted(set(window_start_frame_ids)),
    )


def align_window_to_prior(prior, pi3x):
    if len(prior) == 0:
        return pi3x
    return np.einsum("ij,njk->nik", prior[0] @ np.linalg.inv(pi3x[0]), pi3x)


def load_h5_keyframes(h5_path):
    import h5py
    import torch

    positions_by_frame = {}
    with h5py.File(h5_path, "r") as h5_file:
        for key in sorted(h5_file.keys(), key=frame_key_sort):
            frame = torch.load(io.BytesIO(bytes(h5_file[key][()])), map_location="cpu", weights_only=False)
            frame_id = int(np.asarray(frame.get("id", frame_key_sort(key))).reshape(-1)[0])
            pose = np.asarray(frame["T_WC"], dtype=np.float64).reshape(-1, 8)[-1]
            positions_by_frame[frame_id] = pose[:3]
    return trajectory_from_map(positions_by_frame)


def load_kitti_ground_truth(gt_path):
    data = np.loadtxt(gt_path)
    data = np.atleast_2d(data)
    positions_by_frame = {}
    for row_idx, row in enumerate(data):
        if row.size < 8:
            raise ValueError(f"KITTI-360 gt_local.txt row must have at least 8 values, got {row.size}.")
        positions_by_frame[row_idx] = row[1:4].astype(np.float64)
    return trajectory_from_map(positions_by_frame)


def frame_key_sort(key):
    try:
        return int(str(key).split("_")[-1])
    except ValueError:
        return str(key)


def trajectory_index(trajectory):
    return {int(row[0]): row[1:4] for row in trajectory}


def window_start_trajectory(start_frame_ids, prior_traj):
    prior_by_frame = trajectory_index(prior_traj)
    rows = [[frame_id, *prior_by_frame[frame_id]] for frame_id in start_frame_ids if frame_id in prior_by_frame]
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)


def make_trace(trajectory, name, color, mode="lines+markers", symbol="circle", size=4):
    import plotly.graph_objects as go

    if trajectory.size == 0:
        print(f"[WARN] empty trajectory skipped: {name}")
        return None
    return go.Scatter3d(
        x=trajectory[:, 1],
        y=trajectory[:, 2],
        z=trajectory[:, 3],
        mode=mode,
        name=name,
        text=[f"frame_id={int(frame_id)}" for frame_id in trajectory[:, 0]],
        hovertemplate="%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{z:.6g}<extra></extra>",
        line={"color": color, "width": 4},
        marker={"color": color, "size": size, "symbol": symbol},
    )


def write_plotly_html(output_path, prior_traj, pi3x_traj, slam_traj, start_traj, gt_traj=None):
    import plotly.graph_objects as go

    traces = [
        make_trace(prior_traj, "PI3X input prior / IMU preintegration", "orange", symbol="circle", size=3),
        make_trace(pi3x_traj, "PI3X output pose", "green", symbol="diamond", size=3),
        make_trace(slam_traj, "SLAM H5 keyframes", "blue", symbol="circle", size=3),
        make_trace(start_traj, "PI3X window starts", "red", mode="markers", symbol="circle", size=6),
    ]
    if gt_traj is not None:
        traces.append(make_trace(gt_traj, "KITTI ground truth", "black", mode="lines", size=2))
    fig = go.Figure(data=[trace for trace in traces if trace is not None])
    fig.update_layout(
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def default_output_path(debug_csv):
    return Path("pi3x-viz.html")


def parse_args():
    parser = argparse.ArgumentParser(description="Create an interactive PI3X pose-debug trajectory HTML.")
    parser.add_argument("--debug-file", type=Path, required=True, help="Path to result.txt.pi3x_pose_debug.csv.")
    parser.add_argument("--h5", type=Path, required=True, help="Path to data.h5.")
    parser.add_argument("--gt-pose", type=Path, default=None, help="Optional KITTI official ground-truth pose txt.")
    return parser.parse_args()


def main():
    args = parse_args()
    prior_traj, pi3x_traj, start_frame_ids = load_pi3x_debug(args.debug_file)
    slam_traj = load_h5_keyframes(args.h5)
    gt_traj = load_kitti_ground_truth(args.gt_pose) if args.gt_pose is not None else None
    start_traj = window_start_trajectory(start_frame_ids, prior_traj)
    output_path = default_output_path(args.debug_file)
    write_plotly_html(output_path, prior_traj, pi3x_traj, slam_traj, start_traj, gt_traj)
    print(f"[INFO] saved {output_path}")


if __name__ == "__main__":
    main()
