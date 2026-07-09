import argparse
import csv
import io
from pathlib import Path

import numpy as np


def load_pose_debug(path):
    prior_by_frame = {}
    pi3x_by_frame = {}
    window_start_frame_ids = []
    windows = {}

    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = (row["record_time"], row["window_start"], row["window_end"])
            windows.setdefault(key, []).append(row)

    for rows in windows.values():
        rows = sorted(rows, key=lambda row: int(row["local_idx"]))
        frame_ids = [int(row["frame_id"]) for row in rows]
        if frame_ids:
            window_start_frame_ids.append(frame_ids[0])
        prior = np.stack([matrix_from_row(row, "prior") for row in rows], axis=0)
        pi3x = np.stack([matrix_from_row(row, "pi3x") for row in rows], axis=0)
        if len(prior) > 0:
            pi3x = np.einsum("ij,njk->nik", prior[0] @ np.linalg.inv(pi3x[0]), pi3x)
        for frame_id, prior_T, pi3x_T in zip(frame_ids, prior, pi3x):
            prior_by_frame[frame_id] = prior_T[:3, 3]
            pi3x_by_frame[frame_id] = pi3x_T[:3, 3]

    return (
        dict_to_trajectory(prior_by_frame),
        dict_to_trajectory(pi3x_by_frame),
        sorted(set(window_start_frame_ids)),
    )


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


def load_h5_keyframes(path):
    import h5py
    import torch

    by_frame = {}
    with h5py.File(path, "r") as h5_file:
        for key in sorted(h5_file.keys(), key=frame_key_sort):
            data = bytes(h5_file[key][()])
            frame = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            frame_id = int(np.asarray(frame.get("id", frame_key_sort(key))).reshape(-1)[0])
            pose = np.asarray(frame["T_WC"], dtype=np.float64).reshape(-1, 8)[-1]
            by_frame[frame_id] = pose[:3]
    return dict_to_trajectory(by_frame)


def frame_key_sort(key):
    try:
        return int(str(key).split("_")[-1])
    except ValueError:
        return str(key)


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


def trajectory_map(traj):
    return {int(row[0]): row[1:4] for row in traj}


def plot_window_start_markers(ax, start_frame_ids, prior_traj):
    marker_traj = window_start_marker_traj(start_frame_ids, prior_traj)
    if marker_traj.size == 0:
        return
    ax.scatter(
        marker_traj[:, 1],
        marker_traj[:, 2],
        marker_traj[:, 3],
        color="red",
        s=36,
        marker="o",
        label="PI3X window starts",
        zorder=5,
    )


def window_start_marker_traj(start_frame_ids, prior_traj):
    prior_by_frame = trajectory_map(prior_traj)
    points = [
        [frame_id, *prior_by_frame[frame_id]]
        for frame_id in start_frame_ids
        if frame_id in prior_by_frame
    ]
    if not points:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def choose_backend(args):
    if args.backend != "auto":
        return args.backend
    return "plotly" if Path(args.output).suffix.lower() == ".html" else "matplotlib"


def plot_matplotlib(prior_traj, pi3x_traj, slam_traj, slam_label, window_start_frame_ids, args):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_traj(ax, prior_traj, "PI3X input prior / IMU preintegration", "tab:orange", marker="o")
    plot_traj(ax, pi3x_traj, "PI3X output pose", "tab:green", marker="^")
    plot_traj(ax, slam_traj, slam_label, "tab:blue")
    if not args.no_window_start_markers:
        plot_window_start_markers(ax, window_start_frame_ids, prior_traj)
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
    if args.show:
        plt.show()
    return output


def plotly_trace(traj, name, color, mode="lines+markers", symbol="circle", size=4):
    import plotly.graph_objects as go

    if traj.size == 0:
        print(f"[WARN] empty trajectory skipped: {name}")
        return None
    return go.Scatter3d(
        x=traj[:, 1],
        y=traj[:, 2],
        z=traj[:, 3],
        mode=mode,
        name=name,
        text=[f"frame_id={int(frame_id)}" for frame_id in traj[:, 0]],
        hovertemplate="%{text}<br>x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{z:.6g}<extra></extra>",
        line={"color": color, "width": 4},
        marker={"color": color, "size": size, "symbol": symbol},
    )


def plot_plotly(prior_traj, pi3x_traj, slam_traj, slam_label, window_start_frame_ids, args):
    import plotly.graph_objects as go

    traces = [
        plotly_trace(prior_traj, "PI3X input prior / IMU preintegration", "orange", symbol="circle", size=3),
        plotly_trace(pi3x_traj, "PI3X output pose", "green", symbol="diamond", size=3),
        plotly_trace(slam_traj, slam_label, "blue", mode="lines+markers", symbol="circle", size=3),
    ]
    if not args.no_window_start_markers:
        marker_traj = window_start_marker_traj(window_start_frame_ids, prior_traj)
        traces.append(plotly_trace(marker_traj, "PI3X window starts", "red", mode="markers", symbol="circle", size=6))

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

    output = Path(args.output)
    if output.suffix.lower() != ".html":
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    if args.show:
        fig.show()
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot PI3X pose-prior debug trajectories.")
    parser.add_argument("--debug-csv", required=True, help="Path to *.pi3x_pose_debug.csv.")
    parser.add_argument("--result", default=None, help="SLAM result.txt path.")
    parser.add_argument("--h5", default=None, help="data.h5 path for SLAM keyframe poses.")
    parser.add_argument("--output", default="pi3x_pose_debug_trajectory.png")
    parser.add_argument("--backend", choices=["auto", "matplotlib", "plotly"], default="auto")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-window-start-markers", action="store_true", help="Do not mark PI3X window start keyframes.")
    parser.add_argument("--slam-keyframes-only", action="store_true", help="Plot only result rows with keyframe flag == 1.")
    args = parser.parse_args()

    prior_traj, pi3x_traj, window_start_frame_ids = load_pose_debug(args.debug_csv)
    if args.h5 is not None:
        slam_traj = load_h5_keyframes(args.h5)
        slam_label = "SLAM H5 keyframes"
    elif args.result is not None:
        slam_traj = load_slam_result(args.result, keyframes_only=args.slam_keyframes_only)
        slam_label = "SLAM result"
    else:
        raise ValueError("Either --h5 or --result must be provided.")

    backend = choose_backend(args)
    if backend == "plotly":
        output = plot_plotly(prior_traj, pi3x_traj, slam_traj, slam_label, window_start_frame_ids, args)
    else:
        output = plot_matplotlib(prior_traj, pi3x_traj, slam_traj, slam_label, window_start_frame_ids, args)
    print(f"[INFO] saved {output}")


if __name__ == "__main__":
    main()
