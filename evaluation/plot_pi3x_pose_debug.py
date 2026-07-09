import argparse
import csv
import io
from pathlib import Path

import numpy as np


def kitti360_camera_to_imu():
    import mast3r_fusion.geoFunc.trans as trans

    Tic = np.array(
        [
            [0.99944133, -0.00228419, -0.03334389, -0.03734697],
            [0.03268308, -0.14183394, 0.98935078, 1.75837780],
            [-0.00698916, -0.98988784, -0.14168005, 0.59911765],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000],
        ],
        dtype=np.float64,
    )
    Tic[:3, :3] = Tic[:3, :3] @ trans.att2m(np.array([-0.15, -0.1, 0.0], dtype=np.float64) / 57.3)
    return Tic


def camera_pose_to_imu_pose(T_wc, T_ci):
    return T_wc @ T_ci


def matrix_from_row(row, prefix):
    values = [float(row[f"{prefix}_{idx}"]) for idx in range(16)]
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def matrix_from_sim3_data(sim3_data):
    from scipy.spatial.transform import Rotation

    pose = np.asarray(sim3_data, dtype=np.float64).reshape(-1, 8)[-1]
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    T_wc[:3, 3] = pose[:3]
    return T_wc


def trajectory_from_map(positions_by_frame):
    if not positions_by_frame:
        return np.empty((0, 4), dtype=np.float64)
    rows = [[frame_id, *positions_by_frame[frame_id]] for frame_id in sorted(positions_by_frame)]
    return np.asarray(rows, dtype=np.float64)


def transform_trajectory(trajectory, T):
    if trajectory.size == 0:
        return trajectory
    transformed = trajectory.copy()
    transformed[:, 1:4] = trajectory[:, 1:4] @ T[:3, :3].T + T[:3, 3]
    return transformed


def align_to_gt_with_evo(source_traj, timestamp_by_frame, gt_traj, gt_timestamp_by_frame, align_count=200):
    from evo.core import lie_algebra, sync
    from evo.core.trajectory import PoseTrajectory3D

    source_rows = [row for row in source_traj if int(row[0]) in timestamp_by_frame]
    gt_rows = [row for row in gt_traj if int(row[0]) in gt_timestamp_by_frame]
    if len(source_rows) < 3 or len(gt_rows) < 3:
        print(f"[WARN] skip GT alignment: source={len(source_rows)}, gt={len(gt_rows)}")
        return np.eye(4, dtype=np.float64)

    source_rows = np.asarray(source_rows, dtype=np.float64)
    gt_rows = np.asarray(gt_rows, dtype=np.float64)
    traj_est = make_evo_trajectory(
        source_rows[:, 1:4],
        [timestamp_by_frame[int(frame_id)] for frame_id in source_rows[:, 0]],
    )
    traj_ref = make_evo_trajectory(
        gt_rows[:, 1:4],
        [gt_timestamp_by_frame[int(frame_id)] for frame_id in gt_rows[:, 0]],
    )
    traj_ref_sel, traj_est_sel = sync.associate_trajectories(traj_ref, traj_est, 0.01, 0.0)
    if traj_est_sel.num_poses < 3:
        print(f"[WARN] skip GT alignment: evo associated {traj_est_sel.num_poses} poses.")
        return np.eye(4, dtype=np.float64)
    n_to_align = min(align_count, traj_est_sel.num_poses)
    print(f"[INFO] align to GT with first {n_to_align} associated poses")
    return lie_algebra.sim3(*traj_est_sel.align(traj_ref_sel, correct_scale=False, n=n_to_align))


def make_evo_trajectory(positions_xyz, timestamps):
    from evo.core.trajectory import PoseTrajectory3D

    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    orientations = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (positions_xyz.shape[0], 1))
    return PoseTrajectory3D(
        positions_xyz=positions_xyz,
        orientations_quat_wxyz=orientations,
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def load_pi3x_debug(csv_path, T_ci):
    windows = {}
    timestamp_by_frame = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            key = (row["record_time"], row["window_start"], row["window_end"])
            windows.setdefault(key, []).append(row)
            timestamp = float(row["timestamp"])
            if np.isfinite(timestamp):
                timestamp_by_frame[int(row["frame_id"])] = timestamp

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
            prior_by_frame[frame_id] = camera_pose_to_imu_pose(prior_T, T_ci)[:3, 3]
            pi3x_by_frame[frame_id] = camera_pose_to_imu_pose(pi3x_T, T_ci)[:3, 3]

    return (
        trajectory_from_map(prior_by_frame),
        trajectory_from_map(pi3x_by_frame),
        sorted(set(window_start_frame_ids)),
        timestamp_by_frame,
    )


def align_window_to_prior(prior, pi3x):
    if len(prior) == 0:
        return pi3x
    return np.einsum("ij,njk->nik", prior[0] @ np.linalg.inv(pi3x[0]), pi3x)


def load_h5_keyframes(h5_path, T_ci):
    import h5py
    import torch

    positions_by_frame = {}
    with h5py.File(h5_path, "r") as h5_file:
        for key in sorted(h5_file.keys(), key=frame_key_sort):
            frame = torch.load(io.BytesIO(bytes(h5_file[key][()])), map_location="cpu", weights_only=False)
            frame_id = int(np.asarray(frame.get("id", frame_key_sort(key))).reshape(-1)[0])
            T_wc = matrix_from_sim3_data(frame["T_WC"])
            positions_by_frame[frame_id] = camera_pose_to_imu_pose(T_wc, T_ci)[:3, 3]
    return trajectory_from_map(positions_by_frame)


def load_kitti_ground_truth(gt_path):
    data = np.loadtxt(gt_path)
    data = np.atleast_2d(data)
    positions_by_frame = {}
    timestamp_by_frame = {}
    for row_idx, row in enumerate(data):
        if row.size < 8:
            raise ValueError(f"KITTI-360 gt_local.txt row must have at least 8 values, got {row.size}.")
        positions_by_frame[row_idx] = row[1:4].astype(np.float64)
        timestamp_by_frame[row_idx] = float(row[0])
    return trajectory_from_map(positions_by_frame), timestamp_by_frame


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
    T_ci = np.linalg.inv(kitti360_camera_to_imu())
    prior_traj, pi3x_traj, start_frame_ids, timestamp_by_frame = load_pi3x_debug(args.debug_file, T_ci)
    slam_traj = load_h5_keyframes(args.h5, T_ci)
    gt_data = load_kitti_ground_truth(args.gt_pose) if args.gt_pose is not None else None
    gt_traj = gt_data[0] if gt_data is not None else None
    start_traj = window_start_trajectory(start_frame_ids, prior_traj)
    if gt_traj is not None:
        T_align = align_to_gt_with_evo(slam_traj, timestamp_by_frame, gt_data[0], gt_data[1])
        prior_traj = transform_trajectory(prior_traj, T_align)
        pi3x_traj = transform_trajectory(pi3x_traj, T_align)
        slam_traj = transform_trajectory(slam_traj, T_align)
        start_traj = transform_trajectory(start_traj, T_align)
    output_path = default_output_path(args.debug_file)
    write_plotly_html(output_path, prior_traj, pi3x_traj, slam_traj, start_traj, gt_traj)
    print(f"[INFO] saved {output_path}")


if __name__ == "__main__":
    main()
