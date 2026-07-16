import argparse
import csv
import gc
import os
import pathlib
import sys
import time
import lietorch
import torch
import yaml
from mast3r_fusion.pi3x_global_opt import FactorGraph

from mast3r_fusion.config import load_config, config
from mast3r_fusion.dataloader import Intrinsics, load_dataset
from mast3r_fusion.foxglove_debug import run_foxglove_publisher
from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_fusion.frontend_model import load_frontend_model
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg
from mast3r_fusion.sparse_flow_frontend import overlay_to_uimg_tensor
from mast3r_fusion.sparse_map import SparseMap
from mast3r_fusion.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation

import io
import h5py

pi3x_pose_debug_path = None
PI3X_POSE_DEBUG_HEADER = (
    ["record_time", "window_start", "window_end", "local_idx", "window_idx", "frame_id", "timestamp"]
    + [f"prior_{i}" for i in range(16)]
    + [f"pi3x_{i}" for i in range(16)]
)

def matrix_to_sim3(T, device='cpu', scale=1.0):
    TSim3 = lietorch.Sim3.Identity(1, device=device)
    q = Rotation.from_matrix(T[0:3, 0:3]).as_quat()
    TSim3[0].data[0] = T[0, 3]
    TSim3[0].data[1] = T[1, 3]
    TSim3[0].data[2] = T[2, 3]
    TSim3[0].data[3] = q[0]
    TSim3[0].data[4] = q[1]
    TSim3[0].data[5] = q[2]
    TSim3[0].data[6] = q[3]
    TSim3[0].data[7] = scale
    return TSim3


def sim3_to_se3_matrix(T_WC):
    T_WC64 = lietorch.Sim3(T_WC.data.to(torch.float64))
    matrix = T_WC64.matrix().detach().cpu().numpy()[0]
    scale = T_WC64.data.reshape(-1, T_WC64.data.shape[-1])[0, -1].detach().cpu().item()
    if np.isfinite(scale) and abs(scale) > 1e-8:
        matrix[:3, :3] /= scale
    return matrix


def pose_timestamp(frame_id):
    stamps = factor_graph.poses_stamps
    if hasattr(stamps, "get"):
        return float(stamps.get(int(frame_id), np.nan))
    try:
        return float(stamps[int(frame_id)])
    except Exception:
        return float("nan")


def save_pi3x_pose_debug(window_indices, window_frames, pi3x_poses):
    if pi3x_pose_debug_path is None:
        return
    poses_np = pi3x_poses.detach().cpu().numpy() if torch.is_tensor(pi3x_poses) else np.asarray(pi3x_poses)
    record_time = time.time()
    window_start = int(min(window_indices))
    window_end = int(max(window_indices))
    with open(pi3x_pose_debug_path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for local_idx, (window_idx, frame) in enumerate(zip(window_indices, window_frames)):
            prior_T = sim3_to_se3_matrix(frame.T_WC).reshape(-1)
            pi3x_T = np.asarray(poses_np[local_idx], dtype=np.float64).reshape(4, 4).reshape(-1)
            writer.writerow(
                [
                    record_time,
                    window_start,
                    window_end,
                    local_idx,
                    int(window_idx),
                    int(frame.frame_id),
                    pose_timestamp(frame.frame_id),
                    *prior_T.tolist(),
                    *pi3x_T.tolist(),
                ]
            )


def integrate_camera_gyro_prior(factor_graph, t0, t1):
    if t0 is None or t1 is None or t1 <= t0 or not hasattr(factor_graph, "imu_pool"):
        return None
    try:
        records = factor_graph.imu_pool.get_records(float(t0), float(t1))
    except Exception as exc:
        print(f"[WARN] sparse flow IMU prior skipped: {exc}")
        return None

    R_imu = np.eye(3, dtype=np.float64)
    for seg_t0, seg_t1, data in records:
        dt = float(seg_t1 - seg_t0)
        if dt <= 0:
            continue
        R_imu = R_imu @ Rotation.from_rotvec(np.asarray(data[0:3], dtype=np.float64) * np.pi / 180.0 * dt).as_matrix()

    R_ic = np.asarray(factor_graph.Tic[:3, :3], dtype=np.float64)
    return R_ic.T @ R_imu @ R_ic


def get_backend_edges(idx):
    """Return local backend edges for the PI3X delayed keyframe.

    Delayed PI3X windows are built from consecutive keyframes and then
    converted into pairwise factors after one multi-frame inference pass.
    """
    kf_idx = []
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame_idx = [idx] * len(kf_idx)
    return kf_idx, frame_idx


def finish_backend_update(
    states,
    keyframes,
    skip_marginalization=False,
    window_start=None,
    window_end=None,
    marginalize_to=None,
):
    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    factor_graph.solve_GN_calib(
        config["use_calib"],
        skip_marginalization=skip_marginalization,
        window_start=window_start,
        window_end=window_end,
    )

    # the fisrt time that VI init is finished
    # transform current states
    if factor_graph.init_vi_signal:
        factor_graph.solve_GN_calib(
            config["use_calib"],
            skip_marginalization=skip_marginalization,
            window_start=window_start,
            window_end=window_end,
        )
        factor_graph.init_vi_signal = False
        states.T_WC[:] = factor_graph.frames.last_keyframe().T_WC[:].data

        for i in range(int(keyframes.n_size.value)):
            frame_id = keyframes.dataset_idx[i].item()
            dd = keyframes.T_WC[i].data.cpu().numpy()[0]
            bb = factor_graph.bs[i].vector()
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()

    if marginalize_to is not None:
        print('[INFO] post optim marg', time.time(), marginalize_to)
        factor_graph.marginalize_to(marginalize_to)
        print('[INFO] post optim marg.', time.time())


def add_precomputed_factor_matches(factor_graph, ii, jj, matches, min_match_frac):
    if not ii:
        return False

    (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii,
        Qjj,
        Qji,
        Qij,
    ) = [torch.cat(values, dim=0) for values in zip(*matches)]

    batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
        :, None
    ].repeat(1, idx_i2j.shape[1])

    w = factor_graph.frames[ii[0]].img_true_shape[0, 1].item()
    idx_i2j_orig = idx_i2j.clone()
    idx_i2j_orig = (idx_i2j_orig // (factor_graph.subpixel_factor * w)) // factor_graph.subpixel_factor * w + (
        idx_i2j_orig % (factor_graph.subpixel_factor * w)
    ) // factor_graph.subpixel_factor
    idx_j2i_orig = idx_j2i.clone()
    idx_j2i_orig = (idx_j2i_orig // (factor_graph.subpixel_factor * w)) // factor_graph.subpixel_factor * w + (
        idx_j2i_orig % (factor_graph.subpixel_factor * w)
    ) // factor_graph.subpixel_factor

    Qj = torch.sqrt(Qii[batch_inds, idx_i2j_orig] * Qji)
    Qi = torch.sqrt(Qjj[batch_inds, idx_j2i_orig] * Qij)

    valid_Qj = Qj > factor_graph.cfg["Q_conf"]
    valid_Qi = Qi > factor_graph.cfg["Q_conf"]
    valid_j = valid_match_j & valid_Qj
    valid_i = valid_match_i & valid_Qi
    nj = valid_j.shape[1] * valid_j.shape[2]
    ni = valid_i.shape[1] * valid_i.shape[2]
    match_frac_j = valid_j.sum(dim=(1, 2)) / nj
    match_frac_i = valid_i.sum(dim=(1, 2)) / ni

    ii_tensor = torch.as_tensor(ii, device=factor_graph.device)
    jj_tensor = torch.as_tensor(jj, device=factor_graph.device)
    invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
    invalid_edges_orig = invalid_edges.clone()
    consecutive_edges = ii_tensor == (jj_tensor - 1)
    invalid_edges = (~consecutive_edges) & invalid_edges

    valid_edges = ~invalid_edges
    ii_tensor = ii_tensor[valid_edges]
    jj_tensor = jj_tensor[valid_edges]
    idx_i2j = idx_i2j[valid_edges]
    idx_j2i = idx_j2i[valid_edges]
    valid_match_j = valid_match_j[valid_edges]
    valid_match_i = valid_match_i[valid_edges]
    Qj[invalid_edges_orig, :] *= 0.0001
    Qi[invalid_edges_orig, :] *= 0.0001
    Qj = Qj[valid_edges]
    Qi = Qi[valid_edges]

    factor_graph.ii = torch.cat([factor_graph.ii, ii_tensor])
    factor_graph.jj = torch.cat([factor_graph.jj, jj_tensor])
    factor_graph.idx_ii2jj = torch.cat([factor_graph.idx_ii2jj, idx_i2j])
    factor_graph.idx_jj2ii = torch.cat([factor_graph.idx_jj2ii, idx_j2i])
    factor_graph.valid_match_j = torch.cat([factor_graph.valid_match_j, valid_match_j])
    factor_graph.valid_match_i = torch.cat([factor_graph.valid_match_i, valid_match_i])
    factor_graph.Q_ii2jj = torch.cat([factor_graph.Q_ii2jj, Qj])
    factor_graph.Q_jj2ii = torch.cat([factor_graph.Q_jj2ii, Qi])

    factor_graph.save_match_visualizations(
        ii_tensor, jj_tensor, idx_i2j, valid_match_j
    )

    retain_mask = torch.logical_not(
        torch.logical_and(
            factor_graph.ii < torch.max(factor_graph.ii) - 20,
            factor_graph.jj < torch.max(factor_graph.jj) - factor_graph.retain_num,
        )
    )
    factor_graph.ii = factor_graph.ii[retain_mask]
    factor_graph.jj = factor_graph.jj[retain_mask]
    factor_graph.idx_ii2jj = factor_graph.idx_ii2jj[retain_mask]
    factor_graph.idx_jj2ii = factor_graph.idx_jj2ii[retain_mask]
    factor_graph.valid_match_j = factor_graph.valid_match_j[retain_mask]
    factor_graph.valid_match_i = factor_graph.valid_match_i[retain_mask]
    factor_graph.Q_ii2jj = factor_graph.Q_ii2jj[retain_mask]
    factor_graph.Q_jj2ii = factor_graph.Q_jj2ii[retain_mask]

    return valid_edges.sum() > 0


def run_pi3x_window_backend_indices(states, keyframes, indices):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused() or not indices:
        return False

    pending_indices = list(indices)
    all_kf_idx = []
    all_frame_idx = []
    for idx in pending_indices:
        kf_idx, frame_idx = get_backend_edges(idx)
        all_kf_idx += kf_idx
        all_frame_idx += frame_idx
    if not all_kf_idx:
        finish_backend_update(states, keyframes)
        indices.clear()
        return True

    window_indices = sorted(set(all_kf_idx + all_frame_idx))
    window_index_to_local = {idx: local for local, idx in enumerate(window_indices)}
    window_frames = [keyframes[idx] for idx in window_indices]
    local_edges = [
        (window_index_to_local[ii], window_index_to_local[jj])
        for ii, jj in zip(all_kf_idx, all_frame_idx)
    ]

    print('[INFO] pi3x window inference', time.time(), window_indices)
    Xs, Cs, poses, constraints = factor_graph.model.build_pair_constraints_from_window(
        window_frames,
        local_edges,
        subpixel_factor=factor_graph.subpixel_factor,
    )
    save_pi3x_pose_debug(window_indices, window_frames, poses)
    for local_idx, frame in enumerate(window_frames):
        frame.update_pointmap(Xs[local_idx : local_idx + 1], Cs[local_idx : local_idx + 1])
        keyframes[window_indices[local_idx]] = frame

    matches = [constraints[edge] for edge in local_edges]
    window_start = min(all_kf_idx + all_frame_idx)
    window_end = max(all_kf_idx + all_frame_idx)
    print('[INFO] add pi3x window factor', time.time())
    added_edges = add_precomputed_factor_matches(
        factor_graph,
        all_kf_idx,
        all_frame_idx,
        matches,
        config["local_opt"]["min_match_frac"],
    )
    print('[INFO] add pi3x window factor.', time.time())
    if not added_edges:
        raise RuntimeError(f"PI3X delayed window produced no valid backend edges: {window_indices}")
    finish_backend_update(
        states,
        keyframes,
        skip_marginalization=True,
        window_start=window_start,
        window_end=window_end,
        marginalize_to=max(window_end - factor_graph.window_num, 0),
    )
    indices.clear()
    return True


def run_delayed_imu_backend(states, keyframes, idx):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return False

    print('[INFO] delayed imu backend', time.time(), idx)
    optimized = factor_graph.solve_imu_prior_window(window_end=idx)
    if optimized:
        latest_frame = keyframes[idx]
        states.T_WC[:] = latest_frame.T_WC[:].data
    print('[INFO] delayed imu backend.', time.time(), optimized)
    return optimized


def set_sparse_map_overlay(states, overlay_image):
    if overlay_image is None:
        return
    overlay = overlay_to_uimg_tensor(overlay_image, dtype=states.uimg.dtype)
    with states.lock:
        states.uimg[:] = overlay
    states.notify_frame_updated()


def update_current_state(states, frame, overlay_image=None):
    with states.lock:
        states.dataset_idx[:] = frame.frame_id
        states.img[:] = frame.img
        states.uimg[:] = overlay_to_uimg_tensor(overlay_image, dtype=states.uimg.dtype) if overlay_image is not None else frame.uimg
        states.img_shape[:] = frame.img_shape
        states.img_true_shape[:] = frame.img_true_shape
        states.T_WC[:] = frame.T_WC.data
    states.notify_frame_updated()


def update_sparse_map_points(states, sparse_map):
    points = sparse_map.export_point_cloud()
    states.set_sparse_map_points(points)


def align_sparse_map_to_last_keyframe(sparse_map, keyframes):
    last_kf = keyframes.last_keyframe()
    if last_kf is None:
        return
    last_kf_idx = len(keyframes) - 1 + keyframes.rollup_sum.value
    sparse_map.align_world_to_keyframe(last_kf_idx, last_kf)


def initialize_delayed_keyframe_placeholders(states, frame):
    frame.X_canon = torch.zeros_like(states.X)
    frame.C = torch.zeros_like(states.C)
    frame.feat = torch.zeros_like(states.feat)
    frame.pos = torch.zeros_like(states.pos)
    frame.N = 0
    frame.N_updates = 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="config/intrinsics_zyx.yaml")
    parser.add_argument("--imu_path", default="")
    parser.add_argument("--imu_dt", type = float, default=-0.0)
    parser.add_argument("--stamp_path", default="")
    parser.add_argument("--result_path", default="result.txt")
    parser.add_argument("--start_from", type =  int, default=0)
    parser.add_argument("--end_at", type =  int, default=-1)
    parser.add_argument("--save_h5", action="store_true")
    parser.add_argument("--pi3x_pose_debug_path", default=None)
    parser.add_argument("--frontend-model", choices=["pi3x"], default=None)
    parser.add_argument("--frontend-weights", default=None)
    parser.add_argument("--foxglove", action="store_true", help="Enable Foxglove WebSocket debug publisher.")
    parser.add_argument("--foxglove-host", default="127.0.0.1")
    parser.add_argument("--foxglove-port", type=int, default=8765)
    parser.add_argument("--foxglove-hz", type=float, default=5.0)


    args = parser.parse_args()
    if args.pi3x_pose_debug_path is None:
        result_path = pathlib.Path(args.result_path)
        pi3x_pose_debug_path = str(result_path.with_suffix(result_path.suffix + ".pi3x_pose_debug.csv"))
    else:
        pi3x_pose_debug_path = args.pi3x_pose_debug_path
    pathlib.Path(pi3x_pose_debug_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pi3x_pose_debug_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(PI3X_POSE_DEBUG_HEADER)
    print(f"[INFO] PI3X pose debug: {pi3x_pose_debug_path}")
    load_config(args.config)

    if not args.no_viz and os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print("[WARN] No display server found; disabling visualization. Pass --no-viz explicitly for headless runs.")
        args.no_viz = True


    if args.save_h5:
        f_h5 = h5py.File('data.h5', "w")
    
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset,args.stamp_path)
    dataset.subsample(config["dataset"]["subsample"],args.start_from,args.end_at)
    h, w = dataset.get_img_shape()[0]
    
    if args.calib and config["use_calib"]:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
            False, intrinsics.get("model","pinhole"), intrinsics.get("scale",1), intrinsics.get("height_new",None)
        )
    if (
        config.get("dataset", {}).get("target_img_size") is None
        and not (intrinsics.get("height_new",None) is None)
    ):
        h = intrinsics.get("height_new",None) * w // intrinsics["width"]

    model = load_frontend_model(
        name=args.frontend_model,
        path=args.frontend_weights,
        device=device,
    )
    if model.name != "pi3x":
        raise ValueError("main_pi3x_delayed.py only supports the PI3X frontend.")
    model.share_memory()
    feature_spec = model.get_feature_spec()

    keyframes = SharedKeyframes(manager, h, w, feature_spec=feature_spec)
    states = SharedStates(manager, h, w, feature_spec=feature_spec)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    if args.foxglove:
        foxglove = mp.Process(
            target=run_foxglove_publisher,
            args=(
                config,
                states,
                keyframes,
                args.calib,
                args.foxglove_host,
                args.foxglove_port,
                args.foxglove_hz,
                0.5,
                120000,
                10,
                70,
            ),
        )
        foxglove.start()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]
    pi3x_cfg = config.get("pi3x", {})

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    sparse_map_cfg = pi3x_cfg.get("sparse_map", {})
    if not sparse_map_cfg.get("enabled", True):
        raise ValueError("main_pi3x_delayed.py requires pi3x.sparse_map.enabled=true.")
    if K is None:
        raise ValueError("SparseMap requires calibrated camera intrinsics.")
    sparse_map = SparseMap.from_config(
        K.detach().cpu().numpy(),
        w,
        h,
        sparse_map_cfg,
    )

    last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device, args)
    factor_graph.poses_stamps = dataset.timestamps
    if not pi3x_cfg.get("delayed_matching", False):
        raise ValueError("main_pi3x_delayed.py requires pi3x.delayed_matching=true.")
    delayed_batch_keyframes = max(1, int(pi3x_cfg.get("delayed_batch_keyframes", 5)))
    pending_delayed_kf_idx = []
    i = 0
    fps_timer = time.time()
    prev_frame_timestamp = None

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            if pending_delayed_kf_idx:
                run_pi3x_window_backend_indices(states, keyframes, pending_delayed_kf_idx)
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        # time.sleep(0.2)

        camera_gyro_R = None
        if pi3x_cfg.get("imu_predict", False):
            camera_gyro_R = integrate_camera_gyro_prior(factor_graph, prev_frame_timestamp, timestamp)

        TSim3 = lietorch.Sim3.Identity(1, device='cpu')
        Tic0 = np.array([1, 0,  0, 0,
                         0, 0,  1, 0,
                         0,-1,  0, 0,
                         0, 0,  0, 1]).reshape([4,4]) 
        TTTc = Tic0
        qqq = Rotation.from_matrix(TTTc[0:3,0:3]).as_quat()
        TSim3[0].data[0] = TTTc[0,3]
        TSim3[0].data[1] = TTTc[1,3]
        TSim3[0].data[2] = TTTc[2,3]
        TSim3[0].data[3] = qqq[0]
        TSim3[0].data[4] = qqq[1]
        TSim3[0].data[5] = qqq[2]
        TSim3[0].data[6] = qqq[3]
        # get frames last camera pose
        T_WC = (
            TSim3
            if i == 0
            else states.get_frame().T_WC
        )
        wTc_pred = None
        pred_dt = float("inf")
        use_imu_pose_prior = (
            pi3x_cfg.get("imu_predict", False)
            and factor_graph.enable_ms
            and i > 100
        )
        if use_imu_pose_prior:
            dT, wTc_pred, pred_dt = factor_graph.predict_pose(i)
            if pred_dt <= pi3x_cfg.get("pose_prior_max_dt", 5.0):
                T_WC = matrix_to_sim3(wTc_pred)
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        if use_calib:
            frame.K = K
        align_sparse_map_to_last_keyframe(sparse_map, keyframes)
        last_kf = keyframes.last_keyframe()
        last_kf_frame_id = last_kf.frame_id if last_kf is not None else -1
        tracking_result = sparse_map.process_frame(
            frame,
            timestamp,
            gyro_R=camera_gyro_R,
            frames_since_keyframe=frame.frame_id - last_kf_frame_id,
        )
        sparse_map_overlay = sparse_map.draw_overlay(tracking_result)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = model.infer_single(frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            initial_keyframe_idx = len(keyframes) - 1 + keyframes.rollup_sum.value
            sparse_map.register_keyframe(initial_keyframe_idx, frame)
            update_sparse_map_points(states, sparse_map)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame, notify=sparse_map_overlay is None)
            set_sparse_map_overlay(states, sparse_map_overlay)
            prev_frame_timestamp = timestamp
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf = sparse_map.need_new_keyframe(
                frame.frame_id,
                last_kf_frame_id,
                tracking_result,
            )
            if add_new_kf:
                sparse_map_was_initialized = sparse_map.initialized
                initialize_delayed_keyframe_placeholders(states, frame)
                keyframes.append(frame)
                delayed_kf_idx = len(keyframes) - 1 + keyframes.rollup_sum.value
                states.set_frame(frame, notify=sparse_map_overlay is None)
                set_sparse_map_overlay(states, sparse_map_overlay)
                if factor_graph.enable_ms:
                    pending_delayed_kf_idx.append(delayed_kf_idx)
                    if sparse_map_was_initialized:
                        sparse_map.register_keyframe(
                            delayed_kf_idx,
                            frame,
                            tracking_result,
                        )
                    run_delayed_imu_backend(states, keyframes, delayed_kf_idx)
                    if not sparse_map_was_initialized:
                        sparse_map.register_keyframe(
                            delayed_kf_idx,
                            keyframes[delayed_kf_idx],
                            tracking_result,
                        )
                    if len(pending_delayed_kf_idx) >= delayed_batch_keyframes:
                        run_pi3x_window_backend_indices(states, keyframes, pending_delayed_kf_idx)
                else:
                    sparse_map.register_keyframe(
                        delayed_kf_idx,
                        frame,
                        tracking_result,
                    )
                    run_pi3x_window_backend_indices(states, keyframes, [delayed_kf_idx])
            else:
                update_current_state(states, frame, sparse_map_overlay)
        else:
            raise Exception("Invalid mode")
        prev_frame_timestamp = timestamp

        print('[INFO] backend',time.time())

        print(factor_graph.frames_to_save)
        if args.save_h5:
            for iframe in factor_graph.frames_to_save:
                frame_temp = keyframes[iframe] 
                buffer = io.BytesIO()
                torch.save({
                    'feat': frame_temp.feat.cpu(), 
                    'pos': frame_temp.pos.cpu(),   
                    'X': frame_temp.X_canon.cpu(),
                    'C': frame_temp.C.cpu(),
                    'K': frame_temp.K.cpu(),
                    'N': frame_temp.N,
                    'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),
                    'img_shape': frame_temp.img_shape.cpu(),
                    'T_WC': frame_temp.T_WC.data.cpu(),
                    'id': frame_temp.frame_id,
                }, buffer)
                buffer.seek(0)
                f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))
        factor_graph.frames_to_save = []



        # write results
        dd = states.T_WC[0].data.cpu().numpy()
        frame_id = frame.frame_id
        try:
            bb = factor_graph.bs[-1].vector()
        except:
            bb = np.zeros(6)
        if factor_graph.enable_ms and frame.frame_id>100 and wTc_pred is not None and pred_dt < 5.0: # IMU prediction
            dd = np.concatenate([wTc_pred[0:3,3],Rotation.from_matrix(wTc_pred[0:3,0:3]).as_quat(),np.array([1.0])])
        factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 0\n' % (factor_graph.poses_stamps[frame_id],
                                                                             dd[0].item(),
                                                                             dd[1].item(),
                                                                             dd[2].item(),
                                                                             dd[3].item(),
                                                                             dd[4].item(),
                                                                             dd[5].item(),
                                                                             dd[6].item(),
                                                                             dd[7].item(),
                                                                             bb[0],bb[1],bb[2],
                                                                             bb[3],bb[4],bb[5],
                                                                             frame_id))
        factor_graph.fp.flush()
        
        if add_new_kf:
            dd = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            frame_id = keyframes.last_keyframe().frame_id
            try:
                bb = factor_graph.bs[-1].vector()
            except:
                bb = np.zeros(6)
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()

        print('[INFO] backend.',time.time())


        # handling sliding window
        # notice that we main very few frames to save GPU memory usage
        # generally 8 GB is enough
        if len(keyframes) > 30:
            if pending_delayed_kf_idx:
                run_pi3x_window_backend_indices(states, keyframes, pending_delayed_kf_idx)
            rollup = 15
            rollup = min(rollup, max(factor_graph.last_pin - keyframes.rollup_sum.value, 0))
            if rollup > 0:
                keyframes.roll_up(rollup)
                sparse_map.prune_before(keyframes.rollup_sum.value)

        align_sparse_map_to_last_keyframe(sparse_map, keyframes)
        update_sparse_map_points(states, sparse_map)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1


    if args.save_h5:
        last_pin = factor_graph.get_unique_kf_idx()[-1]
        for iframe in range(factor_graph.last_pin,last_pin+1):
            frame_temp = keyframes[iframe] 
            buffer = io.BytesIO()
            torch.save({
                'feat': frame_temp.feat.cpu(), 
                'pos': frame_temp.pos.cpu(),   
                'X': frame_temp.X_canon.cpu(),
                'C': frame_temp.C.cpu(),
                'K': frame_temp.K.cpu(),
                'N': frame_temp.N,
                'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),
                'img_shape': frame_temp.img_shape.cpu(),
                'T_WC': frame_temp.T_WC.data.cpu(),
                'id': frame_temp.frame_id,
            }, buffer)
            buffer.seek(0)
            f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))
        f_h5.close()

    factor_graph.save_graph('graph.pkl')

    print("done")
    states.set_mode(Mode.TERMINATED)
    if not args.no_viz:
        viz.join()
    if args.foxglove:
        foxglove.join()
    factor_graph.close()
    manager.shutdown()
    torch.cuda.empty_cache()
    gc.collect()
    os._exit(0)
