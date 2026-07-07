import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_fusion.pi3x_delayed import Pi3XDelayedFactorGraph
from mast3r_fusion.pi3x_delayed.keyframes import (
    configure_feature_storage,
    limited_roll_up,
    set_keyframe_global,
)

from mast3r_fusion.config import load_config, config, set_global_config
from mast3r_fusion.dataloader import Intrinsics, load_dataset
import mast3r_fusion.evaluate as eval
from mast3r_fusion.foxglove_debug import run_foxglove_publisher
from mast3r_fusion.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_fusion.frontend_model import load_frontend_model
from mast3r_fusion.mast3r_utils import _crop_resize, load_retriever
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg
from mast3r_fusion.tracker import FrameTracker
from mast3r_fusion.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation

import pickle
import io
import h5py

def find_valid_numbers(a, b):
    result = []
    for i, c in enumerate(b):
        if abs(c - a) <= 1:
            continue 
        close_indices = [j for j, d in enumerate(b) if abs(d - c) <= 20]
        if i == min(close_indices) or c == a - 2 :
            result.append(c)
    return result


def matrix_to_sim3(T, device='cpu'):
    TSim3 = lietorch.Sim3.Identity(1, device=device)
    q = Rotation.from_matrix(T[0:3, 0:3]).as_quat()
    TSim3[0].data[0] = T[0, 3]
    TSim3[0].data[1] = T[1, 3]
    TSim3[0].data[2] = T[2, 3]
    TSim3[0].data[3] = q[0]
    TSim3[0].data[4] = q[1]
    TSim3[0].data[5] = q[2]
    TSim3[0].data[6] = q[3]
    TSim3[0].data[7] = 1.0
    return TSim3


def create_delayed_frame(i, img, T_WC, dataset, device="cuda:0"):
    target_img_size = config.get("dataset", {}).get("target_img_size")
    if target_img_size is None:
        return create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

    resized = _crop_resize(img, target_img_size)
    rgb = resized["img"].to(device=device)
    img_shape = torch.tensor(resized["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(resized["unnormalized_img"].copy()) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    return Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)


def get_backend_edges(idx, keyframes):
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]

    # find local(!) co-visible frames
    if retrieval_database is not None:
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
    else:
        retrieval_inds = []

    retrieval_inds_selected = []
    retrieval_inds = find_valid_numbers(idx,retrieval_inds)

    for kkk in retrieval_inds:
        if np.fabs(idx - kkk) < 20:
            retrieval_inds_selected.append(kkk)
    kf_idx += retrieval_inds_selected

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
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


def run_backend_indices(states, keyframes, indices):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused() or not indices:
        return False

    all_kf_idx = []
    all_frame_idx = []
    for idx in indices:
        kf_idx, frame_idx = get_backend_edges(idx, keyframes)
        all_kf_idx += kf_idx
        all_frame_idx += frame_idx

    print('[INFO] add factor',time.time())
    if all_kf_idx:
        factor_graph.add_factors(
            all_kf_idx, all_frame_idx, config["local_opt"]["min_match_frac"]
        )
    print('[INFO] add factor.',time.time())

    finish_backend_update(states, keyframes)
    return True


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

    all_kf_idx = []
    all_frame_idx = []
    for idx in indices:
        kf_idx, frame_idx = get_backend_edges(idx, keyframes)
        all_kf_idx += kf_idx
        all_frame_idx += frame_idx
    if not all_kf_idx:
        finish_backend_update(states, keyframes)
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
    for local_idx, frame in enumerate(window_frames):
        frame.update_pointmap(Xs[local_idx : local_idx + 1], Cs[local_idx : local_idx + 1])
        set_keyframe_global(keyframes, window_indices[local_idx], frame)

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
    return True


def run_backend(states, keyframes):
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    did_run = run_backend_indices(states, keyframes, [idx])
    if not did_run:
        return

    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)


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


def flush_delayed_backend(states, keyframes, pending_kf_idx):
    if not pending_kf_idx:
        return
    run_pi3x_window_backend_indices(states, keyframes, pending_kf_idx)
    pending_kf_idx.clear()


def set_frame_pose_only(states, frame):
    with states.lock:
        states.dataset_idx[:] = frame.frame_id
        states.img[:] = frame.img
        states.uimg[:] = frame.uimg
        states.img_shape[:] = frame.img_shape
        states.img_true_shape[:] = frame.img_true_shape
        states.T_WC[:] = frame.T_WC.data


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
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="config/intrinsics_zyx.yaml")
    parser.add_argument("--imu_path", default="")
    parser.add_argument("--imu_dt", type = float, default=-0.0)
    parser.add_argument("--stamp_path", default="")
    parser.add_argument("--result_path", default="result.txt")
    parser.add_argument("--start_from", type =  int, default=0)
    parser.add_argument("--end_at", type =  int, default=-1)
    parser.add_argument("--save_h5", action="store_true")
    parser.add_argument("--frontend-model", choices=["mast3r", "pi3", "pi3x"], default=None)
    parser.add_argument("--frontend-weights", default=None)
    parser.add_argument("--foxglove", action="store_true", help="Enable Foxglove WebSocket debug publisher.")
    parser.add_argument("--foxglove-host", default="127.0.0.1")
    parser.add_argument("--foxglove-port", type=int, default=8765)
    parser.add_argument("--foxglove-hz", type=float, default=5.0)


    args = parser.parse_args()
    load_config(args.config)


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
    model.share_memory()
    feature_spec = model.get_feature_spec() if hasattr(model, "get_feature_spec") else None

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)
    configure_feature_storage(keyframes, h, w, feature_spec)
    configure_feature_storage(states, h, w, feature_spec)

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
                1.5,
                120000,
                10,
                70,
            ),
        )
        foxglove.start()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"\

        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    factor_graph = Pi3XDelayedFactorGraph(model, keyframes, K, device, args)
    factor_graph.poses_stamps = dataset.timestamps
    pi3x_cfg = config.get("pi3x", {})
    pi3x_delayed_matching = (
        getattr(model, "name", "mast3r") == "pi3x"
        and pi3x_cfg.get("delayed_matching", False)
        and pi3x_cfg.get("imu_predict", False)
    )
    delayed_batch_keyframes = max(1, int(pi3x_cfg.get("delayed_batch_keyframes", 5)))
    delayed_keyframe_stride = max(1, int(pi3x_cfg.get("delayed_keyframe_stride", 2)))
    pending_delayed_kf_idx = []
    
    if getattr(model, "name", "mast3r") == "mast3r":
        retrieval_database = load_retriever(model)
    else:
        retrieval_database = None

    i = 0
    fps_timer = time.time()

    frames = []

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
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        # time.sleep(0.2)
        if save_frames:
            frames.append(img)


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
            getattr(model, "name", "mast3r") == "pi3x"
            and pi3x_cfg.get("imu_predict", False)
            and factor_graph.enable_ms
            and i > 100
        )
        if use_imu_pose_prior:
            dT, wTc_pred, pred_dt = factor_graph.predict_pose(i)
            if pred_dt <= pi3x_cfg.get("pose_prior_max_dt", 5.0):
                T_WC = matrix_to_sim3(wTc_pred)
        frame = create_delayed_frame(i, img, T_WC, dataset, device=device)
        if use_calib:
            frame.K = K

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = model.infer_single(frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        add_new_kf = False
        delayed_frame = False
        if mode == Mode.TRACKING:
            delayed_frame = (
                pi3x_delayed_matching
                and factor_graph.enable_ms
                and pred_dt <= pi3x_cfg.get("pose_prior_max_dt", 5.0)
            )
            if delayed_frame:
                last_kf = keyframes.last_keyframe()
                last_kf_frame_id = last_kf.frame_id if last_kf is not None else -delayed_keyframe_stride
                add_new_kf = frame.frame_id - last_kf_frame_id >= delayed_keyframe_stride
                try_reloc = False
                match_info = []
                if add_new_kf:
                    initialize_delayed_keyframe_placeholders(states, frame)
                    keyframes.append(frame)
                    delayed_kf_idx = len(keyframes) - 1 + keyframes.rollup_sum.value
                    pending_delayed_kf_idx.append(delayed_kf_idx)
                    states.set_frame(frame)
                    run_delayed_imu_backend(states, keyframes, delayed_kf_idx)
                    if len(pending_delayed_kf_idx) >= delayed_batch_keyframes:
                        flush_delayed_backend(states, keyframes, pending_delayed_kf_idx)
                else:
                    set_frame_pose_only(states, frame)
            else:
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)
        elif mode == Mode.RELOC:
            X, C = model.infer_single(frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
        else:
            raise Exception("Invalid mode")
        
        # using IMU prediction to adjust keyframe selectiion
        if (not delayed_frame) and factor_graph.enable_ms and frame.frame_id>100:
            dd_old = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            dd_new = states.T_WC[0].data.cpu().numpy()
            dT, wTc_pred, pred_dt = factor_graph.predict_pose(frame.frame_id)
            if pred_dt > 5.0: # if prediction is too long, just use visual tracking
                pass #do nothing
            else:
                if (not add_new_kf) and  np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())>30.0/57.3:
                    add_new_kf = True
                    tracker.reset_idx_f2k()
                if add_new_kf and (np.linalg.norm(dT[0:3,3]) < 1.0 and np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())<5.0/57.3):
                    add_new_kf = False
                    tracker.idx_f2k = tracker.idx_f2k_backup
    
        if add_new_kf and not delayed_frame:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)
        print('[INFO] backend',time.time())
        if not delayed_frame:
            run_backend(states, keyframes)
        if pi3x_delayed_matching and pending_delayed_kf_idx and len(keyframes) > 30:
            flush_delayed_backend(states, keyframes, pending_delayed_kf_idx)
        
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
        dd = states.T_WC[0].data.cpu().numpy() # visual tracking
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
            bb = factor_graph.bs[-1].vector()
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
            if pi3x_delayed_matching and pending_delayed_kf_idx:
                flush_delayed_backend(states, keyframes, pending_delayed_kf_idx)
            if pi3x_delayed_matching:
                limited_roll_up(keyframes, 15, factor_graph.last_pin)
            else:
                keyframes.roll_up(15)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1


    # finally
    if pi3x_delayed_matching and pending_delayed_kf_idx:
        flush_delayed_backend(states, keyframes, pending_delayed_kf_idx)

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

    # if dataset.save_results:
    #     save_dir, seq_name = eval.prepare_savedir(args, dataset)
    #     eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
    #     eval.save_reconstruction(
    #         save_dir,
    #         f"{seq_name}.ply",
    #         keyframes,
    #         last_msg.C_conf_threshold,
    #     )
    #     eval.save_keyframes(
    #         save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
    #     )
    # if save_frames:
    #     savedir = pathlib.Path(f"logs/frames/{datetime_now}")
    #     savedir.mkdir(exist_ok=True, parents=True)
    #     for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
    #         frame = (frame * 255).clip(0, 255)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    states.set_mode(Mode.TERMINATED)
    if not args.no_viz:
        viz.join()
    if args.foxglove:
        foxglove.join()
