import torch
import yaml
import pickle 
from mast3r_fusion.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg
from mast3r_fusion.config import load_config, config, set_global_config
from mast3r_fusion.dataloader import Intrinsics, load_dataset
from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame
import numpy as np
import h5py
import io
import cv2
from mast3r_fusion.geometry import (
    constrain_points_to_ray,
)
import matplotlib.pyplot as plt
from natsort import natsorted
import tqdm
import time
import argparse

def len_h5(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        return len(f.keys())

def mask_sky(img):
    mm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (mm > 250).astype(np.uint8)
    h, w = mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    out_mask = np.zeros_like(mask, dtype=np.uint8)
    for x in range(w):
        if mask[0, x] == 1:
            cv2.floodFill(mask, flood_mask, (x, 0), 2)
    out_mask = (mask == 2)
    return out_mask

def pose_xyz(pose):
    if hasattr(pose, "detach") and hasattr(pose, "cpu"):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose).reshape(-1)[:3].astype(np.float64)

def save_trajectory_overview(id_poses, selected_ids, frame_id, output_path):
    if output_path is None:
        return
    all_ids = sorted(id_poses.keys())
    if not all_ids:
        return
    x_series = []
    y_series = []
    for i in all_ids:
        xyz = pose_xyz(id_poses[i])
        x_series.append(xyz[0])
        y_series.append(xyz[1])
    selected_ids = [i for i in selected_ids if i in id_poses]
    ref_ids = [i for i in range(frame_id - 10, frame_id + 10) if i in id_poses]

    plt.figure("check_h5_trajectory_window", figsize=[10 * 0.7, 14 * 0.7])
    plt.clf()
    plt.subplot(1, 1, 1)
    plt.plot(
        x_series,
        y_series,
        c=[0, 0, 0],
        linestyle="--",
        linewidth=1.0,
        label="all poses",
        zorder=100,
    )

    if selected_ids:
        selected_x_series = []
        selected_y_series = []
        for i in selected_ids:
            xyz = pose_xyz(id_poses[i])
            selected_x_series.append(xyz[0])
            selected_y_series.append(xyz[1])
        plt.plot(
            selected_x_series,
            selected_y_series,
            c=[1, 0, 0],
            linewidth=3.0,
            label="visualized segment",
            zorder=200,
        )
        plt.scatter(
            selected_x_series,
            selected_y_series,
            s=42,
            facecolor="yellow",
            edgecolor="red",
            linewidth=1.5,
            zorder=250,
        )

    if ref_ids:
        ref_x_series = []
        ref_y_series = []
        for i in ref_ids:
            xyz = pose_xyz(id_poses[i])
            ref_x_series.append(xyz[0])
            ref_y_series.append(xyz[1])
        plt.scatter(
            ref_x_series,
            ref_y_series,
            s=80,
            marker="*",
            facecolor="deepskyblue",
            edgecolor="black",
            linewidth=0.8,
            label="frame_id +/- 10 refs",
            zorder=300,
        )

    if frame_id in id_poses:
        center_xyz = pose_xyz(id_poses[frame_id])
        plt.scatter(
            [center_xyz[0]],
            [center_xyz[1]],
            s=180,
            marker="*",
            facecolor="gold",
            edgecolor="black",
            linewidth=1.2,
            label=f"frame_id {frame_id}",
            zorder=350,
        )
        plt.annotate(
            str(frame_id),
            (center_xyz[0], center_xyz[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
        )

    plt.title("Trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    print(f"Saved trajectory overview to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mast3r_fusion visualization with options")

    parser.add_argument("--frame_id", type=int, default=240)
    parser.add_argument("--h5", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--pose_file", type=str, default =None)
    parser.add_argument("--traj_viz_output", type=str, default="check_h5_trajectory_window.png")
    parser.add_argument("--no_traj_viz", action="store_true")

    args = parser.parse_args()

    FRAME_ID = args.frame_id

    load_config(args.config)
    config["use_calib"] = True

    manager = mp.Manager()
    main2viz = new_queue(manager, False)
    viz2main = new_queue(manager, False)

    def load_frame_from_h5(h5_filename, iframe):
        with h5py.File(h5_filename, "r") as f:
            blob = bytes(f[f"frame_{iframe}"][()])
            buffer = io.BytesIO(blob)
            return torch.load(buffer, map_location="cpu")

    dataset = load_dataset("")
    dataset.subsample(2,0,999999)

    if config["use_calib"]:
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

    device = 'cuda'
    H5_FILE = args.h5

    id_poses = {}

    print('Scanning poses...')
    for i in tqdm.tqdm(range(len_h5(args.h5))):
        dd = load_frame_from_h5(H5_FILE, i)
        id_poses[i] = dd['T_WC'][-1,:]
        h,w = dd['uimg'].shape[:2]
    if not args.pose_file is None:
        id_poses = {}
        pppp = np.loadtxt(args.pose_file)
        for i in range(len(pppp)):
            id_poses[int(pppp[i,15])] = pppp[i,1:9]


    keyframes = SharedKeyframes(manager, h, w,buffer=1024)
    states = SharedStates(manager, h, w)
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
    keyframes.set_intrinsics(K)



    selected_ids = []
    for i in tqdm.tqdm(range(len_h5(args.h5))):
        if i not in id_poses:
            continue
        is_nearby=False
        for ii in range(FRAME_ID-10,FRAME_ID+10):
            if ii not in id_poses:
                continue
            if np.linalg.norm(pose_xyz(id_poses[i]) - pose_xyz(id_poses[ii])) < 30:
                is_nearby = True
        if not is_nearby:continue
        selected_ids.append(i)
        dd = load_frame_from_h5(H5_FILE, i)
        dd['T_WC'][-1,:] = torch.tensor(id_poses[i])
        dd['X'] *= dd['T_WC'][-1,-1]
        dd['X'] = dd['X'][None]
        dd['T_WC'][-1,-1] = 1.0

        dd['X'][0,dd['X'][0,:,2]>15.0,:] = 0.01
        frame = create_frame(i, dd['uimg'].astype(np.float32)/255.0, dd['T_WC'], img_size=dataset.img_size, device=device)
        frame.update_pointmap(dd['X'], dd['C']/dd['N'])
        frame.feat = 0
        frame.pos = 0
        frame.dataset_idx = i
        states.set_frame(frame)
        keyframes.append(frame)
        states.set_mode(Mode.TRACKING)

    if not args.no_traj_viz:
        save_trajectory_overview(id_poses, selected_ids, FRAME_ID, args.traj_viz_output)

    run_visualization(config, states, keyframes, main2viz, viz2main, max_show = 1000)




