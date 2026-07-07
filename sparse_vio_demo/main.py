from __future__ import annotations

import argparse
import os
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_vio_demo.sparse_vio.backend import SparseBackend
from sparse_vio_demo.sparse_vio.backend.sparse_ba import BackendConfig
from sparse_vio_demo.sparse_vio.dataset import ImageDataset, load_camera
from sparse_vio_demo.sparse_vio.frontend import VinsFrontend
from sparse_vio_demo.sparse_vio.frontend.vins_frontend import VinsFrontendConfig
from sparse_vio_demo.sparse_vio.imu import ImuBuffer
from sparse_vio_demo.sparse_vio.trajectory import save_tum_trajectory
from sparse_vio_demo.sparse_vio.viz import DebugVisualizer, FoxgloveConfig, FoxglovePublisher


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_frontend(camera, cfg):
    vins_cfg = VinsFrontendConfig(
        max_features=int(cfg.get("max_features", 250)),
        min_distance=int(cfg.get("min_distance", 30)),
        keyframe_gap=int(cfg.get("keyframe_gap", 5)),
        keyframe_parallax=float(cfg.get("keyframe_parallax", 20.0)),
        fb_thresh=float(cfg.get("fb_thresh", 0.5)),
        f_ransac_thresh=float(cfg.get("f_ransac_thresh", 1.0)),
        reject_with_f=bool(cfg.get("reject_with_f", False)),
    )
    return VinsFrontend(camera, vins_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sparse_vio_demo/configs/kitti360_sparse.yaml")
    parser.add_argument("--frontend", choices=["vins"], default="vins")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--foxglove", action="store_true")
    parser.add_argument("--foxglove-host", default=None)
    parser.add_argument("--foxglove-port", type=int, default=None)
    parser.add_argument("--imu-path", default=None)
    parser.add_argument("--imu-dt", type=float, default=None)
    parser.add_argument("--use-imu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--result-tum", default=None)
    parser.add_argument("--trajectory-frame", choices=["body", "camera"], default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    ds_cfg = cfg["dataset"]
    fe_cfg = cfg["frontend"]
    be_cfg = cfg["backend"]
    viz_cfg = cfg["viz"]

    camera = load_camera(ds_cfg["calib"])
    dataset = ImageDataset(
        ds_cfg["image_dir"],
        stamp_path=ds_cfg.get("stamp_path"),
        start=int(ds_cfg.get("start", 0) if args.start is None else args.start),
        end=int(ds_cfg.get("end", -1) if args.end is None else args.end),
        subsample=int(ds_cfg.get("subsample", 1) if args.subsample is None else args.subsample),
    )
    frontend_name = "vins"
    imu_cfg = cfg.get("imu", {})
    use_imu = bool(imu_cfg.get("use_imu", True) if args.use_imu is None else args.use_imu)
    imu_path = args.imu_path if args.imu_path is not None else imu_cfg.get("path")
    imu = None
    if use_imu and imu_path:
        imu = ImuBuffer(
            imu_path,
            dt=float(args.imu_dt if args.imu_dt is not None else imu_cfg.get("dt", 0.0)),
            gyro_unit=imu_cfg.get("gyro_unit", "deg"),
        )
    frontend = make_frontend(camera, fe_cfg)
    default_keyframe_only = False
    default_use_odom_prior = True
    backend = SparseBackend(
        camera,
        BackendConfig(
            window_size=int(be_cfg.get("window_size", 8)),
            min_triangulation_parallax=float(be_cfg.get("min_triangulation_parallax", 8.0)),
            max_ba_iters=int(be_cfg.get("max_ba_iters", 10)),
            max_ba_points=int(be_cfg.get("max_ba_points", 80)),
            robust_reprojection=bool(be_cfg.get("robust_reprojection", False)),
            vi_init_min_keyframes=int(be_cfg.get("vi_init_min_keyframes", 8)),
            vi_init_min_points=int(be_cfg.get("vi_init_min_points", 50)),
            vi_init_min_baseline=float(be_cfg.get("vi_init_min_baseline", 0.05)),
            vi_init_disable_scale=bool(be_cfg.get("vi_init_disable_scale", True)),
            vi_init_min_scale=float(be_cfg.get("vi_init_min_scale", 0.05)),
            vi_init_max_scale=float(be_cfg.get("vi_init_max_scale", 20.0)),
            vi_init_max_gyro_bias=float(be_cfg.get("vi_init_max_gyro_bias", 0.2)),
            accel_noise_sigma=float(be_cfg.get("accel_noise_sigma", 0.08)),
            gyro_noise_sigma=float(be_cfg.get("gyro_noise_sigma", 0.004)),
            accel_bias_rw_sigma=float(be_cfg.get("accel_bias_rw_sigma", 0.0004)),
            gyro_bias_rw_sigma=float(be_cfg.get("gyro_bias_rw_sigma", 0.00002)),
            marginal_pose_sigma=float(be_cfg.get("marginal_pose_sigma", 0.03)),
            marginal_velocity_sigma=float(be_cfg.get("marginal_velocity_sigma", 0.3)),
            marginal_bias_sigma=float(be_cfg.get("marginal_bias_sigma", 0.03)),
            marginal_point_sigma=float(be_cfg.get("marginal_point_sigma", 0.2)),
            final_global_ba=bool(be_cfg.get("final_global_ba", True)),
            max_global_ba_points=int(be_cfg.get("max_global_ba_points", 1600)),
            max_global_ba_iters=int(be_cfg.get("max_global_ba_iters", 8)),
            min_pnp_inliers=int(be_cfg.get("min_pnp_inliers", 30)),
            min_pnp_inlier_ratio=float(be_cfg.get("min_pnp_inlier_ratio", 0.2)),
            max_pnp_reproj_rmse=float(be_cfg.get("max_pnp_reproj_rmse", 3.5)),
            max_init_rotation_deg=float(be_cfg.get("max_init_rotation_deg", 12.0)),
            max_init_translation_factor=float(be_cfg.get("max_init_translation_factor", 4.0)),
            max_init_translation_abs=float(be_cfg.get("max_init_translation_abs", 8.0)),
            min_track_observations=int(be_cfg.get("min_track_observations", 3)),
            keyframe_only=bool(be_cfg.get("keyframe_only", default_keyframe_only)),
            use_odom_prior=bool(be_cfg.get("use_odom_prior", default_use_odom_prior)),
            vo_init_translation=float(be_cfg.get("vo_init_translation", 0.25)),
            vo_max_translation=float(be_cfg.get("vo_max_translation", 0.6)),
            vo_pose_prior_rot_sigma=float(be_cfg.get("vo_pose_prior_rot_sigma", 0.03)),
            vo_pose_prior_trans_sigma=float(be_cfg.get("vo_pose_prior_trans_sigma", 0.05)),
            odom_prior_rot_sigma=float(be_cfg.get("odom_prior_rot_sigma", 0.05)),
            odom_prior_trans_sigma=float(be_cfg.get("odom_prior_trans_sigma", 0.2)),
            vio_imu_mode=str(be_cfg.get("vio_imu_mode", "gyro")),
            gyro_factor_rot_sigma=float(be_cfg.get("gyro_factor_rot_sigma", 0.01)),
            gyro_factor_trans_sigma=float(be_cfg.get("gyro_factor_trans_sigma", 1.0e6)),
            vio_visual_pose_prior_rot_sigma=float(be_cfg.get("vio_visual_pose_prior_rot_sigma", 0.02)),
            vio_visual_pose_prior_trans_sigma=float(be_cfg.get("vio_visual_pose_prior_trans_sigma", 0.05)),
            vio_max_pose_update_trans=float(be_cfg.get("vio_max_pose_update_trans", 0.25)),
            vio_max_pose_update_rot_deg=float(be_cfg.get("vio_max_pose_update_rot_deg", 5.0)),
            vio_use_imu_rotation=bool(be_cfg.get("vio_use_imu_rotation", True)),
            fix_vo_poses=bool(be_cfg.get("fix_vo_poses", True)),
            fixed_pose_sigma=float(be_cfg.get("fixed_pose_sigma", 1e-6)),
            max_landmark_depth=float(be_cfg.get("max_landmark_depth", 80.0)),
            max_landmark_reproj_error=float(be_cfg.get("max_landmark_reproj_error", 4.0)),
        ),
        imu=imu,
    )
    visualizer = DebugVisualizer(
        viz_cfg.get("out_dir", "sparse_vio_demo/output/debug"),
        write_video=bool(viz_cfg.get("write_video", True)) and not args.no_video,
    )
    fg_cfg = viz_cfg.get("foxglove", {})
    foxglove = None
    if args.foxglove or bool(fg_cfg.get("enabled", False)):
        foxglove = FoxglovePublisher(
            camera,
            FoxgloveConfig(
                host=args.foxglove_host or fg_cfg.get("host", "127.0.0.1"),
                port=int(args.foxglove_port or fg_cfg.get("port", 8765)),
                jpeg_quality=int(fg_cfg.get("jpeg_quality", 85)),
                publish_every=int(fg_cfg.get("publish_every", 1)),
                max_points=int(fg_cfg.get("max_points", 3000)),
            ),
        )
        foxglove.start()

    timestamps = {}
    prev_packet = None
    for packet in dataset:
        timestamps[packet.idx] = packet.timestamp
        gyro_R = None
        if imu is not None and prev_packet is not None:
            gyro_R = imu.delta_rotation(prev_packet.timestamp, packet.timestamp)
        result = frontend.process(packet, gyro_R=gyro_R)
        prev_packet = packet
        state = backend.push(result)
        visualizer.draw(result, state)
        if foxglove is not None:
            foxglove.publish(result, state)
        if packet.idx % 20 == 0:
            print(
                f"[sparse_vio] frame={packet.idx} frontend={frontend_name} "
                f"use_imu={int(use_imu)} "
                f"tracks={len(result.tracks)} kf={int(result.new_keyframe)} "
                f"poses={len(state.poses)} points={len(state.points)}"
            )
    if hasattr(backend, "finalize"):
        backend.finalize()
    visualizer.save_trajectory(backend.state)
    result_tum = args.result_tum or viz_cfg.get("result_tum")
    if result_tum:
        trajectory_frame = args.trajectory_frame or viz_cfg.get("trajectory_frame", "body")
        save_tum_trajectory(result_tum, backend.state, timestamps, frame=trajectory_frame)
        print(f"[sparse_vio] saved TUM trajectory: {result_tum}")
    visualizer.close()
    if foxglove is not None:
        foxglove.close()
    print(
        f"[sparse_vio] done frontend={frontend_name} "
        f"use_imu={int(use_imu)} "
        f"poses={len(backend.state.poses)} points={len(backend.state.points)}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
