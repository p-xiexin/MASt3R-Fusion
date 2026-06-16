import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import mast3r_fusion.frontend_model.pi3_matching as pi3_matching
from mast3r_fusion.frontend_model.pi3x_utils import load_pi3x
from mast3r_fusion.mast3r_utils import _crop_resize


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether PI3X local pointmaps and camera poses can produce "
            "dense pair correspondences by reprojection."
        )
    )
    parser.add_argument("--image-a", required=True, help="First RGB image.")
    parser.add_argument("--image-b", required=True, help="Second RGB image.")
    parser.add_argument(
        "--weights",
        default="checkpoints/pi3x/model.safetensors",
        help="PI3X checkpoint or from_pretrained identifier.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Legacy square resize size. Prefer --target-size H W for the same "
            "crop/resize pipeline used by MASt3R-Fusion."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(224, 840),
        help=(
            "Target image size [H W]. Uses MASt3R-Fusion _crop_resize(), and "
            "both dimensions must be divisible by PI3X patch size 14."
        ),
    )
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument(
        "--match-grid-size",
        type=int,
        default=32,
        help="Source-grid cell size used to keep spatially balanced PI3X matches.",
    )
    parser.add_argument(
        "--match-max-per-cell",
        type=int,
        default=32,
        help="Maximum high-confidence matches kept per source-grid cell.",
    )
    parser.add_argument(
        "--no-unique-target",
        dest="unique_target",
        action="store_false",
        help="Allow multiple source pixels to match the same target pixel.",
    )
    parser.set_defaults(unique_target=True)
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="pi3x_reprojection_verify")
    return parser.parse_args()


def load_rgb(path: str, target_size, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"))
    processed = _crop_resize(image, target_size)
    array = processed["unnormalized_img"].astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return tensor.contiguous()


def save_rgb(path: Path, image: torch.Tensor):
    array = image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


def to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).astype(np.uint8)


def pick_output(output, names):
    for name in names:
        if name in output:
            return output[name]
    raise KeyError(f"PI3X output is missing one of: {names}")


def normalize_output(output):
    local_points = pick_output(output, ("local_points", "points_local", "pts3d"))
    conf = pick_output(output, ("conf", "confidence", "confidences"))
    poses = output.get("camera_poses")
    if poses is None:
        poses = output.get("poses")
    if poses is None:
        poses = output.get("extrinsics")
    if poses is None:
        raise KeyError("PI3X output is missing camera poses.")

    if local_points.ndim == 4:
        local_points = local_points.unsqueeze(0)
    if conf.ndim == 4:
        conf = conf.unsqueeze(0)
    if conf.shape[-1] == 1:
        conf = conf[..., 0]
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)

    return local_points[0].float(), torch.sigmoid(conf[0].float()), poses[0].float()


def pixel_grid(h: int, w: int, device: torch.device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return x, y


def fit_intrinsics(points: torch.Tensor, conf: torch.Tensor, conf_threshold: float, min_depth: float):
    h, w = points.shape[:2]
    x_grid, y_grid = pixel_grid(h, w, points.device)
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > min_depth) & (conf > conf_threshold)
    if valid.sum() < 32:
        raise ValueError(f"Not enough valid points to fit pseudo intrinsics: {int(valid.sum())}.")

    xn = points[..., 0][valid] / z[valid]
    yn = points[..., 1][valid] / z[valid]
    u = x_grid[valid]
    v = y_grid[valid]

    ones = torch.ones_like(xn)
    Ax = torch.stack((xn, ones), dim=-1)
    Ay = torch.stack((yn, ones), dim=-1)
    fx_cx = torch.linalg.lstsq(Ax, u[:, None]).solution[:, 0]
    fy_cy = torch.linalg.lstsq(Ay, v[:, None]).solution[:, 0]

    fx, cx = fx_cx[0], fx_cx[1]
    fy, cy = fy_cy[0], fy_cy[1]
    u_hat = fx * xn + cx
    v_hat = fy * yn + cy
    err = torch.sqrt((u_hat - u).square() + (v_hat - v).square())
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=points.device,
        dtype=points.dtype,
    )
    return K, valid, err


def draw_matches(
    path: Path,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    valid: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    max_lines: int,
    seed: int,
    match_shape,
):
    a = to_uint8_rgb(image_a)
    b = to_uint8_rgb(image_b)
    h, w = a.shape[:2]
    match_h, match_w = match_shape
    canvas = np.concatenate((a, b), axis=1)
    overlay = canvas.copy()

    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).cpu().numpy()
    if valid_indices.size == 0:
        Image.fromarray(canvas).save(path)
        return
    rng = np.random.default_rng(seed)
    if valid_indices.size > max_lines:
        valid_indices = rng.choice(valid_indices, size=max_lines, replace=False)
    for idx in valid_indices:
        y0_match, x0_match = divmod(int(idx), match_w)
        x0 = x0_match
        y0 = y0_match
        x1 = int(round(float(u[idx].detach().cpu()))) + w
        y1 = int(round(float(v[idx].detach().cpu())))
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, w, 2 * w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        color = (
            int(255 * x0 / max(w - 1, 1)),
            int(255 * y0 / max(h - 1, 1)),
            255 - int(255 * x0 / max(w - 1, 1)),
        )
        cv2.line(overlay, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)

    canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0)
    Image.fromarray(canvas).save(path)


def print_stats(name: str, idx: torch.Tensor, valid: torch.Tensor, h: int, w: int, debug):
    valid_count = int(valid.sum().item())
    total = valid.numel()
    valid_before_balance = debug.get("valid_before_balance")
    if valid_before_balance is not None:
        raw_count = int(valid_before_balance[0].sum().item())
        print(
            f"{name}.balanced_ratio={valid_count / max(raw_count, 1):.4f} "
            f"({valid_count}/{raw_count})"
        )
    unique = int(torch.unique(idx[valid]).numel()) if valid_count else 0
    print(f"{name}.valid_ratio={valid_count / total:.4f} ({valid_count}/{total})")
    print(f"{name}.unique_target_ratio={unique / max(valid_count, 1):.4f} ({unique}/{valid_count})")
    if valid_count:
        ys = (idx[valid] // w).float()
        xs = (idx[valid] % w).float()
        print(
            f"{name}.target_bounds="
            f"x[{xs.min().item():.1f},{xs.max().item():.1f}] "
            f"y[{ys.min().item():.1f},{ys.max().item():.1f}]"
        )


@torch.inference_mode()
def main():
    args = parse_args()
    target_size = (args.size, args.size) if args.size is not None else tuple(args.target_size)
    if target_size[0] % 14 != 0 or target_size[1] % 14 != 0:
        raise ValueError("--target-size/--size dimensions must be divisible by 14 for PI3X.")
    if args.match_grid_size < 0:
        raise ValueError("--match-grid-size must be >= 0.")
    if args.match_max_per_cell < 0:
        raise ValueError("--match-max-per-cell must be >= 0.")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_a = load_rgb(args.image_a, target_size, device)
    image_b = load_rgb(args.image_b, target_size, device)
    print(f"target_size={target_size}")

    model = load_pi3x(args.weights, device=str(device))
    images = torch.stack((image_a, image_b), dim=0).unsqueeze(0)
    output = model(imgs=images)
    points, conf, poses = normalize_output(output)
    K_a = pi3_matching.fit_intrinsics(
        points[:1], conf[:1], args.conf_threshold, args.min_depth
    )[0]
    K_b = pi3_matching.fit_intrinsics(
        points[1:2], conf[1:2], args.conf_threshold, args.min_depth
    )[0]
    _, fit_valid_a_full, fit_err_a_full = fit_intrinsics(
        points[0], conf[0], args.conf_threshold, args.min_depth
    )
    _, fit_valid_b_full, fit_err_b_full = fit_intrinsics(
        points[1], conf[1], args.conf_threshold, args.min_depth
    )

    h, w = points.shape[1:3]
    points_a, points_b = points[0], points[1]
    conf_a, conf_b = conf[0], conf[1]
    image_a, image_b = images[0, 0], images[0, 1]
    pose_a, pose_b = poses[0], poses[1]
    save_rgb(output_dir / "image_a.png", image_a)
    save_rgb(output_dir / "image_b.png", image_b)

    print(f"K_a=\n{K_a.detach().cpu().numpy()}")
    print(f"K_b=\n{K_b.detach().cpu().numpy()}")
    print(
        "fit_error_px: "
        f"a_median={fit_err_a_full.median().item():.3f}, "
        f"a_p95={fit_err_a_full.quantile(0.95).item():.3f}, "
        f"b_median={fit_err_b_full.median().item():.3f}, "
        f"b_p95={fit_err_b_full.quantile(0.95).item():.3f}"
    )
    print(
        "confidence: "
        f"a[min={conf_a.min().item():.3f}, max={conf_a.max().item():.3f}, mean={conf_a.mean().item():.3f}], "
        f"b[min={conf_b.min().item():.3f}, max={conf_b.max().item():.3f}, mean={conf_b.mean().item():.3f}]"
    )

    idx_a2b, valid_a2b, pair_conf_a2b, debug_a2b = pi3_matching.match(
        points_a[None],
        points_b[None],
        pose_a[None],
        pose_b[None],
        conf_a[None],
        conf_b[None],
        K_dst=K_b,
        conf_threshold=args.conf_threshold,
        min_depth=args.min_depth,
        balance_grid_size=args.match_grid_size,
        max_matches_per_cell=args.match_max_per_cell,
        unique_target=args.unique_target,
        return_debug=True,
    )
    idx_b2a, valid_b2a, pair_conf_b2a, debug_b2a = pi3_matching.match(
        points_b[None],
        points_a[None],
        pose_b[None],
        pose_a[None],
        conf_b[None],
        conf_a[None],
        K_dst=K_a,
        conf_threshold=args.conf_threshold,
        min_depth=args.min_depth,
        balance_grid_size=args.match_grid_size,
        max_matches_per_cell=args.match_max_per_cell,
        unique_target=args.unique_target,
        return_debug=True,
    )
    idx_a2b = idx_a2b[0]
    idx_b2a = idx_b2a[0]
    valid_a2b = valid_a2b[0, :, 0]
    valid_b2a = valid_b2a[0, :, 0]
    pair_conf_a2b = pair_conf_a2b[0, :, 0]
    pair_conf_b2a = pair_conf_b2a[0, :, 0]
    u_a2b = debug_a2b["u"][0]
    v_a2b = debug_a2b["v"][0]
    u_b2a = debug_b2a["u"][0]
    v_b2a = debug_b2a["v"][0]

    print(
        "match_filter: "
        f"grid_size={args.match_grid_size}, "
        f"max_per_cell={args.match_max_per_cell}, "
        f"unique_target={args.unique_target}"
    )
    print_stats("a2b", idx_a2b, valid_a2b, h, w, debug_a2b)
    print_stats("b2a", idx_b2a, valid_b2a, h, w, debug_b2a)
    if valid_a2b.any():
        print(
            "a2b.pair_conf: "
            f"median={pair_conf_a2b[valid_a2b].median().item():.3f}, "
            f"mean={pair_conf_a2b[valid_a2b].mean().item():.3f}"
        )
    if valid_b2a.any():
        print(
            "b2a.pair_conf: "
            f"median={pair_conf_b2a[valid_b2a].median().item():.3f}, "
            f"mean={pair_conf_b2a[valid_b2a].mean().item():.3f}"
        )

    draw_matches(
        output_dir / "a_to_b_reprojection.png",
        image_a,
        image_b,
        valid_a2b,
        u_a2b,
        v_a2b,
        args.max_lines,
        args.seed,
        (h, w),
    )
    draw_matches(
        output_dir / "b_to_a_reprojection.png",
        image_b,
        image_a,
        valid_b2a,
        u_b2a,
        v_b2a,
        args.max_lines,
        args.seed,
        (h, w),
    )

    np.savez_compressed(
        output_dir / "reprojection_correspondences.npz",
        idx_a2b=idx_a2b.detach().cpu().numpy(),
        valid_a2b=valid_a2b.detach().cpu().numpy(),
        idx_b2a=idx_b2a.detach().cpu().numpy(),
        valid_b2a=valid_b2a.detach().cpu().numpy(),
        pair_conf_a2b=pair_conf_a2b.detach().cpu().numpy(),
        pair_conf_b2a=pair_conf_b2a.detach().cpu().numpy(),
        K_a=K_a.detach().cpu().numpy(),
        K_b=K_b.detach().cpu().numpy(),
        match_grid_size=args.match_grid_size,
        match_max_per_cell=args.match_max_per_cell,
        unique_target=args.unique_target,
        fit_valid_a=fit_valid_a_full.detach().cpu().numpy(),
        fit_valid_b=fit_valid_b_full.detach().cpu().numpy(),
    )
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
