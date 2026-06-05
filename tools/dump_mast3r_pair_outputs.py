import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRDPARTY_MAST3R = REPO_ROOT / "thirdparty" / "mast3r"
if str(THIRDPARTY_MAST3R) not in sys.path:
    sys.path.insert(0, str(THIRDPARTY_MAST3R))


torch = None
inference = None
load_images = None
fast_reciprocal_NNs = None
load_mast3r = None


def _load_runtime_imports():
    global torch
    global inference
    global load_images
    global fast_reciprocal_NNs
    global load_mast3r

    import torch as torch_module

    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.inference import inference as dust3r_inference
    from dust3r.utils.image import load_images as dust3r_load_images
    from mast3r.fast_nn import fast_reciprocal_NNs as mast3r_fast_reciprocal_NNs
    from mast3r_fusion.mast3r_utils import load_mast3r as repo_load_mast3r

    torch = torch_module
    inference = dust3r_inference
    load_images = dust3r_load_images
    fast_reciprocal_NNs = mast3r_fast_reciprocal_NNs
    load_mast3r = repo_load_mast3r


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _tensor_stats(value):
    tensor = value.detach().float().cpu()
    finite = torch.isfinite(tensor)
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(value.dtype),
        "finite_ratio": float(finite.float().mean().item()) if tensor.numel() else 1.0,
    }
    if finite.any():
        valid = tensor[finite]
        stats.update(
            {
                "min": float(valid.min().item()),
                "max": float(valid.max().item()),
                "mean": float(valid.mean().item()),
            }
        )
    return stats


def _summarize_mapping(mapping):
    summary = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            summary[key] = _tensor_stats(value)
        elif isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "value": _jsonable(value) if value.size <= 16 else None,
            }
        else:
            summary[key] = _jsonable(value)
    return summary


def _save_rgb(path, image):
    image = np.clip(image, 0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def _view_image(view):
    h, w = _true_shape(view)
    image = view["img"][0].detach().float().cpu()
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image[:h, :w]


def _save_scalar_image(path, values, cmap="turbo"):
    values = torch.as_tensor(values).detach().float().cpu()
    finite = torch.isfinite(values)
    if finite.any():
        lo = values[finite].quantile(0.02)
        hi = values[finite].quantile(0.98)
        values = (values - lo) / (hi - lo + 1e-8)
    else:
        values = torch.zeros_like(values)
    values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = values.clamp(0, 1).numpy()

    try:
        import matplotlib.cm as cm

        colored = (cm.get_cmap(cmap)(values)[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = (values * 255).astype(np.uint8)
        colored = np.repeat(gray[..., None], 3, axis=-1)
    Image.fromarray(colored).save(path)


def _write_ply(path, points, colors=None, conf=None, max_points=200000):
    points = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
    valid = torch.isfinite(points).all(dim=1) & (points[:, 2] > 0)
    if conf is not None:
        conf = torch.as_tensor(conf).detach().float().cpu().reshape(-1)
        valid &= torch.isfinite(conf)
    points = points[valid]

    if colors is not None:
        colors = torch.as_tensor(colors).detach().float().cpu().reshape(-1, 3)
        colors = colors[valid].clamp(0, 1)
    else:
        colors = torch.ones_like(points)

    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]
        colors = colors[::step]

    xyz = points.numpy()
    rgb_values = (colors * 255).to(torch.uint8).numpy()
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, rgb_values):
            f.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _true_shape(view):
    shape = _as_numpy(view["true_shape"])[0]
    return int(shape[0]), int(shape[1])


def _crop_to_true_shape(value, true_shape):
    h, w = true_shape
    return value[:h, :w]


def _save_prediction_outputs(out_dir, name, view, pred, max_points):
    true_shape = _true_shape(view)
    image = _view_image(view)
    _save_rgb(out_dir / f"{name}_input.png", image)

    pts_key = "pts3d" if "pts3d" in pred else "pts3d_in_other_view"
    pts3d = _crop_to_true_shape(pred[pts_key][0], true_shape)
    conf = _crop_to_true_shape(pred["conf"][0], true_shape)
    _save_scalar_image(out_dir / f"{name}_{pts_key}_z.png", pts3d[..., 2])
    _save_scalar_image(out_dir / f"{name}_conf.png", conf, cmap="viridis")
    _write_ply(
        out_dir / f"{name}_{pts_key}.ply",
        pts3d,
        colors=image,
        conf=conf,
        max_points=max_points,
    )

    if "desc_conf" in pred:
        desc_conf = _crop_to_true_shape(pred["desc_conf"][0], true_shape)
        _save_scalar_image(out_dir / f"{name}_desc_conf.png", desc_conf, cmap="viridis")


def _filter_border_matches(matches0, matches1, shape0, shape1, border):
    h0, w0 = shape0
    h1, w1 = shape1
    valid0 = (
        (matches0[:, 0] >= border)
        & (matches0[:, 0] < w0 - border)
        & (matches0[:, 1] >= border)
        & (matches0[:, 1] < h0 - border)
    )
    valid1 = (
        (matches1[:, 0] >= border)
        & (matches1[:, 0] < w1 - border)
        & (matches1[:, 1] >= border)
        & (matches1[:, 1] < h1 - border)
    )
    valid = valid0 & valid1
    return matches0[valid], matches1[valid]


def _save_matches_csv(path, matches0, matches1):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "y0", "x1", "y1"])
        for (x0, y0), (x1, y1) in zip(matches0, matches1):
            writer.writerow([int(x0), int(y0), int(x1), int(y1)])


def _save_match_visualization(path, view1, view2, matches0, matches1, n_viz):
    if matches0.shape[0] == 0:
        return

    import matplotlib.pyplot as plt

    img0 = _view_image(view1)
    img1 = _view_image(view2)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    img0 = np.pad(img0, ((0, max(h1 - h0, 0)), (0, 0), (0, 0)))
    img1 = np.pad(img1, ((0, max(h0 - h1, 0)), (0, 0), (0, 0)))
    canvas = np.concatenate((img0, img1), axis=1)

    count = min(n_viz, matches0.shape[0])
    indices = np.round(np.linspace(0, matches0.shape[0] - 1, count)).astype(int)
    cmap = plt.get_cmap("jet")

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)
    for i, idx in enumerate(indices):
        x0, y0 = matches0[idx]
        x1, y1 = matches1[idx]
        denom = max(count - 1, 1)
        plt.plot([x0, x1 + w0], [y0, y1], "-+", color=cmap(i / denom), linewidth=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=160)
    plt.close()


def _save_npz(path, view1, view2, pred1, pred2, matches0, matches1, save_desc):
    data = {
        "view1_img": _as_numpy(view1["img"]),
        "view2_img": _as_numpy(view2["img"]),
        "view1_true_shape": _as_numpy(view1["true_shape"]),
        "view2_true_shape": _as_numpy(view2["true_shape"]),
        "pred1_pts3d": _as_numpy(pred1["pts3d"]),
        "pred1_conf": _as_numpy(pred1["conf"]),
        "pred2_conf": _as_numpy(pred2["conf"]),
        "matches_im0": matches0,
        "matches_im1": matches1,
    }
    pred2_pts_key = "pts3d_in_other_view" if "pts3d_in_other_view" in pred2 else "pts3d"
    data[f"pred2_{pred2_pts_key}"] = _as_numpy(pred2[pred2_pts_key])
    if "desc_conf" in pred1:
        data["pred1_desc_conf"] = _as_numpy(pred1["desc_conf"])
    if "desc_conf" in pred2:
        data["pred2_desc_conf"] = _as_numpy(pred2["desc_conf"])
    if save_desc:
        data["pred1_desc"] = _as_numpy(pred1["desc"])
        data["pred2_desc"] = _as_numpy(pred2["desc"])
    np.savez_compressed(path, **data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the official MASt3R two-image demo path and save raw outputs."
    )
    parser.add_argument("image_a", help="First input image.")
    parser.add_argument("image_b", help="Second input image.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Local checkpoint path or Hugging Face model name. Defaults to mast3r_utils.load_mast3r().",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--image-size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="mast3r_pair_outputs")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--match-border", type=int, default=3)
    parser.add_argument("--match-block-size", type=int, default=2**13)
    parser.add_argument("--num-viz-matches", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument(
        "--save-desc",
        action="store_true",
        help="Also save dense descriptor tensors in outputs.npz.",
    )
    parser.add_argument(
        "--no-raw-pt",
        action="store_true",
        help="Skip saving the raw torch inference dictionary.",
    )
    return parser.parse_args()


def dump_pair_outputs(model, image_a, image_b, out_dir, args, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_images([str(image_a), str(image_b)], size=args.image_size, verbose=verbose)
    output = inference(
        [tuple(images)],
        model,
        args.device,
        batch_size=args.batch_size,
        verbose=verbose,
    )
    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()
    matches0, matches1 = fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=args.subsample,
        device=args.device,
        dist="dot",
        block_size=args.match_block_size,
    )
    raw_match_count = int(matches0.shape[0])
    matches0, matches1 = _filter_border_matches(
        matches0,
        matches1,
        _true_shape(view1),
        _true_shape(view2),
        args.match_border,
    )

    _save_prediction_outputs(out_dir, "view1_pred1", view1, pred1, args.max_points)
    _save_prediction_outputs(out_dir, "view2_pred2", view2, pred2, args.max_points)
    np.save(out_dir / "matches_im0.npy", matches0)
    np.save(out_dir / "matches_im1.npy", matches1)
    _save_matches_csv(out_dir / "matches.csv", matches0, matches1)
    _save_match_visualization(
        out_dir / "matches.png",
        view1,
        view2,
        matches0,
        matches1,
        args.num_viz_matches,
    )
    _save_npz(
        out_dir / "outputs.npz",
        view1,
        view2,
        pred1,
        pred2,
        matches0,
        matches1,
        args.save_desc,
    )
    if not args.no_raw_pt:
        torch.save(output, out_dir / "raw_output.pt")

    summary = {
        "image_a": str(Path(image_a).resolve()),
        "image_b": str(Path(image_b).resolve()),
        "weights": args.weights or "<mast3r_utils default>",
        "device": args.device,
        "image_size": args.image_size,
        "view1": _summarize_mapping(view1),
        "view2": _summarize_mapping(view2),
        "pred1": _summarize_mapping(pred1),
        "pred2": _summarize_mapping(pred2),
        "raw_match_count": raw_match_count,
        "valid_match_count": int(matches0.shape[0]),
        "match_border": args.match_border,
        "saved_dense_descriptors": bool(args.save_desc),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"Saved MASt3R pair outputs to {out_dir.resolve()}")
        print(f"raw matches: {raw_match_count}")
        print(f"valid matches after border filtering: {matches0.shape[0]}")
        for path in sorted(out_dir.iterdir()):
            print(f"  {path}")

    return summary


def main():
    args = parse_args()
    _load_runtime_imports()

    model = load_mast3r(args.weights, device=args.device)
    model.eval()
    dump_pair_outputs(model, args.image_a, args.image_b, args.output_dir, args)


if __name__ == "__main__":
    main()
