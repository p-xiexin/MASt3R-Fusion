from pathlib import Path
import types

import einops
import torch
import torch.nn.functional as F

import mast3r_fusion.frontend_model.pi3_matching as pi3_matching
from mast3r_fusion.config import config


DEFAULT_PI3X_WEIGHTS = "checkpoints/pi3x/model.safetensors"


def _patch_pi3x_rope_contiguous(model):
    for module in model.modules():
        if module.__class__.__name__ == "cuRoPE2D" and hasattr(module, "base"):
            forward_func = getattr(module.forward, "__func__", module.forward)
            rope_func = getattr(forward_func, "__globals__", {}).get("cuRoPE2D_func")
            if rope_func is None:
                continue

            def contiguous_rope_forward(self, tokens, positions, _rope_func=rope_func):
                tokens_t = tokens.transpose(1, 2).contiguous()
                _rope_func.apply(tokens_t, positions, self.base, self.F0)
                return tokens_t.transpose(1, 2).contiguous()

            module.forward = types.MethodType(contiguous_rope_forward, module)
    return model


def load_pi3x(path=None, device="cuda"):
    weights_path = DEFAULT_PI3X_WEIGHTS if path is None else path
    try:
        from pi3.models.pi3x import Pi3X
    except ImportError as exc:
        raise ImportError(
            "PI3X is not installed. Install the PI3X package and pass "
            "--frontend-model pi3x --frontend-weights <path>."
        ) from exc

    weights_path_obj = Path(weights_path)
    if weights_path_obj.suffix == ".safetensors" and not weights_path_obj.exists():
        raise FileNotFoundError(
            f"PI3X checkpoint not found: {weights_path}. Download it to "
            "checkpoints/pi3x/model.safetensors or pass --frontend-weights."
        )
    if hasattr(Pi3X, "from_pretrained") and (
        not weights_path_obj.exists() or weights_path_obj.is_dir()
    ):
        model = Pi3X.from_pretrained(weights_path).eval()
        if hasattr(model, "disable_multimodal"):
            model.disable_multimodal()
        model = _patch_pi3x_rope_contiguous(model)
        return model.to(device)

    model = Pi3X()
    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weights_path, device="cpu")
    else:
        state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if hasattr(model, "disable_multimodal"):
        model.disable_multimodal()
    model = _patch_pi3x_rope_contiguous(model)
    return model.to(device)


def _frame_image(frame):
    image = frame.uimg.to(device=frame.img.device, dtype=frame.img.dtype)
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def encode_frame_image(frame):
    image = _frame_image(frame)
    h, w = image.shape[-2:]
    feat = image.permute(0, 2, 3, 1).reshape(1, h * w, 3).contiguous()
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing="ij",
    )
    pos = torch.stack((xs, ys), dim=-1).view(1, h * w, 2).long()
    frame.feat = feat
    frame.pos = pos
    return frame.feat, frame.pos


def _features_to_images(feat, shapes, pos=None):
    images = []
    for b in range(feat.shape[0]):
        shape = shapes[b]
        if isinstance(shape, torch.Tensor):
            h, w = int(shape.reshape(-1)[0].item()), int(shape.reshape(-1)[1].item())
        else:
            h, w = int(shape[0]), int(shape[1])
        if h * w != feat.shape[1] and pos is not None:
            w = int(pos[b, :, 0].max().item()) + 1
            h = int(pos[b, :, 1].max().item()) + 1
        if h * w != feat.shape[1]:
            side = int(feat.shape[1] ** 0.5)
            if side * side == feat.shape[1]:
                h, w = side, side
            else:
                raise ValueError(
                    f"Cannot reshape PI3X image features of length {feat.shape[1]} "
                    f"to requested shape {(h, w)}."
                )
        images.append(feat[b].reshape(h, w, 3).permute(2, 0, 1))
    return torch.stack(images, dim=0).contiguous()


def _call_pi3x(model, images):
    h, w = images.shape[-2:]
    patch_size = getattr(model, "patch_size", 14)
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"PI3X expects image height/width divisible by {patch_size}, "
            f"got {(h, w)}. Set dataset.target_img_size to multiples of {patch_size}."
        )
    try:
        output = model(imgs=images)
    except TypeError:
        output = model(images)
    if not isinstance(output, dict):
        raise TypeError("PI3X inference must return a dict-like output.")
    return output


def _pick_output(output, names):
    for name in names:
        if name in output:
            return output[name]
    raise KeyError(f"PI3X output is missing one of: {names}")


def _resize_map(value, h, w, is_descriptor=False):
    if value.shape[-3:-1] == (h, w):
        return value.contiguous()
    channels_last = value.permute(0, 3, 1, 2)
    mode = "bilinear" if is_descriptor else "nearest"
    resized = F.interpolate(channels_last, size=(h, w), mode=mode)
    return resized.permute(0, 2, 3, 1).contiguous()


def _homogeneous_transform(points, transform):
    h, w = points.shape[-3:-1]
    points_flat = points.reshape(-1, 3)
    ones = torch.ones(points_flat.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat((points_flat, ones), dim=-1)
    transformed = (transform @ points_h.T).T[..., :3]
    return transformed.reshape(h, w, 3)


def _pose_to_matrix(pose, device, dtype):
    if pose is None:
        return None
    if hasattr(pose, "matrix"):
        matrix = pose.matrix()
    else:
        matrix = torch.as_tensor(pose)
    matrix = matrix.to(device=device, dtype=dtype)
    if matrix.ndim == 3:
        matrix = matrix[0]
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected relative pose matrix shape (4,4), got {tuple(matrix.shape)}.")
    return matrix


def _pair_output_to_maps(output, images):
    local_points = _pick_output(output, ("local_points", "points_local", "pts3d"))
    confidences = _pick_output(output, ("conf", "confidence", "confidences"))
    poses = output.get("camera_poses")
    if poses is None:
        poses = output.get("poses")
    if poses is None:
        poses = output.get("extrinsics")

    if local_points.ndim == 4:
        local_points = local_points.unsqueeze(0)
    if confidences.ndim == 4:
        confidences = confidences.unsqueeze(0)
    if confidences.shape[-1] != 1:
        confidences = confidences.unsqueeze(-1)

    h, w = images.shape[-2:]
    local_points = _resize_map(local_points[0], h, w)
    confidences = _resize_map(confidences[0], h, w)[..., 0]
    # Pi3X documents `conf` as raw logits; convert it before thresholding.
    confidences = torch.sigmoid(confidences)

    Xii = local_points[0]
    Xjj = local_points[1]
    T_w_ci = None
    T_w_cj = None
    if poses is not None:
        poses = torch.as_tensor(poses, device=local_points.device, dtype=local_points.dtype)
        if poses.ndim == 3:
            poses = poses.unsqueeze(0)
        T_w_ci = poses[0, 0]
        T_w_cj = poses[0, 1]
        T_ci_cj = torch.linalg.inv(T_w_ci) @ T_w_cj
        T_cj_ci = torch.linalg.inv(T_w_cj) @ T_w_ci
        Xji = _homogeneous_transform(Xjj, T_ci_cj)
        Xij = _homogeneous_transform(Xii, T_cj_ci)
    else:
        # PI3X builds without relative camera poses cannot produce cross-view
        # pointmaps. Reuse local maps as a degraded pairwise fallback.
        Xji = Xjj
        Xij = Xii

    # PI3X does not expose MASt3R-style dense descriptors. Keep empty
    # placeholders for debug/decode APIs; matching uses geometry only.
    Dii = Xii.new_empty((*Xii.shape[:2], 0))
    Djj = Xjj.new_empty((*Xjj.shape[:2], 0))

    return Xii, Xji, Xjj, Xij, confidences[0], confidences[1], Dii, Djj, T_w_ci, T_w_cj


@torch.inference_mode()
def pi3x_inference_pair(model, frame_i, frame_j):
    encode_frame_image(frame_i)
    encode_frame_image(frame_j)
    images = torch.cat((_frame_image(frame_i), _frame_image(frame_j)), dim=0)
    output = _call_pi3x(model, images.unsqueeze(0))
    Xii, Xji, _, _, Cii, Cji, Dii, Dji, _, _ = _pair_output_to_maps(output, images.unsqueeze(0))
    X, C, D, Q = torch.stack((Xii, Xji)), torch.stack((Cii, Cji)), torch.stack((Dii, Dji)), torch.stack((Cii, Cji))
    return _downsample(X, C, D, Q)


@torch.inference_mode()
def pi3x_inference_mono(model, frame):
    X, C, _, _ = pi3x_inference_pair(model, frame, frame)
    Xii, _ = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, _ = einops.rearrange(C, "b h w -> b (h w) 1")
    return Xii, Cii


def _downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


def _match_conf_threshold():
    return config.get("pi3x", {}).get("match_conf_threshold", 0.0)


def _match_debug_enabled():
    return config.get("pi3x", {}).get("debug_match", True)


def _ratio(mask):
    return float(mask.float().mean().item()) if mask.numel() else 0.0


def _print_match_debug(name, frame_i, frame_j, valid, pair_conf, conf_src, conf_dst, debug, pose_source):
    valid_flat = valid.reshape(valid.shape[0], -1)
    pair_flat = pair_conf.reshape(pair_conf.shape[0], -1)
    for batch_idx in range(valid_flat.shape[0]):
        valid_b = valid_flat[batch_idx]
        pair_b = pair_flat[batch_idx]
        valid_count = int(valid_b.sum().item())
        total = valid_b.numel()
        if valid_count:
            pair_valid = pair_b[valid_b]
            pair_summary = (
                f"pair_conf[min={pair_valid.min().item():.3f}, "
                f"mean={pair_valid.mean().item():.3f}, "
                f"max={pair_valid.max().item():.3f}]"
            )
        else:
            pair_summary = "pair_conf[empty]"
        z = debug["z"][batch_idx]
        print(
            f"[PI3X match:{name}] "
            f"frame_i={getattr(frame_i, 'frame_id', '?')} "
            f"frame_j={getattr(frame_j, 'frame_id', '?')} "
            f"pose={pose_source} "
            f"valid={valid_count}/{total} ({valid_count / max(total, 1):.6f}) "
            f"in_bounds={_ratio(debug['in_bounds'][batch_idx]):.6f} "
            f"depth={_ratio(debug['positive_depth'][batch_idx]):.6f} "
            f"src_conf={_ratio(debug['valid_conf_src'][batch_idx]):.6f} "
            f"dst_conf={_ratio(debug['valid_conf_dst'][batch_idx]):.6f} "
            f"z[min={torch.nan_to_num(z).min().item():.3g}, "
            f"max={torch.nan_to_num(z).max().item():.3g}] "
            f"Cii[min={conf_src[batch_idx].min().item():.3f}, "
            f"mean={conf_src[batch_idx].mean().item():.3f}, "
            f"max={conf_src[batch_idx].max().item():.3f}] "
            f"Cjj[min={conf_dst[batch_idx].min().item():.3f}, "
            f"mean={conf_dst[batch_idx].mean().item():.3f}, "
            f"max={conf_dst[batch_idx].max().item():.3f}] "
            f"{pair_summary}"
        )


def pi3x_decoder(*args, **kwargs):
    # PI3X does not expose MASt3R's private _decoder/_downstream_head API.
    # Pair inference must go through pi3x_inference_pair instead.
    raise NotImplementedError("PI3X does not provide a MASt3R-compatible decoder.")


def pi3x_decode_symmetric_batch(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = [], [], [], []
    images_i = _features_to_images(feat_i, shape_i, pos_i)
    images_j = _features_to_images(feat_j, shape_j, pos_j)
    for b in range(feat_i.shape[0]):
        images = torch.stack((images_i[b], images_j[b]), dim=0).unsqueeze(0)
        output = _call_pi3x(model, images)
        Xii, Xji, Xjj, Xij, Cii, Cjj, Dii, Djj, _, _ = _pair_output_to_maps(output, images)
        X.append(torch.stack((Xii, Xji, Xjj, Xij)))
        C.append(torch.stack((Cii, Cjj, Cjj, Cii)))
        D.append(torch.stack((Dii, Djj, Djj, Dii)))
        Q.append(torch.stack((Cii, Cjj, Cjj, Cii)))

    X = torch.stack(X, dim=1)
    C = torch.stack(C, dim=1)
    D = torch.stack(D, dim=1)
    Q = torch.stack(Q, dim=1)
    return _downsample(X, C, D, Q)


def pi3x_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, subpixel_factor=1):
    X, C, Q = [], [], []
    pose_i, pose_j = [], []
    images_i = _features_to_images(feat_i, shape_i, pos_i)
    images_j = _features_to_images(feat_j, shape_j, pos_j)
    for batch_idx in range(feat_i.shape[0]):
        images = torch.stack((images_i[batch_idx], images_j[batch_idx]), dim=0).unsqueeze(0)
        output = _call_pi3x(model, images)
        Xii, Xji, Xjj, Xij, Cii, Cjj, _, _, T_w_ci, T_w_cj = _pair_output_to_maps(output, images)
        if T_w_ci is None or T_w_cj is None:
            raise KeyError("PI3X geometry matching requires camera poses.")
        X.append(torch.stack((Xii, Xji, Xjj, Xij)))
        C.append(torch.stack((Cii, Cjj, Cjj, Cii)))
        Q.append(torch.stack((Cii, Cjj, Cjj, Cii)))
        pose_i.append(T_w_ci)
        pose_j.append(T_w_cj)

    X = torch.stack(X, dim=1)
    C = torch.stack(C, dim=1)
    Q = torch.stack(Q, dim=1)
    X, C, _, Q = _downsample(X, C, X, Q)
    b = X.shape[1]
    Xii, _, Xjj, _ = X[0], X[1], X[2], X[3]
    Cii, Cjj = C[0], C[1]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]
    pose_i = torch.stack(pose_i, dim=0)
    pose_j = torch.stack(pose_j, dim=0)

    conf_threshold = _match_conf_threshold()
    idx_i2j, valid_match_j, pair_conf_i2j = pi3_matching.match(
        Xjj, Xii, pose_j, pose_i, Cjj, Cii, conf_threshold=conf_threshold
    )
    idx_j2i, valid_match_i, pair_conf_j2i = pi3_matching.match(
        Xii, Xjj, pose_i, pose_j, Cii, Cjj, conf_threshold=conf_threshold
    )

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        pair_conf_i2j,
        pair_conf_j2i,
    )


def pi3x_match_asymmetric(
    model, frame_i, frame_j, idx_i2j_init=None, init_relative_pose=None
):
    encode_frame_image(frame_i)
    encode_frame_image(frame_j)
    images = torch.cat((_frame_image(frame_i), _frame_image(frame_j)), dim=0)
    output = _call_pi3x(model, images.unsqueeze(0))
    Xii, Xji, Xjj, _, Cii, Cji, _, _, T_w_ci, T_w_cj = _pair_output_to_maps(
        output, images.unsqueeze(0)
    )
    if T_w_ci is None or T_w_cj is None:
        raise KeyError("PI3X geometry matching requires camera poses.")
    X, C, D, Q = (
        torch.stack((Xii, Xji)),
        torch.stack((Cii, Cji)),
        Xii.new_empty((2, *Xii.shape[:2], 0)),
        torch.stack((Cii, Cji)),
    )
    X, C, D, Q = _downsample(X, C, D, Q)
    X_match, C_match, _, _ = _downsample(
        torch.stack((Xii, Xjj)),
        torch.stack((Cii, Cji)),
        torch.stack((Xii, Xjj)),
        torch.stack((Cii, Cji)),
    )
    Xii_match, Xjj_match = X_match[:1], X_match[1:]
    Cii_match, Cjj_match = C_match[:1], C_match[1:]
    # MASt3R-Fusion expects idx_i2j to be indexed by frame_j/keyframe pixels,
    # with each value pointing to the corresponding frame_i/current pixel.
    pose_src = T_w_cj[None]
    pose_dst = T_w_ci[None]
    pose_source = "pi3x_camera_poses"

    idx_i2j, valid_match_j, pair_conf_i2j, debug = pi3_matching.match(
        Xjj_match,
        Xii_match,
        pose_src,
        pose_dst,
        Cjj_match,
        Cii_match,
        conf_threshold=_match_conf_threshold(),
        return_debug=True,
    )
    if _match_debug_enabled():
        _print_match_debug(
            "asym",
            frame_i,
            frame_j,
            valid_match_j,
            pair_conf_i2j,
            Cjj_match,
            Cii_match,
            debug,
            pose_source,
        )
    Xii, Xji = X[:1], X[1:]
    Cii, Cji = C[:1], C[1:]
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
    Qji = pair_conf_i2j[0]
    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji
