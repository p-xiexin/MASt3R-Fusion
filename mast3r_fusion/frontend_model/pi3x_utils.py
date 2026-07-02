from pathlib import Path
import types
import contextlib

import einops
import torch
import torch.nn.functional as F

import mast3r_fusion.frontend_model.pi3_matching as pi3_matching
from mast3r_fusion.config import config


DEFAULT_PI3X_WEIGHTS = "checkpoints/pi3x/model.safetensors"


def _prior_enabled():
    cfg = config.get("pi3x", {})
    return bool(cfg.get("use_intrinsics_prior", False) or cfg.get("imu_predict", False))


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
        if not _prior_enabled() and hasattr(model, "disable_multimodal"):
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
    if not _prior_enabled() and hasattr(model, "disable_multimodal"):
        model.disable_multimodal()
    model = _patch_pi3x_rope_contiguous(model)
    return model.to(device)


def _frame_image(frame):
    image = frame.uimg.to(device=frame.img.device, dtype=frame.img.dtype)
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def _patch_positions(h, w, patch_size, device):
    patch_h = h // patch_size
    patch_w = w // patch_size
    ys, xs = torch.meshgrid(
        torch.arange(patch_h, device=device),
        torch.arange(patch_w, device=device),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).view(1, patch_h * patch_w, 2).long()


@torch.inference_mode()
def encode_frame_pi3x(model, frame):
    image = _frame_image(frame)
    h, w = image.shape[-2:]
    patch_size = getattr(model, "patch_size", 14)
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"PI3X encoder expects image height/width divisible by {patch_size}, "
            f"got {(h, w)}. Set dataset.target_img_size to multiples of {patch_size}."
        )

    if not hasattr(model, "encoder"):
        raise AttributeError("PI3X model does not expose an encoder for frame feature caching.")

    image_mean = getattr(model, "image_mean", None)
    image_std = getattr(model, "image_std", None)
    if image_mean is not None and image_std is not None:
        image_for_encoder = (image - image_mean.to(image.device, image.dtype)) / image_std.to(
            image.device, image.dtype
        )
    else:
        image_for_encoder = image

    try:
        encoded = model.encoder(image_for_encoder, is_training=True)
    except TypeError:
        encoded = model.encoder(image_for_encoder)
    if isinstance(encoded, dict):
        feat = encoded.get("x_norm_patchtokens")
        if feat is None:
            feat = encoded.get("patch_tokens")
    else:
        feat = encoded
    if feat is None:
        raise KeyError("PI3X encoder did not expose patch tokens.")

    frame.feat = feat.contiguous()
    frame.pos = _patch_positions(h, w, patch_size, feat.device)
    return frame.feat, frame.pos


def _sim3_to_c2w_matrix(T_WC, device, dtype):
    if T_WC is None:
        return None
    matrix = T_WC.matrix()
    if matrix.ndim == 3:
        matrix = matrix[0]
    matrix = matrix.to(device=device, dtype=dtype).clone()
    data = getattr(T_WC, "data", None)
    if data is not None:
        scale = data.reshape(-1, data.shape[-1])[0, -1].to(device=device, dtype=dtype)
        if torch.isfinite(scale).item() and torch.abs(scale).item() > 1e-8:
            matrix[:3, :3] = matrix[:3, :3] / scale
    matrix[3] = matrix.new_tensor([0, 0, 0, 1])
    return matrix


def _stack_frame_intrinsics(frames, images):
    pi3x_cfg = config.get("pi3x", {})
    if not (
        pi3x_cfg.get("use_intrinsics_prior", False)
        or pi3x_cfg.get("imu_predict", False)
    ):
        return None
    if frames is None:
        raise ValueError("PI3X pose/intrinsics prior requires pair frames.")
    Ks = []
    for frame in frames:
        K = getattr(frame, "K", None)
        if K is None:
            raise ValueError("PI3X pose/intrinsics prior requires frame.K for all pair frames.")
        Ks.append(K.to(device=images.device, dtype=images.dtype))
    if len(Ks) != images.shape[1]:
        raise ValueError("PI3X prior frame count must match the image pair count.")
    return torch.stack(Ks, dim=0).unsqueeze(0).contiguous()


def _stack_frame_poses(frames, images):
    if not config.get("pi3x", {}).get("imu_predict", False):
        return None
    if frames is None:
        raise ValueError("PI3X pose prior requires pair frames.")
    poses = []
    for frame in frames:
        pose = _sim3_to_c2w_matrix(
            getattr(frame, "T_WC", None),
            images.device,
            images.dtype,
        )
        if pose is None:
            raise ValueError("PI3X pose prior requires frame.T_WC for all pair frames.")
        poses.append(pose)
    if len(poses) != images.shape[1]:
        raise ValueError("PI3X prior frame count must match the image pair count.")
    return torch.stack(poses, dim=0).unsqueeze(0).contiguous()


def _pixel_grid_homogeneous(h, w, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)


def _require_cached_encoder_output(model, cached_feat):
    if cached_feat is None:
        raise ValueError("PI3X matching requires cached encoder tokens from encode_frame_pi3x().")
    missing = [
        name for name in ("decode", "forward_head") if not hasattr(model, name)
    ]
    if missing:
        raise AttributeError(
            "PI3X cached matching requires model methods: " + ", ".join(missing)
        )


def _cached_multimodal_priors(model, images, frames):
    b, n, _, h, w = images.shape
    device = images.device
    dtype = images.dtype
    poses = _stack_frame_poses(frames, images)
    intrinsics = _stack_frame_intrinsics(frames, images)
    use_ray = config.get("pi3x", {}).get("use_intrinsics_prior", False) or poses is not None
    mask_add_ray = None
    mask_add_pose = None
    ray_emb = None
    poses_relative = None

    if use_ray:
        if intrinsics is None:
            raise ValueError("PI3X pose/intrinsics prior requires frame.K for all pair frames.")
        pixel_grid = _pixel_grid_homogeneous(h, w, device, dtype)
        rays = torch.einsum(
            "bnij,hwj->bnhwi",
            torch.linalg.inv(intrinsics),
            pixel_grid,
        )[..., :2]
        ray_emb = model.ray_embed(rays.reshape(b * n, h, w, 2).permute(0, 3, 1, 2))
        mask_add_ray = torch.ones((b, n), device=device, dtype=torch.bool)

    if poses is not None:
        poses_relative = torch.linalg.inv(poses[:, :1]) @ poses
        pose_scale = poses_relative[..., 1:, :3, 3].norm(dim=-1)
        static_threshold = 2e-2
        is_static = pose_scale.max(dim=1)[0] < static_threshold
        mean_scale = pose_scale.mean(dim=1)
        moving = ~is_static
        poses_relative[moving, ..., :3, 3] /= mean_scale.view(b, 1, 1)[moving] + 1e-8
        mask_add_pose = torch.ones((b, n), device=device, dtype=torch.bool)
        bad_pose = mask_add_pose.sum(dim=1) == 1
        mask_add_pose[bad_pose] = False

    return ray_emb, mask_add_ray, poses_relative, mask_add_pose


def _disabled_autocast(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def _call_pi3x_from_cached_encoder(model, images, cached_feat, frames=None):
    b, n, _, h, w = images.shape
    patch_size = getattr(model, "patch_size", 14)
    patch_h, patch_w = h // patch_size, w // patch_size
    hidden = cached_feat.to(device=images.device, dtype=images.dtype).reshape(
        b, n, -1, cached_feat.shape[-1]
    )
    poses = None
    use_pose_mask = torch.zeros((b, n), device=images.device, dtype=torch.bool)
    if _prior_enabled() and getattr(model, "use_multimodal", False):
        with _disabled_autocast(images.device):
            ray_emb, mask_add_ray, poses, prior_pose_mask = _cached_multimodal_priors(
                model,
                images,
                frames,
            )
        if prior_pose_mask is not None:
            use_pose_mask = prior_pose_mask
        if ray_emb is not None:
            hidden = hidden.reshape(b * n, -1, hidden.shape[-1])
            hidden = hidden + ray_emb.to(hidden.dtype) * mask_add_ray.reshape(b * n, 1, 1)
            hidden = hidden.reshape(b, n, -1, hidden.shape[-1])

    decoded, pos = model.decode(hidden, n, h, w, poses=poses, use_pose_mask=use_pose_mask)
    return model.forward_head(decoded, pos, b, n, h, w, patch_h, patch_w)


def _call_pi3x(model, images, frames=None, cached_feat=None):
    h, w = images.shape[-2:]
    patch_size = getattr(model, "patch_size", 14)
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"PI3X expects image height/width divisible by {patch_size}, "
            f"got {(h, w)}. Set dataset.target_img_size to multiples of {patch_size}."
        )
    _require_cached_encoder_output(model, cached_feat)
    output = _call_pi3x_from_cached_encoder(model, images, cached_feat, frames=frames)
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


def _window_output_to_maps(output, images):
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
    confidences = torch.sigmoid(confidences)
    X, C, _, _ = _downsample(local_points, confidences, local_points, confidences)

    if poses is None:
        raise KeyError("PI3X window matching requires camera poses.")
    poses = torch.as_tensor(poses, device=X.device, dtype=X.dtype)
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    poses = poses[0]

    return X, C, poses


@torch.inference_mode()
def pi3x_inference_window(model, frames):
    if len(frames) == 0:
        raise ValueError("PI3X window inference requires at least one frame.")
    for frame in frames:
        encode_frame_pi3x(model, frame)
    images = torch.cat([_frame_image(frame) for frame in frames], dim=0).unsqueeze(0)
    cached_feat = torch.stack([frame.feat[0] for frame in frames], dim=0).unsqueeze(0)
    output = _call_pi3x(model, images, frames=frames, cached_feat=cached_feat)
    return _window_output_to_maps(output, images)


@torch.inference_mode()
def pi3x_match_window_edges(model, frames, edges, subpixel_factor=1):
    if subpixel_factor != 1:
        raise ValueError("PI3X window matching currently supports subpixel_factor=1 only.")
    X, C, poses = pi3x_inference_window(model, frames)
    h, w = X.shape[1:3]
    X_flat = X.reshape(X.shape[0], h * w, 3)
    C_flat = C.reshape(C.shape[0], h * w, 1)

    conf_threshold = _match_conf_threshold()
    constraints = {}
    for edge_i, edge_j in edges:
        Xii = X[edge_i : edge_i + 1]
        Xjj = X[edge_j : edge_j + 1]
        Cii = C[edge_i : edge_i + 1]
        Cjj = C[edge_j : edge_j + 1]
        pose_i = poses[edge_i : edge_i + 1]
        pose_j = poses[edge_j : edge_j + 1]
        idx_i2j, valid_match_j, pair_conf_i2j = pi3_matching.match(
            Xjj, Xii, pose_j, pose_i, Cjj, Cii, conf_threshold=conf_threshold
        )
        idx_j2i, valid_match_i, pair_conf_j2i = pi3_matching.match(
            Xii, Xjj, pose_i, pose_j, Cii, Cjj, conf_threshold=conf_threshold
        )
        constraints[(edge_i, edge_j)] = (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            C_flat[edge_i : edge_i + 1],
            C_flat[edge_j : edge_j + 1],
            pair_conf_i2j,
            pair_conf_j2i,
        )

    return X_flat, C_flat, poses, constraints


@torch.inference_mode()
def pi3x_inference_pair(model, frame_i, frame_j):
    encode_frame_pi3x(model, frame_i)
    encode_frame_pi3x(model, frame_j)
    images = torch.cat((_frame_image(frame_i), _frame_image(frame_j)), dim=0)
    cached_feat = torch.stack((frame_i.feat[0], frame_j.feat[0]), dim=0).unsqueeze(0)
    output = _call_pi3x(
        model,
        images.unsqueeze(0),
        frames=[frame_i, frame_j],
        cached_feat=cached_feat,
    )
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


def pi3x_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, frames_i=None, frames_j=None
):
    X, C, D, Q = [], [], [], []
    if frames_i is None or frames_j is None:
        raise ValueError("PI3X symmetric decode requires frames_i/frames_j for RGB images.")
    for b in range(feat_i.shape[0]):
        frames = [frames_i[b], frames_j[b]]
        images = torch.cat(
            (_frame_image(frames[0]), _frame_image(frames[1])), dim=0
        ).unsqueeze(0)
        cached_feat = torch.stack((feat_i[b], feat_j[b]), dim=0).unsqueeze(0)
        output = _call_pi3x(model, images, frames=frames, cached_feat=cached_feat)
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


def pi3x_match_symmetric(
    model,
    feat_i,
    pos_i,
    feat_j,
    pos_j,
    shape_i,
    shape_j,
    subpixel_factor=1,
    frames_i=None,
    frames_j=None,
):
    X, C, Q = [], [], []
    pose_i, pose_j = [], []
    if frames_i is None or frames_j is None:
        raise ValueError("PI3X symmetric matching requires frames_i/frames_j for RGB images.")
    for batch_idx in range(feat_i.shape[0]):
        frames = [frames_i[batch_idx], frames_j[batch_idx]]
        images = torch.cat(
            (_frame_image(frames[0]), _frame_image(frames[1])), dim=0
        ).unsqueeze(0)
        cached_feat = torch.stack((feat_i[batch_idx], feat_j[batch_idx]), dim=0).unsqueeze(0)
        output = _call_pi3x(model, images, frames=frames, cached_feat=cached_feat)
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
    encode_frame_pi3x(model, frame_i)
    encode_frame_pi3x(model, frame_j)
    images = torch.cat((_frame_image(frame_i), _frame_image(frame_j)), dim=0)
    cached_feat = torch.stack((frame_i.feat[0], frame_j.feat[0]), dim=0).unsqueeze(0)
    output = _call_pi3x(
        model,
        images.unsqueeze(0),
        frames=[frame_i, frame_j],
        cached_feat=cached_feat,
    )
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
