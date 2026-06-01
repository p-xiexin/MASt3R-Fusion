from pathlib import Path

import einops
import torch
import torch.nn.functional as F

import mast3r_fusion.matching as matching
from mast3r_fusion.config import config


DEFAULT_PI3X_WEIGHTS = "checkpoints/pi3x/model.safetensors"


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
        return Pi3X.from_pretrained(weights_path).to(device).eval()

    model = Pi3X()
    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weights_path, device="cpu")
    else:
        state = torch.load(weights_path, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


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
    if hasattr(model, "infer"):
        output = model.infer(images)
    else:
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
    confidences = torch.sigmoid(confidences) if confidences.max() > 1 else confidences

    Xii = local_points[0]
    Xjj = local_points[1]
    if poses is not None:
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

    descriptors = images[0].permute(0, 2, 3, 1).contiguous()
    Dii = F.normalize(descriptors[0], dim=-1)
    Djj = F.normalize(descriptors[1], dim=-1)

    return Xii, Xji, Xjj, Xij, confidences[0], confidences[1], Dii, Djj


@torch.inference_mode()
def pi3x_inference_pair(model, frame_i, frame_j):
    encode_frame_image(frame_i)
    encode_frame_image(frame_j)
    images = torch.cat((_frame_image(frame_i), _frame_image(frame_j)), dim=0)
    output = _call_pi3x(model, images.unsqueeze(0))
    Xii, Xji, _, _, Cii, Cji, Dii, Dji = _pair_output_to_maps(output, images.unsqueeze(0))
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
        Xii, Xji, Xjj, Xij, Cii, Cjj, Dii, Djj = _pair_output_to_maps(output, images)
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
    X, C, D, Q = pi3x_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )
    b = X.shape[1]
    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, subpixel_factor=subpixel_factor
    )
    idx_j2i, valid_match_i = matching.match(
        Xjj, Xij, Djj, Dij, subpixel_factor=subpixel_factor
    )

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


def pi3x_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = pi3x_inference_pair(model, frame_i, frame_j)
    Xii, Xji = X[:1], X[1:]
    Dii, Dji = D[:1], D[1:]
    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji
