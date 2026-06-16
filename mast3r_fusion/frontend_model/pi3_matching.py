import torch


def _ensure_batch_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 3:
        return points.unsqueeze(0)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point map shape (B,H,W,3), got {tuple(points.shape)}.")
    return points


def _ensure_batch_conf(conf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if conf.ndim == 2:
        conf = conf.unsqueeze(0)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim != 3:
        raise ValueError(f"Expected confidence shape (B,H,W), got {tuple(conf.shape)}.")
    if conf.shape[:3] != points.shape[:3]:
        raise ValueError(
            "Confidence and point map shapes do not align: "
            f"C={tuple(conf.shape)}, X={tuple(points.shape)}."
        )
    return conf


def _ensure_batch_poses(poses: torch.Tensor) -> torch.Tensor:
    if poses.ndim == 2:
        return poses.unsqueeze(0)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose shape (B,4,4), got {tuple(poses.shape)}.")
    return poses


def _pixel_grid(h: int, w: int, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return x, y


def fit_intrinsics(
    points: torch.Tensor,
    conf: torch.Tensor,
    conf_threshold: float = 0.0,
    min_depth: float = 1e-6,
    min_points: int = 32,
) -> torch.Tensor:
    """Fit a per-batch pinhole K from PI3 local points to their pixel grid."""

    points = _ensure_batch_points(points)
    conf = _ensure_batch_conf(conf, points)
    b, h, w, _ = points.shape
    x_grid, y_grid = _pixel_grid(h, w, points.device, points.dtype)
    Ks = []
    for batch_idx in range(b):
        X = points[batch_idx]
        C = conf[batch_idx]
        z = X[..., 2]
        valid = torch.isfinite(X).all(dim=-1) & (z > min_depth) & (C > conf_threshold)
        if valid.sum() < min_points:
            raise ValueError(
                f"Not enough valid PI3 points to fit intrinsics: {int(valid.sum())}."
            )

        xn = X[..., 0][valid] / z[valid]
        yn = X[..., 1][valid] / z[valid]
        u = x_grid[valid]
        v = y_grid[valid]
        ones = torch.ones_like(xn)

        Ax = torch.stack((xn, ones), dim=-1)
        Ay = torch.stack((yn, ones), dim=-1)
        fx_cx = torch.linalg.lstsq(Ax, u[:, None]).solution[:, 0]
        fy_cy = torch.linalg.lstsq(Ay, v[:, None]).solution[:, 0]
        fx, cx = fx_cx[0], fx_cx[1]
        fy, cy = fy_cy[0], fy_cy[1]
        Ks.append(
            torch.stack(
                (
                    torch.stack((fx, torch.zeros_like(fx), cx)),
                    torch.stack((torch.zeros_like(fy), fy, cy)),
                    torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype),
                )
            )
        )
    return torch.stack(Ks, dim=0)


def transform_points(points: torch.Tensor, pose_src: torch.Tensor, pose_dst: torch.Tensor):
    """Transform points from source camera coordinates into destination camera coordinates."""

    points = _ensure_batch_points(points)
    pose_src = _ensure_batch_poses(pose_src)
    pose_dst = _ensure_batch_poses(pose_dst)
    b, h, w, _ = points.shape
    if pose_src.shape[0] != b or pose_dst.shape[0] != b:
        raise ValueError("Pose batch size must match point map batch size.")

    flat = points.reshape(b, -1, 3)
    ones = torch.ones(b, flat.shape[1], 1, device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((flat, ones), dim=-1)
    dst_from_src = torch.linalg.inv(pose_dst) @ pose_src
    transformed = torch.bmm(dst_from_src, homogeneous.transpose(1, 2)).transpose(1, 2)
    return transformed[..., :3].reshape(b, h, w, 3)


def project_to_index(
    points_dst: torch.Tensor,
    K_dst: torch.Tensor,
    conf_src: torch.Tensor,
    conf_dst: torch.Tensor,
    conf_threshold: float = 0.0,
    min_depth: float = 1e-6,
    balance_grid_size: int = 0,
    max_matches_per_cell: int = 0,
    unique_target: bool = False,
    return_debug: bool = False,
):
    """Project source-grid points in destination coordinates to destination indices."""

    points_dst = _ensure_batch_points(points_dst)
    conf_src = _ensure_batch_conf(conf_src, points_dst)
    conf_dst = _ensure_batch_conf(conf_dst, points_dst)
    if K_dst.ndim == 2:
        K_dst = K_dst.unsqueeze(0)
    b, h, w, _ = points_dst.shape
    if K_dst.shape[0] != b:
        raise ValueError("K batch size must match point map batch size.")

    z = points_dst[..., 2]
    fx = K_dst[:, None, None, 0, 0]
    fy = K_dst[:, None, None, 1, 1]
    cx = K_dst[:, None, None, 0, 2]
    cy = K_dst[:, None, None, 1, 2]
    u = fx * (points_dst[..., 0] / z) + cx
    v = fy * (points_dst[..., 1] / z) + cy
    u_round = torch.round(u).long()
    v_round = torch.round(v).long()
    in_bounds = (u_round >= 0) & (u_round < w) & (v_round >= 0) & (v_round < h)

    conf_projected_dst = torch.zeros_like(conf_src)
    batch_idx = torch.arange(b, device=points_dst.device)[:, None, None].expand(b, h, w)
    conf_projected_dst[in_bounds] = conf_dst[
        batch_idx[in_bounds], v_round[in_bounds], u_round[in_bounds]
    ]
    finite_points = torch.isfinite(points_dst).all(dim=-1)
    finite_projection = torch.isfinite(u) & torch.isfinite(v)
    positive_depth = z > min_depth
    valid_conf_src = conf_src > conf_threshold
    valid_conf_dst = conf_projected_dst > conf_threshold
    pair_conf = torch.sqrt(conf_src * conf_projected_dst)
    valid = (
        finite_points
        & finite_projection
        & positive_depth
        & valid_conf_src
        & valid_conf_dst
        & in_bounds
        & torch.isfinite(pair_conf)
    )
    idx = torch.zeros(b, h, w, device=points_dst.device, dtype=torch.long)
    idx[valid] = v_round[valid] * w + u_round[valid]
    valid_before_balance = valid
    if unique_target:
        valid = _filter_unique_target(idx, valid, pair_conf, h, w)
    if balance_grid_size > 0 and max_matches_per_cell > 0:
        valid = _filter_grid_topk(valid, pair_conf, balance_grid_size, max_matches_per_cell)

    if not return_debug:
        return idx.view(b, -1), valid.view(b, -1, 1), pair_conf.view(b, -1, 1)

    debug = {
        "u": u.view(b, -1),
        "v": v.view(b, -1),
        "in_bounds": in_bounds.view(b, -1),
        "finite_points": finite_points.view(b, -1),
        "finite_projection": finite_projection.view(b, -1),
        "positive_depth": positive_depth.view(b, -1),
        "valid_conf_src": valid_conf_src.view(b, -1),
        "valid_conf_dst": valid_conf_dst.view(b, -1),
        "valid_before_balance": valid_before_balance.view(b, -1),
        "conf_projected_dst": conf_projected_dst.view(b, -1, 1),
        "z": z.view(b, -1),
    }
    return idx.view(b, -1), valid.view(b, -1, 1), pair_conf.view(b, -1, 1), debug


def _filter_unique_target(idx, valid, score, h, w):
    valid_flat = valid.view(valid.shape[0], -1)
    idx_flat = idx.view(idx.shape[0], -1)
    score_flat = score.view(score.shape[0], -1)
    filtered = torch.zeros_like(valid_flat)
    neg_inf = -torch.inf
    for batch_idx in range(valid_flat.shape[0]):
        valid_b = valid_flat[batch_idx]
        if not valid_b.any():
            continue
        src_ids = torch.nonzero(valid_b, as_tuple=False).squeeze(1)
        dst_ids = idx_flat[batch_idx, src_ids]
        scores = score_flat[batch_idx, src_ids]
        max_scores = torch.full(
            (h * w,), neg_inf, device=score.device, dtype=score.dtype
        )
        max_scores.scatter_reduce_(0, dst_ids, scores, reduce="amax", include_self=True)
        keep = scores >= max_scores[dst_ids]
        filtered[batch_idx, src_ids[keep]] = True
    return filtered.view_as(valid)


def _filter_grid_topk(valid, score, grid_size, max_per_cell):
    filtered = torch.zeros_like(valid)
    _, h, w = valid.shape
    for batch_idx in range(valid.shape[0]):
        for y0 in range(0, h, grid_size):
            y1 = min(y0 + grid_size, h)
            for x0 in range(0, w, grid_size):
                x1 = min(x0 + grid_size, w)
                cell_valid = valid[batch_idx, y0:y1, x0:x1]
                count = int(cell_valid.sum().item())
                if count == 0:
                    continue
                cell_filtered = filtered[batch_idx, y0:y1, x0:x1]
                if count <= max_per_cell:
                    cell_filtered[cell_valid] = True
                    continue
                cell_score = score[batch_idx, y0:y1, x0:x1]
                valid_yx = torch.nonzero(cell_valid, as_tuple=False)
                topk = torch.topk(cell_score[cell_valid], max_per_cell).indices
                keep_yx = valid_yx[topk]
                cell_filtered[keep_yx[:, 0], keep_yx[:, 1]] = True
    return filtered


def match(
    points_src: torch.Tensor,
    points_dst: torch.Tensor,
    pose_src: torch.Tensor,
    pose_dst: torch.Tensor,
    conf_src: torch.Tensor,
    conf_dst: torch.Tensor,
    K_dst: torch.Tensor = None,
    conf_threshold: float = 0.0,
    min_depth: float = 1e-6,
    balance_grid_size: int = 0,
    max_matches_per_cell: int = 0,
    unique_target: bool = False,
    return_debug: bool = False,
):
    """Return dense source-to-destination correspondences from PI3 geometry.

    Args:
        points_src: PI3 local point map in source camera coordinates, `(B,H,W,3)`.
        points_dst: PI3 local point map in destination camera coordinates, `(B,H,W,3)`.
        pose_src: source camera-to-world pose, `(B,4,4)`.
        pose_dst: destination camera-to-world pose, `(B,4,4)`.
        conf_src: source point confidence, `(B,H,W)`.
        conf_dst: destination point confidence, `(B,H,W)`.
        K_dst: optional destination pinhole matrix. If omitted, it is fitted
            from `points_dst` and `conf_dst`.

    Returns:
        `(idx_src_to_dst, valid_match_dst, pair_conf)` with flattened shapes
        `(B,H*W)`, `(B,H*W,1)`, and `(B,H*W,1)`. The returned index is ordered
        by source pixels and stores linear destination indices.
    """

    points_src = _ensure_batch_points(points_src)
    points_dst = _ensure_batch_points(points_dst)
    conf_src = _ensure_batch_conf(conf_src, points_src)
    conf_dst = _ensure_batch_conf(conf_dst, points_dst)
    if points_src.shape != points_dst.shape:
        raise ValueError(
            f"PI3 geometry matcher currently expects same-size maps, got "
            f"{tuple(points_src.shape)} and {tuple(points_dst.shape)}."
        )
    if K_dst is None:
        K_dst = fit_intrinsics(points_dst, conf_dst, conf_threshold, min_depth)
    points_src_in_dst = transform_points(points_src, pose_src, pose_dst)
    result = project_to_index(
        points_src_in_dst,
        K_dst,
        conf_src,
        conf_dst,
        conf_threshold,
        min_depth,
        balance_grid_size=balance_grid_size,
        max_matches_per_cell=max_matches_per_cell,
        unique_target=unique_target,
        return_debug=return_debug,
    )
    if return_debug:
        result = (*result[:3], {**result[3], "K_dst": K_dst, "points_src_in_dst": points_src_in_dst})
    return result
