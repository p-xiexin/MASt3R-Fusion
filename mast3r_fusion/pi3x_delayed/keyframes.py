import torch


def configure_feature_storage(shared, h, w, feature_spec):
    if feature_spec is None:
        return

    feat_dim = int(getattr(feature_spec, "feat_dim", 1024))
    patch_size = int(getattr(feature_spec, "patch_size", 16))
    if hasattr(feature_spec, "num_patches"):
        num_patches = int(feature_spec.num_patches(h, w))
    else:
        num_patches = h * w // (patch_size * patch_size)

    shared.feat_dim = feat_dim
    shared.num_patches = num_patches

    if hasattr(shared, "buffer"):
        feat_shape = (shared.buffer, 1, num_patches, feat_dim)
        pos_shape = (shared.buffer, 1, num_patches, 2)
    else:
        feat_shape = (1, num_patches, feat_dim)
        pos_shape = (1, num_patches, 2)

    shared.feat = torch.zeros(
        feat_shape, device=shared.device, dtype=shared.dtype
    ).share_memory_()
    shared.pos = torch.zeros(
        pos_shape, device=shared.device, dtype=torch.long
    ).share_memory_()


def set_keyframe_global(keyframes, idx, frame):
    with keyframes.lock:
        storage_idx = idx - keyframes.rollup_sum.value
        if storage_idx < 0:
            msg = (
                f"[ERROR] PI3X delayed set invalid idx={idx}, "
                f"storage_idx={storage_idx}, rollup_sum={keyframes.rollup_sum.value}, "
                f"n_size={keyframes.n_size.value}"
            )
            print(msg)
            raise IndexError(msg)
        keyframes.n_size.value = max(storage_idx + 1, keyframes.n_size.value)
        keyframes.dataset_idx[storage_idx] = frame.frame_id
        keyframes.img[storage_idx] = frame.img
        keyframes.uimg[storage_idx] = frame.uimg
        keyframes.img_shape[storage_idx] = frame.img_shape
        keyframes.img_true_shape[storage_idx] = frame.img_true_shape
        keyframes.T_WC[storage_idx] = frame.T_WC.data
        keyframes.X[storage_idx] = frame.X_canon
        keyframes.C[storage_idx] = frame.C
        keyframes.feat[storage_idx] = frame.feat
        keyframes.pos[storage_idx] = frame.pos
        keyframes.N[storage_idx] = frame.N
        keyframes.N_updates[storage_idx] = frame.N_updates
        keyframes.is_dirty[storage_idx] = True


def update_t_wcs_global_checked(keyframes, T_WCs, idx):
    with keyframes.lock:
        storage_idx = idx - keyframes.rollup_sum.value
        if torch.any(storage_idx < 0) or torch.any(storage_idx >= keyframes.n_size.value):
            msg = (
                f"[ERROR] PI3X delayed update_T_WCs invalid idx={idx}, "
                f"storage_idx={storage_idx}, rollup_sum={keyframes.rollup_sum.value}, "
                f"n_size={keyframes.n_size.value}"
            )
            print(msg)
            raise IndexError(msg)
        sim3_data = T_WCs.data
        quat_norm = torch.linalg.norm(sim3_data[..., 3:7], dim=-1)
        scale = sim3_data[..., 7]
        invalid = (
            torch.any(~torch.isfinite(sim3_data))
            or torch.any(quat_norm <= 1e-8)
            or torch.any(~torch.isfinite(scale))
            or torch.any(torch.abs(scale) <= 1e-8)
        )
        if invalid:
            msg = (
                f"[ERROR] PI3X delayed update_T_WCs invalid Sim3 idx={idx}, "
                f"quat_norm={quat_norm.detach().cpu().tolist()}, "
                f"scale={scale.detach().cpu().tolist()}, "
                f"rollup_sum={keyframes.rollup_sum.value}, n_size={keyframes.n_size.value}"
            )
            print(msg)
            raise ValueError(msg)
        keyframes.T_WC[storage_idx] = sim3_data


def limited_roll_up(keyframes, requested_rollup, last_pin):
    rollup = min(requested_rollup, max(last_pin - keyframes.rollup_sum.value, 0))
    if rollup > 0:
        keyframes.roll_up(rollup)
    return rollup
