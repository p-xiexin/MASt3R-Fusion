from __future__ import annotations

import argparse

import numpy as np


def load_tum(path: str):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]


def load_reference(path: str):
    data = np.loadtxt(path)
    order = np.argsort(data[:, 0])
    data = data[order]
    keep = np.concatenate([[True], np.diff(data[:, 0]) > 1e-6])
    data = data[keep]
    return data[:, 0], data[:, 1:4]


def match_by_time(ts, xyz, ref_ts, ref_xyz, max_dt: float):
    idx = np.searchsorted(ref_ts, ts)
    matched = []
    ref_matched = []
    for i, t in enumerate(ts):
        candidates = []
        if idx[i] < len(ref_ts):
            candidates.append(idx[i])
        if idx[i] > 0:
            candidates.append(idx[i] - 1)
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(ref_ts[k] - t))
        if abs(ref_ts[j] - t) <= max_dt:
            matched.append(xyz[i])
            ref_matched.append(ref_xyz[j])
    return np.asarray(matched), np.asarray(ref_matched)


def sim3_align(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    variance = np.sum(src_centered * src_centered) / len(src)
    scale = np.trace(np.diag(singular_values) @ sign) / variance
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def summarize_errors(errors):
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--max-dt", type=float, default=0.11)
    args = parser.parse_args()

    ts, xyz = load_tum(args.estimate)
    ref_ts, ref_xyz = load_reference(args.reference)
    matched, ref_matched = match_by_time(ts, xyz, ref_ts, ref_xyz, args.max_dt)
    if len(matched) < 3:
        raise RuntimeError(f"Not enough timestamp matches: {len(matched)}")

    rel_errors = np.linalg.norm((matched - matched[0]) - (ref_matched - ref_matched[0]), axis=1)
    scale, rotation, translation = sim3_align(matched, ref_matched)
    aligned = scale * (matched @ rotation.T) + translation
    sim3_errors = np.linalg.norm(aligned - ref_matched, axis=1)

    print(f"matches: {len(matched)}")
    print(f"estimate span: {ts[0]:.6f} {ts[-1]:.6f}")
    print(f"reference span: {ref_ts[0]:.6f} {ref_ts[-1]:.6f}")
    print(f"sim3 scale: {scale:.9f}")
    for name, values in [("relative", summarize_errors(rel_errors)), ("sim3", summarize_errors(sim3_errors))]:
        print(
            f"{name}: rmse={values['rmse']:.6f} mean={values['mean']:.6f} "
            f"median={values['median']:.6f} p90={values['p90']:.6f} max={values['max']:.6f}"
        )


if __name__ == "__main__":
    main()
