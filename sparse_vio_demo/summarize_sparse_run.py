from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_tum(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        rows.append((ts, xyz))
    return rows


def trajectory_length(rows: list[tuple[float, np.ndarray]]) -> float:
    if len(rows) < 2:
        return 0.0
    pts = np.asarray([xyz for _, xyz in rows], dtype=np.float64)
    delta = np.diff(pts, axis=0)
    return float(np.linalg.norm(delta, axis=1).sum())


def build_summary(args) -> dict:
    tum_rows = read_tum(Path(args.estimate_tum))
    first_ts = tum_rows[0][0] if tum_rows else None
    last_ts = tum_rows[-1][0] if tum_rows else None
    pose_count = len(tum_rows)
    frame_span = pose_count if args.start is None or args.end is None else int(args.end) - int(args.start)
    effective_gate = pose_count >= int(args.min_eval_frames) and frame_span >= int(args.min_eval_frames)
    return {
        "estimate_tum": str(args.estimate_tum),
        "frontend": args.frontend,
        "use_imu": bool(args.use_imu),
        "range": {
            "start": args.start,
            "end": args.end,
            "subsample": args.subsample,
            "frame_span": frame_span,
        },
        "trajectory": {
            "pose_count": pose_count,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "duration_sec": None if first_ts is None or last_ts is None else float(last_ts - first_ts),
            "path_length": trajectory_length(tum_rows),
        },
        "effective_run_gate_passed": bool(effective_gate),
        "notes": [
            "This summary is a sparse landmark + GTSAM SLAM run gate.",
            "Trajectory evaluation should use SE3 alignment only.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate-tum", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frontend", default=None)
    parser.add_argument("--use-imu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--min-eval-frames", type=int, default=1200)
    args = parser.parse_args()

    summary = build_summary(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    traj = summary["trajectory"]
    print(
        "sparse_run: "
        f"poses={traj['pose_count']} "
        f"path_length={traj['path_length']:.3f} "
        f"effective={summary['effective_run_gate_passed']} "
        f"summary={out}"
    )


if __name__ == "__main__":
    main()
