#!/usr/bin/env python3
import argparse
import collections
import math
from pathlib import Path


def _median(values):
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _fmt(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summarize(path):
    light = []
    windows = []
    syncs = []
    plain = 0
    skipped = 0

    path = Path(path)
    if not path.exists():
        print(f"log: {path}")
        print("missing log file")
        return

    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "light" and len(parts) >= 10:
                light.append(
                    {
                        "ref": int(parts[1]),
                        "frame": int(parts[2]),
                        "source": parts[3],
                        "inliers": int(parts[4]),
                        "ratio": float(parts[5]),
                        "parallax": float(parts[6]),
                        "new_kf": int(parts[8]),
                        "reason": parts[9],
                    }
                )
            elif parts[0] == "window" and len(parts) >= 4:
                windows.append((int(parts[1]), int(parts[2]), int(parts[3])))
            elif parts[0] == "light_sync" and len(parts) >= 4:
                syncs.append((int(parts[1]), int(parts[2]), int(parts[3])))
            else:
                plain += 1
        except ValueError:
            skipped += 1

    sources = collections.Counter(item["source"] for item in light)
    reasons = collections.Counter(item["reason"] for item in light)
    inliers = [item["inliers"] for item in light]
    ratios = [item["ratio"] for item in light]
    parallaxes = [item["parallax"] for item in light]

    print(f"log: {path}")
    print(f"light tracks: {len(light)}")
    print(f"  sources: {dict(sources)}")
    print(f"  reasons: {dict(reasons)}")
    print(f"  inliers median/max: {_fmt(_median(inliers))}/{max(inliers) if inliers else 0}")
    print(f"  ratio median/max: {_fmt(_median(ratios))}/{_fmt(max(ratios) if ratios else float('nan'))}")
    print(f"  parallax median/max: {_fmt(_median(parallaxes))}/{_fmt(max(parallaxes) if parallaxes else float('nan'))}")
    print(f"pi3x windows: {len(windows)}")
    if windows:
        print(f"  first/last: {windows[0]} / {windows[-1]}")
    print(f"light syncs: {len(syncs)}")
    if syncs:
        print(f"  first/last: {syncs[0]} / {syncs[-1]}")
    print(f"legacy pose lines: {plain}")
    print(f"skipped malformed lines: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="tracker.log")
    args = parser.parse_args()
    summarize(args.log)


if __name__ == "__main__":
    main()
