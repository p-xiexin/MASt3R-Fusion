import argparse
import json
from pathlib import Path

import dump_mast3r_pair_outputs as pair_dump


def _list_images(image_dir, extensions):
    image_dir = Path(image_dir)
    suffixes = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def _make_pairs(images, start, end, stride, pair_gap):
    end = len(images) - pair_gap if end is None else min(end, len(images) - pair_gap)
    if start < 0 or start >= len(images):
        raise ValueError(f"start={start} is out of range for {len(images)} images")
    if end <= start:
        raise ValueError(f"end={end} must be greater than start={start}")

    pairs = []
    for i in range(start, end, stride):
        j = i + pair_gap
        if j >= len(images):
            break
        pairs.append((i, j, images[i], images[j]))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch dump MASt3R outputs for image pairs in a directory."
    )
    parser.add_argument("image_dir", help="Directory containing input images, e.g. KITTI image_00/data.")
    parser.add_argument("output_dir", help="Directory where per-pair output folders are written.")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--match-border", type=int, default=3)
    parser.add_argument("--match-block-size", type=int, default=2**13)
    parser.add_argument("--num-viz-matches", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--save-desc", action="store_true")
    parser.add_argument("--no-raw-pt", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pair-gap", type=int, default=1)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image filename extensions to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.pair_gap <= 0:
        raise ValueError("--pair-gap must be positive")

    images = _list_images(args.image_dir, args.extensions)
    pairs = _make_pairs(images, args.start, args.end, args.stride, args.pair_gap)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_dump._load_runtime_imports()
    model = pair_dump.load_mast3r(args.weights, device=args.device)
    model.eval()

    index = []
    print(f"Found {len(images)} images")
    print(f"Running {len(pairs)} image pairs")
    print(f"Output directory: {output_dir.resolve()}")

    for pair_idx, (i, j, image_a, image_b) in enumerate(pairs):
        pair_dir = output_dir / f"{pair_idx:06d}"
        print(f"[{pair_idx + 1}/{len(pairs)}] {image_a.name} -> {image_b.name}")
        summary = pair_dump.dump_pair_outputs(
            model,
            image_a,
            image_b,
            pair_dir,
            args,
            verbose=False,
        )
        index.append(
            {
                "pair_index": pair_idx,
                "image_index_a": i,
                "image_index_b": j,
                "image_a": str(image_a),
                "image_b": str(image_b),
                "output_dir": str(pair_dir),
                "raw_match_count": summary["raw_match_count"],
                "valid_match_count": summary["valid_match_count"],
            }
        )

    with open(output_dir / "batch_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Saved batch index to {output_dir / 'batch_index.json'}")


if __name__ == "__main__":
    main()
