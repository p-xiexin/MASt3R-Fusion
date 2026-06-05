import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTENSIONS = {".npy", ".npz", ".png", ".tif", ".tiff"}
REQUIRED_FRAME_COLUMNS = {"timestamp", "image_path"}


class PI3DatasetError(ValueError):
    """Raised when a converted PI3-format dataset is invalid."""


@dataclass(frozen=True)
class PI3Frame:
    index: int
    timestamp: float
    image_path: Path
    depth_path: Path | None = None
    frame_id: str | None = None
    pose: np.ndarray | None = None


@dataclass(frozen=True)
class PI3Camera:
    width: int
    height: int
    K: np.ndarray
    distortion: np.ndarray | None = None
    model: str = "pinhole"


class PI3Dataset:
    """Loader and validator for the experimental PI3 target dataset format."""

    def __init__(self, root, load_images=False, strict=True):
        self.root = Path(root)
        self.load_images = load_images
        self.strict = strict
        self.frames: list[PI3Frame] = []
        self.camera: PI3Camera | None = None
        self.errors: list[str] = []

        self._load()
        if self.errors and strict:
            raise PI3DatasetError(self.format_errors())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        item = {
            "index": frame.index,
            "timestamp": frame.timestamp,
            "image_path": frame.image_path,
            "depth_path": frame.depth_path,
            "frame_id": frame.frame_id,
            "pose": frame.pose,
            "camera": self.camera,
            "camera_intrinsics": self.camera.K if self.camera is not None else None,
            "camera_pose": frame.pose,
            "label": self.root.name,
            "instance": frame.frame_id,
        }
        if self.load_images:
            item["image"] = self.read_image(frame.image_path)
            item["img"] = item["image"]
            if frame.depth_path is not None:
                item["depthmap"] = self.read_depth(frame.depth_path)
        return item

    def as_pi3_view(self, idx, load_image=True, load_depth=True):
        """Return a sample using keys expected by PI3 training dataset views."""
        frame = self.frames[idx]
        view = {
            "img": self.read_image(frame.image_path) if load_image else frame.image_path,
            "camera_intrinsics": self.camera.K if self.camera is not None else None,
            "camera_pose": frame.pose,
            "dataset": "converted_pi3",
            "label": self.root.name,
            "instance": frame.frame_id,
            "timestamp": frame.timestamp,
            "image_path": frame.image_path,
        }
        if frame.depth_path is not None:
            view["depthmap"] = (
                self.read_depth(frame.depth_path) if load_depth else frame.depth_path
            )
        return view

    def read_image(self, image_path):
        try:
            import cv2
        except ImportError as exc:
            raise PI3DatasetError("OpenCV is required to read images: install cv2.") from exc

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise PI3DatasetError(f"Cannot read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read_depth(self, depth_path):
        suffix = depth_path.suffix.lower()
        if suffix == ".npy":
            depth = np.load(depth_path)
        elif suffix == ".npz":
            data = np.load(depth_path)
            if "depthmap" in data:
                depth = data["depthmap"]
            elif "depth" in data:
                depth = data["depth"]
            else:
                raise PI3DatasetError(
                    f"Depth npz must contain 'depthmap' or 'depth': {depth_path}"
                )
        else:
            try:
                import cv2
            except ImportError as exc:
                raise PI3DatasetError("OpenCV is required to read depth images.") from exc

            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise PI3DatasetError(f"Cannot read depth map: {depth_path}")

        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise PI3DatasetError(f"Depth map must be 2D: {depth_path}")
        if not np.isfinite(depth.astype(np.float32)).all():
            raise PI3DatasetError(f"Depth map contains non-finite values: {depth_path}")
        return depth

    def format_errors(self, max_errors=20):
        if not self.errors:
            return "PI3 dataset validation passed."

        shown = self.errors[:max_errors]
        lines = ["PI3 dataset validation failed:"]
        lines.extend(f"  - {error}" for error in shown)
        if len(self.errors) > max_errors:
            lines.append(f"  - ... {len(self.errors) - max_errors} more errors")
        return "\n".join(lines)

    def summary(self):
        camera = "missing"
        if self.camera is not None:
            camera = (
                f"{self.camera.model} {self.camera.width}x{self.camera.height} "
                f"K={self.camera.K.reshape(-1).tolist()}"
            )
        return (
            f"root={self.root}\n"
            f"frames={len(self.frames)}\n"
            f"camera={camera}\n"
            f"errors={len(self.errors)}"
        )

    def _load(self):
        if not self.root.exists():
            self.errors.append(f"Dataset root does not exist: {self.root}")
            return
        if not self.root.is_dir():
            self.errors.append(f"Dataset root is not a directory: {self.root}")
            return

        self.camera = self._load_camera()
        poses = self._load_poses()
        self.frames = self._load_frames(poses)
        self._validate_sequence()

    def _load_camera(self):
        camera_path = self.root / "camera.json"
        if not camera_path.exists():
            self.errors.append("Missing camera.json.")
            return None

        try:
            with camera_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            self.errors.append(f"camera.json is not valid JSON: {exc}")
            return None

        required = ["width", "height", "K"]
        for key in required:
            if key not in data:
                self.errors.append(f"camera.json missing required key: {key}")
        if any(key not in data for key in required):
            return None

        try:
            width = int(data["width"])
            height = int(data["height"])
            K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            distortion = data.get("distortion")
            if distortion is not None:
                distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
            model = str(data.get("model", "pinhole"))
        except (TypeError, ValueError) as exc:
            self.errors.append(f"camera.json has invalid numeric fields: {exc}")
            return None

        if width <= 0 or height <= 0:
            self.errors.append("camera.json width and height must be positive.")
        if not np.isfinite(K).all():
            self.errors.append("camera.json K contains non-finite values.")
        if K[2, 2] == 0:
            self.errors.append("camera.json K[2][2] must be non-zero.")

        return PI3Camera(
            width=width,
            height=height,
            K=K,
            distortion=distortion,
            model=model,
        )

    def _load_poses(self):
        pose_path = self.root / "poses.txt"
        if not pose_path.exists():
            return {}

        poses = {}
        with pose_path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 17:
                    self.errors.append(
                        "poses.txt line "
                        f"{line_number} must contain frame_id plus 16 matrix values."
                    )
                    continue
                frame_id = parts[0]
                try:
                    pose = np.asarray(parts[1:], dtype=np.float64).reshape(4, 4)
                except ValueError as exc:
                    self.errors.append(f"poses.txt line {line_number} invalid pose: {exc}")
                    continue
                if not np.isfinite(pose).all():
                    self.errors.append(f"poses.txt line {line_number} contains non-finite values.")
                    continue
                poses[frame_id] = pose
        return poses

    def _load_frames(self, poses):
        frames_path = self.root / "frames.csv"
        if not frames_path.exists():
            self.errors.append("Missing frames.csv.")
            return []

        frames = []
        with frames_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                self.errors.append("frames.csv is empty or missing a header row.")
                return []

            columns = {name.strip() for name in reader.fieldnames}
            missing = sorted(REQUIRED_FRAME_COLUMNS - columns)
            if missing:
                self.errors.append(
                    "frames.csv missing required columns: " + ", ".join(missing)
                )
                return []

            for index, row in enumerate(reader):
                line_number = index + 2
                frame = self._parse_frame_row(row, index, line_number, poses)
                if frame is not None:
                    frames.append(frame)

        if not frames:
            self.errors.append("frames.csv does not contain any valid frames.")
        return frames

    def _parse_frame_row(self, row, index, line_number, poses):
        raw_timestamp = row.get("timestamp", "").strip()
        raw_image_path = row.get("image_path", "").strip()
        raw_depth_path = row.get("depth_path", "").strip()
        frame_id = row.get("frame_id", "").strip() or f"{index:06d}"

        try:
            timestamp = float(raw_timestamp)
        except ValueError:
            self.errors.append(f"frames.csv line {line_number} has invalid timestamp.")
            return None

        if not np.isfinite(timestamp):
            self.errors.append(f"frames.csv line {line_number} timestamp is not finite.")
            return None

        if not raw_image_path:
            self.errors.append(f"frames.csv line {line_number} has empty image_path.")
            return None

        image_path = self._resolve_relative_path(
            raw_image_path, "image_path", line_number
        )
        if image_path is None:
            return None
        depth_path = None
        if raw_depth_path:
            depth_path = self._resolve_relative_path(
                raw_depth_path, "depth_path", line_number
            )
            if depth_path is None:
                return None

        pose = poses.get(frame_id)
        return PI3Frame(
            index=index,
            timestamp=timestamp,
            image_path=image_path,
            depth_path=depth_path,
            frame_id=frame_id,
            pose=pose,
        )

    def _resolve_relative_path(self, raw_path, field_name, line_number):
        path = (self.root / raw_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            self.errors.append(
                f"frames.csv line {line_number} {field_name} escapes dataset root: {raw_path}"
            )
            return None
        return path

    def _validate_sequence(self):
        timestamps = []
        seen_frame_ids = set()
        seen_paths = set()

        for frame in self.frames:
            timestamps.append(frame.timestamp)

            if frame.frame_id in seen_frame_ids:
                self.errors.append(f"Duplicate frame_id: {frame.frame_id}")
            seen_frame_ids.add(frame.frame_id)

            if frame.image_path in seen_paths:
                self.errors.append(f"Duplicate image_path: {frame.image_path}")
            seen_paths.add(frame.image_path)

            if frame.image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                self.errors.append(f"Unsupported image extension: {frame.image_path}")
            if not frame.image_path.exists():
                self.errors.append(f"Image file does not exist: {frame.image_path}")
                continue
            if not frame.image_path.is_file():
                self.errors.append(f"Image path is not a file: {frame.image_path}")
                continue

            if self.load_images:
                try:
                    image = self.read_image(frame.image_path)
                except PI3DatasetError as exc:
                    self.errors.append(str(exc))
                    continue
                self._validate_image_shape(frame, image)

            if frame.depth_path is not None:
                self._validate_depth(frame)

        if len(timestamps) >= 2:
            diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
            if (diffs < 0).any():
                self.errors.append("frames.csv timestamps must be monotonically non-decreasing.")

    def _validate_image_shape(self, frame, image):
        if image.ndim != 3 or image.shape[2] != 3:
            self.errors.append(f"Image is not RGB-compatible: {frame.image_path}")
            return
        if self.camera is None:
            return

        height, width = image.shape[:2]
        if width != self.camera.width or height != self.camera.height:
            self.errors.append(
                f"Image shape mismatch for {frame.image_path}: "
                f"expected {self.camera.width}x{self.camera.height}, got {width}x{height}"
            )

    def _validate_depth(self, frame):
        if frame.depth_path.suffix.lower() not in DEPTH_EXTENSIONS:
            self.errors.append(f"Unsupported depth extension: {frame.depth_path}")
        if not frame.depth_path.exists():
            self.errors.append(f"Depth file does not exist: {frame.depth_path}")
            return
        if not frame.depth_path.is_file():
            self.errors.append(f"Depth path is not a file: {frame.depth_path}")
            return

        if not self.load_images:
            return

        try:
            depth = self.read_depth(frame.depth_path)
        except PI3DatasetError as exc:
            self.errors.append(str(exc))
            return
        if self.camera is None:
            return

        height, width = depth.shape[:2]
        if width != self.camera.width or height != self.camera.height:
            self.errors.append(
                f"Depth shape mismatch for {frame.depth_path}: "
                f"expected {self.camera.width}x{self.camera.height}, got {width}x{height}"
            )


def load_pi3_dataset(root, load_images=False, strict=True):
    return PI3Dataset(root=root, load_images=load_images, strict=strict)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and load an experimental PI3-format converted dataset."
    )
    parser.add_argument("root", help="Converted dataset root directory.")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help=(
            "Read images and optional depth maps, then validate dimensions "
            "against camera.json."
        ),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=30,
        help="Maximum number of validation errors to print.",
    )
    args = parser.parse_args()

    dataset = PI3Dataset(args.root, load_images=args.check_images, strict=False)
    print(dataset.summary())
    if dataset.errors:
        print(dataset.format_errors(max_errors=args.max_errors))
        raise SystemExit(1)
    print("PI3 dataset validation passed.")


if __name__ == "__main__":
    main()
