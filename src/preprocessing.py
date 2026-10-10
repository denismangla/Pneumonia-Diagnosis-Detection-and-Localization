from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch


def load_xray(path: str | Path) -> np.ndarray:
    """Load a grayscale X-ray as float32 in [0, 1]. DICOM support is optional.

    Relative paths are resolved against the project root so split CSVs remain portable.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = Path(__file__).resolve().parents[1] / path
        if candidate.exists():
            path = candidate
    if path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("DICOM input requires the optional 'pydicom' dependency.") from exc
        ds = pydicom.dcmread(str(path))
        image = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = image.max() - image
    else:
        with Image.open(path) as img:
            image = np.asarray(img.convert("L"), dtype=np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image.astype(np.float32)


def resize_image(image: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    resized = pil.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def normalize(image: np.ndarray, mean: float, std: float, eps: float = 1e-6) -> np.ndarray:
    return (image.astype(np.float32) - float(mean)) / max(float(std), eps)


def preprocess_image(
    path: str | Path,
    size: tuple[int, int] = (128, 128),
    mean: float = 0.5,
    std: float = 0.25,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Load, resize, optionally augment, normalize and return 1xHxW tensor."""
    image = load_xray(path)
    image = resize_image(image, size)
    if augment:
        rng = rng or np.random.default_rng()
        if rng.random() < 0.5:
            image = np.fliplr(image).copy()
        if rng.random() < 0.2:
            image = np.clip(image * float(rng.uniform(0.9, 1.1)), 0, 1)
    image = normalize(image, mean, std)
    return torch.from_numpy(image).unsqueeze(0).float()


def compute_dataset_mean_std(paths: Iterable[str | Path], size: tuple[int, int] = (128, 128), max_images: int | None = None) -> tuple[float, float]:
    """Estimate global training-set mean/std after resizing."""
    total = 0.0
    total_sq = 0.0
    count = 0
    for index, path in enumerate(paths):
        if max_images is not None and index >= max_images:
            break
        image = resize_image(load_xray(path), size)
        values = image.reshape(-1).astype(np.float64)
        total += float(values.sum())
        total_sq += float(np.square(values).sum())
        count += values.size
    if count == 0:
        raise ValueError("No images available to estimate normalization statistics.")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance))


def fit_image_normalization_stats(frame, image_size: tuple[int, int] = (128, 128), split: str | None = None, max_images: int | None = None) -> tuple[float, float]:
    """Compute and cache the mean/std from the selected training split once."""
    rows = frame
    if split is not None:
        rows = rows.loc[rows.get("split", "") == split]
    paths = [Path(row["image_path"]) for _, row in rows.iterrows() if str(row.get("image_path", "")).strip()]
    return compute_dataset_mean_std(paths, size=image_size, max_images=max_images)


def apply_identical_geometric_transform(image_a: np.ndarray, image_b: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random geometric transform to both channels so the pair stays aligned."""
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        image_a = np.fliplr(image_a).copy()
        image_b = np.fliplr(image_b).copy()
    if rng.random() < 0.5:
        k = int(rng.integers(0, 4))
        image_a = np.rot90(image_a, k=k)
        image_b = np.rot90(image_b, k=k)
    return image_a, image_b


def scale_box_xywh(box: list[float], original_size: tuple[int, int], target_size: tuple[int, int]) -> list[float]:
    """Scale an x/y/width/height box from original HxW to target HxW."""
    x, y, w, h = map(float, box)
    oh, ow = original_size
    th, tw = target_size
    sx, sy = tw / ow, th / oh
    return [x * sx, y * sy, w * sx, h * sy]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def make_two_channel(original: np.ndarray, segmented: np.ndarray) -> torch.Tensor:
    """Create a 2xHxW float tensor from two already-resized float images."""
    if original.shape != segmented.shape:
        raise ValueError("Original and segmented images must have the same shape.")
    return torch.from_numpy(np.stack([original, segmented]).astype(np.float32))
