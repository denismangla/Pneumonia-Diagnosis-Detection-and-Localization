from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECT_DATA_DIR", "train")))
    image_dir: Path | None = field(default_factory=lambda: (
        Path(os.getenv("PROJECT_IMAGE_DIR")) if os.getenv("PROJECT_IMAGE_DIR") else None
    ))
    annotation_file: Path | None = field(default_factory=lambda: (
        Path(os.getenv("PROJECT_ANNOTATION_FILE")) if os.getenv("PROJECT_ANNOTATION_FILE") else None
    ))
    negative_category_file: Path | None = field(default_factory=lambda: (
        Path(os.getenv("PROJECT_NEGATIVE_CATEGORY_FILE")) if os.getenv("PROJECT_NEGATIVE_CATEGORY_FILE") else None
    ))
    target_size: tuple[int, int] = (64, 64)
    split_seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.20
    test_fraction: float = 0.10
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    epochs: int = 20
    early_stopping_patience: int = 5
    cam_threshold: float = 0.45
    cam_threshold_candidates: tuple[float, ...] = (0.20, 0.45, 0.70)
    segmentation_threshold: float = 0.50
    min_box_area: int = 20
    normalization_mean: float = 0.5
    normalization_std: float = 0.25
    device_preference: str = "auto"
    classifier_input_channels: int = 1
    model_path: Path = field(default_factory=lambda: Path("models/cnn_classifier_best.pt"))
    unet_weights_path: Path = field(default_factory=lambda: Path("models/unet_lung_best.pt"))
    rcnn_weights_path: Path = field(default_factory=lambda: Path("models/rcnn_benchmark_best.pt"))
    baseline_size: tuple[int, int] = (32, 32)
    baseline_use_pca: bool = True
    baseline_pca_components: int = 128
    baseline_pca_batch_size: int = 256
    rcnn_image_size: tuple[int, int] = (320, 320)
    rcnn_epochs: int = 15
    rcnn_batch_size: int = 4
    seed: int = 42

    def resolve(self) -> "ProjectConfig":
        root = self.project_root
        for name in ("data_dir", "image_dir", "annotation_file", "negative_category_file", "model_path",
                     "unet_weights_path", "rcnn_weights_path"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value if value.is_absolute() else root / value)
        if self.annotation_file is None:
            candidate = self.data_dir / "_annotations.coco.json"
            if candidate.is_file():
                self.annotation_file = candidate
        return self

    def validate_splits(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


CONFIG = ProjectConfig().resolve()
CONFIG.validate_splits()
