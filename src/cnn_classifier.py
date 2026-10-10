from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics, classification_report_dict
from .preprocessing import apply_identical_geometric_transform, fit_image_normalization_stats, load_xray, normalize, resize_image
from .segmentation import segment_lung, UNet


class TenLayerCAMCNN(nn.Module):
    """10-convolution-layer network ending in GAP + one fully-connected layer."""
    def __init__(self, in_channels: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        blocks = []
        channels = [16, 16, 32, 32, 64, 64, 96, 96, 128, 128]
        current = in_channels
        for idx, out_channels in enumerate(channels):
            blocks.append(nn.Conv2d(current, out_channels, kernel_size=3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            if idx in {1, 3, 5, 7}:
                blocks.append(nn.MaxPool2d(2))
            current = out_channels
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def classifier_from_features(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(features).flatten(1)
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_from_features(self.forward_features(x))


@dataclass
class CNNTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 32
    early_stopping_patience: int = 5
    threshold: float = 0.5


class PneumoniaDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: tuple[int, int] = (128, 128),
        mean: float | None = None,
        std: float | None = None,
        use_segmented: bool = False,
        segmentation_model: UNet | None = None,
        segmentation_device: torch.device | None = None,
        segmentation_threshold: float = 0.5,
        augment: bool = False,
        segmentation_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        if mean is None or std is None:
            train_rows = self.frame
            if "split" in train_rows.columns:
                train_rows = train_rows.loc[train_rows["split"].eq("train")]
            if not train_rows.empty:
                mean, std = fit_image_normalization_stats(
                    train_rows, image_size=self.image_size, split=None
                )
            else:
                raise ValueError(
                    "Normalization statistics must be supplied or computed from training-split images."
                )
        self.mean = float(mean)
        self.std = float(std)
        self.use_segmented = use_segmented
        self.segmentation_model = segmentation_model
        self.segmentation_device = segmentation_device
        self.segmentation_threshold = segmentation_threshold
        self.augment = augment
        self.segmentation_cache = segmentation_cache if segmentation_cache is not None else {}

        if use_segmented and segmentation_model is None:
            raise ValueError("use_segmented=True requires a trained U-Net model.")

    def __len__(self) -> int:
        return len(self.frame)

    def _get_segmented_channel(self, image_path: str | Path) -> np.ndarray:
        key = str(Path(image_path))
        if key in self.segmentation_cache:
            return self.segmentation_cache[key].copy()
        assert self.segmentation_model is not None and self.segmentation_device is not None
        with torch.no_grad():
            _, _, segmented = segment_lung(
                self.segmentation_model,
                image_path,
                size=self.image_size,
                device=self.segmentation_device,
                threshold=self.segmentation_threshold,
            )
        self.segmentation_cache[key] = segmented.astype(np.float32, copy=False)
        return self.segmentation_cache[key].copy()

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        raw = resize_image(load_xray(row["image_path"]), self.image_size)
        segmented = raw.copy()
        if self.use_segmented:
            segmented = self._get_segmented_channel(row["image_path"])
        rng = np.random.default_rng() if self.augment else None
        if self.augment and rng is not None:
            raw, segmented = apply_identical_geometric_transform(raw, segmented, rng=rng)
        original = normalize(raw, self.mean, self.std)
        segmented = normalize(segmented, self.mean, self.std)
        channels = [original, segmented] if self.use_segmented else [original]
        tensor = torch.from_numpy(np.stack(channels).astype(np.float32))
        target = torch.tensor(int(row["target"]), dtype=torch.long)
        return tensor, target, row["image_id"]


def build_cnn(in_channels: int = 2, num_classes: int = 2) -> TenLayerCAMCNN:
    return TenLayerCAMCNN(in_channels=in_channels, num_classes=num_classes)


def train_cnn(
    model: TenLayerCAMCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    config: CNNTrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train with early stopping on validation loss and save the best checkpoint."""
    config = config or CNNTrainingConfig()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val = float("inf")
    patience = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses, y_true, y_pred = [], [], []
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        model.eval()
        val_losses, vy_true, vy_pred = [], [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                val_losses.append(float(loss_fn(logits, targets).item()))
                vy_true.extend(targets.cpu().tolist())
                vy_pred.extend(logits.argmax(dim=1).cpu().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        train_acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else float("nan")
        val_acc = float(np.mean(np.asarray(vy_true) == np.asarray(vy_pred))) if vy_true else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch + 1:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "config": vars(config),
            }, output_path)
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break
    return history


def load_cnn_checkpoint(model: TenLayerCAMCNN, path: str | Path, device: torch.device) -> TenLayerCAMCNN:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    return model.to(device).eval()


def predict_loader(model: TenLayerCAMCNN, loader: DataLoader, device: torch.device):
    model.eval()
    ids: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for images, targets, image_ids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            ids.extend(list(image_ids))
            y_true.extend(targets.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
    return ids, np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def evaluate_classifier(model: TenLayerCAMCNN, loader: DataLoader, device: torch.device) -> dict:
    ids, y_true, y_pred, y_prob = predict_loader(model, loader, device)
    metrics = classification_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report_dict(y_true, y_pred)
    return {"image_ids": ids, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "metrics": metrics}
