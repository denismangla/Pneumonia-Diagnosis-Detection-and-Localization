from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_xray, resize_image


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net suitable for 128x128 grayscale X-rays."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


class LungMaskDataset(Dataset):
    """Paired image/mask dataset for training U-Net.

    Mask filenames must share the image stem. This dataset is intentionally generic
    because the RSNA pneumonia challenge does not provide lung segmentation masks.
    """
    def __init__(self, image_paths: Sequence[str | Path], mask_dir: str | Path, size: tuple[int, int] = (128, 128)) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_dir = Path(mask_dir)
        self.size = size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing lung mask for {image_path.name}: {mask_path}")
        image = resize_image(load_xray(image_path), self.size)
        mask = resize_image(load_xray(mask_path), self.size)
        mask = (mask > 0.5).astype(np.float32)
        return torch.from_numpy(image[None]).float(), torch.from_numpy(mask[None]).float()


@dataclass
class UNetTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 16
    early_stopping_patience: int = 5


def train_unet(
    model: UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    config: UNetTrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train U-Net on externally supplied lung masks; saves best checkpoint."""
    config = config or UNetTrainingConfig()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0

    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                val_losses.append(float(loss_fn(model(images), masks).item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "config": vars(config)}, output_path)
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    return history


def load_unet_weights(model: UNet, weights_path: str | Path, device: torch.device) -> UNet:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"U-Net weights not found at {path}. BLOCKED — REQUIREMENT NEEDED: supply a trained U-Net "
            "checkpoint or train this architecture on paired lung masks using train_unet()."
        )
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return model.to(device).eval()


def generate_lung_mask(model: UNet, image: np.ndarray, device: torch.device, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return probability map and binary mask for one resized float image."""
    tensor = torch.from_numpy(image[None, None]).float().to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (probability >= threshold).astype(np.float32)
    return probability, mask


def segment_lung(model: UNet, image_path: str | Path, size: tuple[int, int], device: torch.device, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and resize an X-ray, infer a lung mask, and multiply image by mask."""
    image = resize_image(load_xray(image_path), size)
    probability, mask = generate_lung_mask(model, image, device, threshold=threshold)
    segmented = image * mask
    return probability, mask, segmented
