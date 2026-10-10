from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_boxes(ax, boxes: Sequence[Sequence[float]], label: str = "box", linestyle: str = "-"):
    from matplotlib.patches import Rectangle
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, linestyle=linestyle, label=label)
        ax.add_patch(rect)


def plot_image_with_boxes(image: np.ndarray, boxes: Sequence[Sequence[float]], title: str, save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    plot_boxes(ax, boxes)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_cam_overlay(overlay: np.ndarray, boxes: Sequence[Sequence[float]] = (), title: str = "CAM", save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    plot_boxes(ax, boxes)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_confusion(y_true, y_pred, labels=("No Pneumonia", "Pneumonia"), save_path=None):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_training_history(history: dict[str, list[float]], save_dir: str | Path | None = None):
    figs = []
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history.get("train_accuracy", []), label="Train Accuracy")
    ax1.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_title("Training vs Validation Accuracy")
    fig1.tight_layout()
    figs.append(fig1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(history.get("train_loss", []), label="Train Loss")
    ax2.plot(history.get("val_loss", []), label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Training vs Validation Loss")
    fig2.tight_layout()
    figs.append(fig2)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_dir / "training_validation_accuracy.png", dpi=160, bbox_inches="tight")
        fig2.savefig(save_dir / "training_validation_loss.png", dpi=160, bbox_inches="tight")
    return figs


def plot_iou_histogram(iou_values: Sequence[float], save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(iou_values), bins=20)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("Localization IoU Distribution")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax
