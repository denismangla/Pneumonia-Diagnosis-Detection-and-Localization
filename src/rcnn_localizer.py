from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import parse_boxes_xywh
from .metrics import localization_match_iou
from .preprocessing import load_xray, resize_image


class COCODetectionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: tuple[int, int] = (128, 128)) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = resize_image(load_xray(row["image_path"]), self.image_size)
        tensor = torch.from_numpy(np.stack([image, image, image]).astype(np.float32))
        boxes = []
        for box in parse_boxes_xywh(row["boxes_xywh"]):
            x, y, w, h = box
            sx = self.image_size[1] / float(row["width"])
            sy = self.image_size[0] / float(row["height"])
            boxes.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": row["image_id"],
            "original_height": int(row["height"]),
            "original_width": int(row["width"]),
        }
        return tensor, target


# Retain the old public name for callers outside the project runner.
RSNADetectionDataset = COCODetectionDataset


def detection_collate(batch):
    return tuple(zip(*batch))


def build_faster_rcnn(image_size: int = 128):
    try:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    except ImportError as exc:
        raise ImportError("Installed torchvision does not expose a compatible Faster R-CNN builder.") from exc
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=(image_size,),
        max_size=image_size,
    )
    return model


def train_faster_rcnn(model, loader: DataLoader, val_loader: DataLoader, device: torch.device, output_path: str | Path, epochs: int = 10, lr: float = 1e-4, score_threshold: float = 0.4) -> dict[str, list[float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_iou": []}
    best_val_iou = float("-inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        for images, targets in loader:
            images = [image.to(device) for image in images]
            clean_targets = [
                {key: target[key].to(device) for key in ("boxes", "labels")}
                for target in targets
            ]
            loss_dict = model(images, clean_targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate_rcnn(model, val_loader, device, score_threshold=score_threshold)
        val_iou = float(val_metrics["mean_iou"])
        history["train_loss"].append(mean_loss)
        history["val_iou"].append(val_iou)
        print(f"R-CNN epoch {epoch + 1:02d}: train_loss={mean_loss:.4f} val_mean_iou={val_iou:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val_iou": val_iou}, output_path)
    return history


def load_rcnn_checkpoint(model, path: str | Path, device: torch.device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"R-CNN checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    return model.to(device).eval()


def predict_boxes(model, image_tensor: torch.Tensor, device: torch.device, score_threshold: float = 0.4) -> list[list[float]]:
    boxes, _ = predict_boxes_and_probability(
        model, image_tensor, device, score_threshold=score_threshold
    )
    return boxes


def predict_boxes_and_probability(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.4,
) -> tuple[list[list[float]], float]:
    model.eval()
    with torch.no_grad():
        result = model([image_tensor.to(device)])[0]
    scores = result["scores"].detach().cpu().numpy()
    labels = result["labels"].detach().cpu().numpy()
    class_indices = labels == 1
    pneumonia_scores = scores[class_indices]
    probability = float(pneumonia_scores.max()) if pneumonia_scores.size else 0.0
    keep = class_indices & (scores >= score_threshold)
    return result["boxes"].detach().cpu().numpy()[keep].tolist(), probability


def evaluate_rcnn(model, loader: DataLoader, device: torch.device, score_threshold: float = 0.4) -> dict:
    model.eval()
    per_image = []
    for images, targets in loader:
        for image, target in zip(images, targets):
            pred_boxes, pneumonia_probability = predict_boxes_and_probability(
                model, image, device, score_threshold=score_threshold
            )
            original_height = int(target["original_height"])
            original_width = int(target["original_width"])
            resized_height, resized_width = image.shape[-2:]
            sx = original_width / resized_width
            sy = original_height / resized_height
            pred_boxes = [
                [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                for box in pred_boxes
            ]
            gt_boxes = [
                [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                for box in target["boxes"].tolist()
            ]
            iou = localization_match_iou(pred_boxes, gt_boxes)
            per_image.append({
                "image_id": target["image_id"],
                "pneumonia_probability": pneumonia_probability,
                "pred_boxes": pred_boxes,
                "gt_boxes": gt_boxes,
                "iou": iou,
            })
    values = [row["iou"] for row in per_image]
    return {
        "per_image": per_image,
        "mean_iou": float(np.mean(values)) if values else float("nan"),
        "median_iou": float(np.median(values)) if values else float("nan"),
        "nonzero_iou_rate": float(np.mean(np.asarray(values) > 0)) if values else float("nan"),
    }
