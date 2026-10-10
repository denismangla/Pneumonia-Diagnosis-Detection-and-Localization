from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from PIL import Image


def generate_cam(model, image_tensor: torch.Tensor, class_index: int = 1, device: torch.device | None = None) -> np.ndarray:
    """Compute CAM = sum_k w_k^c f_k(x), then ReLU and min-max normalization."""
    model.eval()
    device = device or next(model.parameters()).device
    x = image_tensor.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    with torch.no_grad():
        features = model.forward_features(x)
        logits = model.classifier_from_features(features)
    weights = model.classifier.weight[class_index].detach().to(device)
    weighted = (features * weights.view(1, -1, 1, 1)).sum(dim=1)
    cam = torch.relu(weighted)[0].detach().cpu().numpy()
    cam = cam - cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam = cam / max_value
    return cam.astype(np.float32)


def resize_heatmap(cam: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(cam * 255.0, 0, 255).astype(np.uint8))
    resized = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay_heatmap(gray_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
    gray = np.clip(gray_image, 0.0, 1.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    color = cm.get_cmap("jet")(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * color, 0, 1)
