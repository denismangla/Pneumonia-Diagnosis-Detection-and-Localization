#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cam import generate_cam, resize_heatmap, overlay_heatmap  # noqa: E402
from src.clustering import threshold_heatmap, dfs_cluster  # noqa: E402
from src.bbox import clusters_to_boxes, filter_boxes_by_area, filter_boxes_by_2std  # noqa: E402
from src.cnn_classifier import build_cnn, load_cnn_checkpoint  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.preprocessing import load_xray, resize_image, normalize  # noqa: E402
from src.utils import get_device  # noqa: E402


def _load_checkpoint_metadata(model_path: str | Path) -> tuple[int, float, float, tuple[int, int]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    normalization = checkpoint.get("normalization", {})
    input_channels = int(checkpoint.get("input_channels", 1))
    mean = float(normalization.get("mean", CONFIG.normalization_mean))
    std = float(normalization.get("std", CONFIG.normalization_std))
    image_size = tuple(int(x) for x in normalization.get("image_size", CONFIG.target_size))
    return input_channels, mean, std, image_size


def run_inference(
    image_path: str | Path,
    model_path: str | Path,
    threshold: float | None = None,
    output_path: str | Path | None = None,
):
    device = get_device(CONFIG.device_preference)
    input_channels, mean, std, image_size = _load_checkpoint_metadata(model_path)
    image = resize_image(load_xray(image_path), image_size)
    original_norm = normalize(image, mean, std)

    if input_channels == 1:
        tensor = torch.from_numpy(original_norm[None, ...].astype(np.float32)).unsqueeze(0)
    elif input_channels == 2:
        tensor = torch.from_numpy(np.stack([original_norm, original_norm]).astype(np.float32)).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported checkpoint input_channels={input_channels}; expected 1 or 2.")

    model = load_cnn_checkpoint(build_cnn(in_channels=input_channels), model_path, device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob = float(torch.softmax(logits, dim=1)[0, 1].item())
    pred = int(prob >= 0.5)

    if threshold is None:
        threshold = CONFIG.cam_threshold
        threshold_file = ROOT / "results" / "metrics" / "selected_cam_threshold.json"
        if threshold_file.is_file():
            threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["selected_threshold"])

    boxes = []
    heatmap = None
    if pred == 1:
        heatmap_small = generate_cam(model, tensor[0], class_index=1, device=device)
        heatmap = resize_heatmap(heatmap_small, image_size)
        mask = threshold_heatmap(heatmap, threshold)
        clusters = dfs_cluster(mask)
        boxes = filter_boxes_by_2std(
            filter_boxes_by_area(clusters_to_boxes(clusters), CONFIG.min_box_area)
        )

    fig, axes = plt.subplots(1, 3 if pred else 1, figsize=(15, 5) if pred else (5, 5))
    if pred:
        overlay = overlay_heatmap(image, heatmap)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"Prediction: Pneumonia\nP(Pneumonia)={prob:.3f}")
        axes[1].imshow(overlay)
        axes[1].set_title(f"CAM heatmap @ threshold={threshold:.2f}")
        axes[2].imshow(image, cmap="gray")
        from src.visualization import plot_boxes
        plot_boxes(axes[2], boxes, label="Predicted box")
        axes[2].set_title("Predicted localization boxes")
        for ax in axes:
            ax.axis("off")
    else:
        axes.imshow(image, cmap="gray")
        axes.set_title(f"Prediction: No Pneumonia\nP(Pneumonia)={prob:.3f}")
        axes.axis("off")
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
    return {"prediction": pred, "pneumonia_probability": prob, "boxes": boxes, "heatmap": heatmap}


def main():
    parser = argparse.ArgumentParser(description="Run CNN + CAM pneumonia inference on one X-ray.")
    parser.add_argument("image", help="Path to PNG/JPEG/DICOM X-ray")
    parser.add_argument("--cnn", default=str(CONFIG.model_path))
    parser.add_argument("--threshold", type=float, default=None, help="CAM threshold; defaults to validation-selected threshold when available.")
    parser.add_argument("--output", default=None, help="Optional output image path. If omitted, display the figure.")
    args = parser.parse_args()
    result = run_inference(args.image, args.cnn, threshold=args.threshold, output_path=args.output)
    print(json.dumps({
        "prediction": "Pneumonia" if result["prediction"] else "No Pneumonia",
        "pneumonia_probability": result["pneumonia_probability"],
        "predicted_boxes": result["boxes"],
    }, indent=2))


if __name__ == "__main__":
    main()
