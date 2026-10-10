#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (  # noqa: E402
    BaselineConfig,
    evaluate_baseline,
    fit_incremental_pca,
    train_baseline_models,
    transform_features,
)
from src.bbox import clusters_to_boxes, scale_boxes_xyxy  # noqa: E402
from src.cam import generate_cam, resize_heatmap  # noqa: E402
from src.clustering import dfs_cluster, threshold_heatmap  # noqa: E402
from src.cnn_classifier import (  # noqa: E402
    CNNTrainingConfig,
    PneumoniaDataset,
    build_cnn,
    load_cnn_checkpoint,
    train_cnn,
)
from src.config import CONFIG  # noqa: E402
from src.data import (  # noqa: E402
    assert_no_leakage,
    find_annotation_file,
    parse_boxes_xywh,
    prepare_and_save,
)
from src.metrics import (  # noqa: E402
    classification_metrics,
    localization_match_iou,
    localization_summary,
)
from src.preprocessing import fit_image_normalization_stats  # noqa: E402
from src.rcnn_localizer import (  # noqa: E402
    COCODetectionDataset,
    build_faster_rcnn,
    detection_collate,
    evaluate_rcnn,
    load_rcnn_checkpoint,
    train_faster_rcnn,
)
from src.utils import get_device, save_json, seed_everything  # noqa: E402
from src.visualization import plot_confusion, plot_iou_histogram, plot_training_history  # noqa: E402


SPLIT_DIR = ROOT / "data" / "splits"
METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
BASELINE_DIR = ROOT / "models" / "baselines"
NORMALIZATION_PATH = METRICS_DIR / "normalization.json"
CNN_HISTORY_PATH = METRICS_DIR / "cnn_training_history.json"
CAM_THRESHOLD_PATH = METRICS_DIR / "selected_cam_threshold.json"


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [SPLIT_DIR / name for name in ("all_metadata.csv", "train.csv", "val.csv", "test.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared COCO splits are missing. Run --prepare-data first. Missing: "
            + ", ".join(missing)
        )
    all_rows, train, val, test = [pd.read_csv(path) for path in required]
    for frame in (all_rows, train, val, test):
        frame["target"] = frame["target"].astype(int)
        frame["image_id"] = frame["image_id"].astype(str)
    assert_no_leakage(all_rows)
    if len(all_rows) != len(train) + len(val) + len(test):
        raise RuntimeError("Saved split files do not cover all metadata records exactly once.")
    return all_rows, train, val, test


def _save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in ("ground_truth_boxes", "predicted_boxes"):
        if column in frame:
            frame[column] = frame[column].map(json.dumps)
    frame.to_csv(path, index=False)


def _load_cnn_and_stats(device: torch.device):
    if not CONFIG.model_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {CONFIG.model_path}; run --train-cnn first.")
    if not NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(
            f"Training-split normalization file not found: {NORMALIZATION_PATH}; run --train-cnn first."
        )
    stats = json.loads(NORMALIZATION_PATH.read_text(encoding="utf-8"))
    model = load_cnn_checkpoint(
        build_cnn(in_channels=CONFIG.classifier_input_channels),
        CONFIG.model_path,
        device,
    )
    return model, stats


def _ground_truth_xyxy(row: pd.Series) -> list[list[float]]:
    return [
        [x, y, x + width, y + height]
        for x, y, width, height in parse_boxes_xywh(row["boxes_xywh"])
    ]


def _cam_boxes(
    model: torch.nn.Module,
    image: torch.Tensor,
    row: pd.Series,
    threshold: float,
    device: torch.device,
) -> list[list[float]]:
    heatmap = generate_cam(model, image, class_index=1, device=device)
    resized_heatmap = resize_heatmap(heatmap, CONFIG.target_size)
    binary = threshold_heatmap(resized_heatmap, threshold)
    pixel_boxes = clusters_to_boxes(
        dfs_cluster(binary, connectivity=8),
        min_area=CONFIG.min_box_area,
    )
    return scale_boxes_xyxy(
        pixel_boxes,
        CONFIG.target_size,
        (int(row["height"]), int(row["width"])),
    )


def _run_cam(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    stats: dict,
    threshold: float,
    device: torch.device,
    emit_localizations: bool,
) -> list[dict]:
    dataset = PneumoniaDataset(
        frame,
        image_size=tuple(stats["image_size"]),
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        use_segmented=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=CONFIG.num_workers)
    lookup = frame.set_index("image_id", drop=False)
    records = []
    model.eval()
    with torch.no_grad():
        for images, targets, image_ids in loader:
            image_id = str(image_ids[0])
            row = lookup.loc[image_id]
            image = images[0]
            probability = float(torch.softmax(model(images.to(device)), dim=1)[0, 1].item())
            predicted_label = int(probability >= 0.5)
            boxes = []
            if emit_localizations or predicted_label == 1:
                boxes = _cam_boxes(model, image, row, threshold, device)
            ground_truth = _ground_truth_xyxy(row)
            records.append({
                "image_id": image_id,
                "true_label": int(targets[0].item()),
                "predicted_label": predicted_label,
                "pneumonia_probability": probability,
                "ground_truth_boxes": ground_truth,
                "predicted_boxes": boxes,
                "IoU": localization_match_iou(boxes, ground_truth),
            })
    return records


def prepare_data(
    data_dir: Path,
    annotation_file: Path | None,
    image_dir: Path | None,
    seed: int,
) -> None:
    prepare_and_save(
        data_dir=data_dir,
        output_csv=SPLIT_DIR / "all_metadata.csv",
        annotation_file=annotation_file,
        image_dir=image_dir,
        seed=seed,
        split_fractions=(CONFIG.train_fraction, CONFIG.val_fraction, CONFIG.test_fraction),
    )


def train_baselines(seed: int) -> None:
    _, train, val, test = load_splits()
    baseline_config = BaselineConfig(
        image_size=CONFIG.baseline_size,
        use_pca=CONFIG.baseline_use_pca,
        pca_components=min(CONFIG.baseline_pca_components, len(train)),
        batch_size=CONFIG.baseline_pca_batch_size,
        random_state=seed,
    )
    pca = fit_incremental_pca(train["image_path"].tolist(), baseline_config)
    X_train = transform_features(train["image_path"].tolist(), pca, baseline_config)
    X_val = transform_features(val["image_path"].tolist(), pca, baseline_config)
    X_test = transform_features(test["image_path"].tolist(), pca, baseline_config)
    models = train_baseline_models(X_train, train["target"].to_numpy(), baseline_config)
    import joblib

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for index, (name, model) in enumerate(models.items()):
        artifact = BASELINE_DIR / f"{('logistic_regression', 'svm', 'random_forest')[index]}.joblib"
        joblib.dump({"model": model, "pca": pca, "config": asdict(baseline_config)}, artifact)
        results[name] = {
            "validation": evaluate_baseline(model, X_val, val["target"].to_numpy()),
            "test": evaluate_baseline(model, X_test, test["target"].to_numpy()),
            "artifact": str(artifact.relative_to(ROOT)),
        }
        print(
            f"{name}: validation accuracy={results[name]['validation']['accuracy']:.4f}; "
            f"test accuracy={results[name]['test']['accuracy']:.4f}"
        )
    save_json(results, METRICS_DIR / "baselines.json")


def train_classifier(device: torch.device, seed: int) -> None:
    _, train, val, _ = load_splits()
    mean, std = fit_image_normalization_stats(train, image_size=CONFIG.target_size)
    stats = {"mean": mean, "std": std, "image_size": list(CONFIG.target_size), "source_split": "train"}
    save_json(stats, NORMALIZATION_PATH)
    train_dataset = PneumoniaDataset(
        train, CONFIG.target_size, mean, std, use_segmented=False, augment=False
    )
    val_dataset = PneumoniaDataset(
        val, CONFIG.target_size, mean, std, use_segmented=False, augment=False
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True,
        num_workers=CONFIG.num_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers
    )
    model = build_cnn(in_channels=CONFIG.classifier_input_channels)
    training_config = CNNTrainingConfig(
        epochs=CONFIG.epochs,
        learning_rate=CONFIG.learning_rate,
        batch_size=CONFIG.batch_size,
        early_stopping_patience=CONFIG.early_stopping_patience,
    )
    history = train_cnn(model, train_loader, val_loader, device, CONFIG.model_path, training_config)
    save_json(history, CNN_HISTORY_PATH)
    checkpoint = torch.load(CONFIG.model_path, map_location="cpu")
    checkpoint["normalization"] = stats
    checkpoint["input_channels"] = CONFIG.classifier_input_channels
    torch.save(checkpoint, CONFIG.model_path)
    plot_training_history(history, FIGURES_DIR)
    print(f"Saved best validation-loss CNN checkpoint: {CONFIG.model_path}")


def select_cam_threshold(device: torch.device) -> None:
    _, _, val, _ = load_splits()
    model, stats = _load_cnn_and_stats(device)
    results = []
    per_image_results = []
    for threshold in CONFIG.cam_threshold_candidates:
        records = _run_cam(model, val, stats, threshold, device, emit_localizations=True)
        positive_ious = [row["IoU"] for row in records if row["true_label"] == 1]
        if not positive_ious:
            raise ValueError("Validation split contains no pneumonia images for CAM threshold selection.")
        per_image_results.extend(
            [{"threshold": threshold, **record} for record in records]
        )
        results.append({
            "threshold": threshold,
            "mean_iou": float(np.mean(positive_ious)),
            "median_iou": float(np.median(positive_ious)),
            "nonzero_iou_rate": float(np.mean(np.asarray(positive_ious) > 0)),
            "positive_validation_images": len(positive_ious),
        })
        print(f"CAM threshold {threshold:.2f}: validation pneumonia mean IoU={np.mean(positive_ious):.4f}")
    _save_rows(per_image_results, TABLES_DIR / "validation_cam_thresholds.csv")
    selected = max(results, key=lambda item: (item["mean_iou"], -item["threshold"]))
    save_json({"selected_threshold": selected["threshold"], "candidates": results}, CAM_THRESHOLD_PATH)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([item["threshold"] for item in results], [item["mean_iou"] for item in results], marker="o")
    ax.set_xlabel("CAM threshold")
    ax.set_ylabel("Validation mean IoU (pneumonia images)")
    ax.set_title("CAM threshold selection")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "cam_threshold_selection.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Selected CAM threshold using validation only: {selected['threshold']:.2f}")


def evaluate_cam(device: torch.device) -> dict:
    _, _, _, test = load_splits()
    if not CAM_THRESHOLD_PATH.is_file():
        raise FileNotFoundError(
            f"Validation-selected CAM threshold missing at {CAM_THRESHOLD_PATH}; "
            "run --select-cam-threshold first."
        )
    threshold = float(json.loads(CAM_THRESHOLD_PATH.read_text(encoding="utf-8"))["selected_threshold"])
    model, stats = _load_cnn_and_stats(device)
    records = _run_cam(model, test, stats, threshold, device, emit_localizations=False)
    y_true = [row["true_label"] for row in records]
    y_pred = [row["predicted_label"] for row in records]
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(localization_summary([row["IoU"] for row in records]))
    metrics["cam_threshold"] = threshold
    metrics["localization_protocol"] = "greedy one-to-one IoU, per-image; empty GT or prediction scores 0"
    _save_rows(records, TABLES_DIR / "test_cam_per_image.csv")
    save_json({"metrics": metrics, "per_image": records}, METRICS_DIR / "test_cam.json")
    plot_confusion(y_true, y_pred, save_path=FIGURES_DIR / "test_cam_confusion.png")
    plot_iou_histogram([row["IoU"] for row in records], FIGURES_DIR / "test_cam_iou.png")
    print("Final held-out CAM test metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def train_rcnn(device: torch.device) -> None:
    _, train, val, _ = load_splits()
    train_dataset = COCODetectionDataset(train, CONFIG.rcnn_image_size)
    val_dataset = COCODetectionDataset(val, CONFIG.rcnn_image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG.rcnn_batch_size, shuffle=True,
        num_workers=CONFIG.num_workers, collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=CONFIG.num_workers, collate_fn=detection_collate,
    )
    model = build_faster_rcnn(image_size=CONFIG.rcnn_image_size[0])
    history = train_faster_rcnn(
        model, train_loader, val_loader, device, CONFIG.rcnn_weights_path,
        epochs=CONFIG.rcnn_epochs, lr=CONFIG.learning_rate,
    )
    save_json(history, METRICS_DIR / "rcnn_training_history.json")


def evaluate_rcnn_test(device: torch.device) -> dict:
    _, _, _, test = load_splits()
    if not CONFIG.rcnn_weights_path.is_file():
        raise FileNotFoundError(
            f"Faster R-CNN checkpoint missing at {CONFIG.rcnn_weights_path}; run --train-rcnn first."
        )
    test_dataset = COCODetectionDataset(test, CONFIG.rcnn_image_size)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=CONFIG.num_workers,
        collate_fn=detection_collate,
    )
    model = load_rcnn_checkpoint(
        build_faster_rcnn(image_size=CONFIG.rcnn_image_size[0]),
        CONFIG.rcnn_weights_path,
        device,
    )
    output = evaluate_rcnn(model, test_loader, device)
    rows = []
    true_labels = []
    predicted_labels = []
    for record in output["per_image"]:
        original = test.loc[test["image_id"].astype(str).eq(str(record["image_id"]))].iloc[0]
        predicted_label = int(bool(record["pred_boxes"]))
        true_label = int(original["target"])
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        rows.append({
            "image_id": str(record["image_id"]),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "pneumonia_probability": record["pneumonia_probability"],
            "ground_truth_boxes": record["gt_boxes"],
            "predicted_boxes": record["pred_boxes"],
            "IoU": record["iou"],
        })
    metrics = classification_metrics(true_labels, predicted_labels)
    metrics.update(localization_summary([row["IoU"] for row in rows]))
    metrics["localization_protocol"] = "greedy one-to-one IoU, per-image; empty GT or prediction scores 0"
    _save_rows(rows, TABLES_DIR / "test_rcnn_per_image.csv")
    save_json({"metrics": metrics, "per_image": rows}, METRICS_DIR / "test_rcnn.json")
    print("Final held-out Faster R-CNN test metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def evaluate_baselines_test() -> dict:
    _, train, _, test = load_splits()
    import joblib

    baseline_config = BaselineConfig(
        image_size=CONFIG.baseline_size,
        use_pca=CONFIG.baseline_use_pca,
        pca_components=min(CONFIG.baseline_pca_components, len(train)),
        batch_size=CONFIG.baseline_pca_batch_size,
    )
    paths = {
        "Logistic Regression": "logistic_regression.joblib",
        "SVM": "svm.joblib",
        "Random Forest": "random_forest.joblib",
    }
    results = {}
    for name, filename in paths.items():
        artifact_path = BASELINE_DIR / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Baseline model missing at {artifact_path}; run --train-baselines first.")
        artifact = joblib.load(artifact_path)
        features = transform_features(test["image_path"].tolist(), artifact["pca"], baseline_config)
        results[name] = evaluate_baseline(
            artifact["model"], features, test["target"].to_numpy()
        )
    save_json(results, METRICS_DIR / "baseline_test_metrics.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO pneumonia classification and localization pipeline.")
    actions = (
        "prepare-data", "train-baselines", "train-cnn", "select-cam-threshold",
        "evaluate-cam", "train-rcnn", "evaluate", "generate-results", "all",
    )
    for action in actions:
        parser.add_argument(f"--{action}", action="store_true", help=f"Run the {action.replace('-', ' ')} stage.")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    parser.add_argument("--annotation-file", type=Path, default=CONFIG.annotation_file)
    parser.add_argument("--image-dir", type=Path, default=CONFIG.image_dir)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    args = parser.parse_args()
    selected = [name for name in actions if getattr(args, name.replace("-", "_"))]
    if not selected:
        parser.print_help()
        return
    run_all = args.all
    stages = (
        ["prepare-data", "train-baselines", "train-cnn", "select-cam-threshold", "train-rcnn", "evaluate", "generate-results"]
        if run_all else selected
    )
    seed_everything(args.seed)
    device = get_device(CONFIG.device_preference)
    print(f"Device: {device}")
    data_dir = _rooted(args.data_dir)
    annotation_file = _rooted(args.annotation_file) if args.annotation_file else find_annotation_file(data_dir)
    image_dir = _rooted(args.image_dir) if args.image_dir else None
    for stage in stages:
        print(f"\n=== {stage} ===")
        if stage == "prepare-data":
            prepare_data(data_dir, annotation_file, image_dir, args.seed)
        elif stage == "train-baselines":
            train_baselines(args.seed)
        elif stage == "train-cnn":
            train_classifier(device, args.seed)
        elif stage == "select-cam-threshold":
            select_cam_threshold(device)
        elif stage == "evaluate-cam":
            evaluate_cam(device)
        elif stage == "train-rcnn":
            train_rcnn(device)
        elif stage == "evaluate":
            evaluate_baselines_test()
            evaluate_cam(device)
            evaluate_rcnn_test(device)
        elif stage == "generate-results":
            import subprocess
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "generate_final_artifacts.py")],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
