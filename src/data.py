from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataPreparationReport:
    annotation_file: str
    image_count: int
    annotation_count: int
    pneumonia_annotations: int
    nopneumonia_annotations: int
    pneumonia_images: int
    nopneumonia_images: int
    images_with_both_categories: int
    unknown_images: int
    clipped_boxes: int
    invalid_boxes: int
    missing_files: int


def find_annotation_file(data_dir: str | Path) -> Path:
    """Find the COCO annotation JSON when given either the image dir or project root."""
    root = Path(data_dir)
    candidates = [
        root / "_annotations.coco.json",
        root / "train" / "_annotations.coco.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob("_annotations.coco.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"COCO annotation file not found under {root}; expected "
        f"{root / '_annotations.coco.json'} or {root / 'train' / '_annotations.coco.json'}."
    )


def _validated_box(
    raw_box: Any, width: int, height: int
) -> tuple[list[float], bool]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(f"COCO bbox must be [x, y, width, height], got {raw_box!r}.")
    x, y, box_width, box_height = map(float, raw_box)
    if not np.all(np.isfinite([x, y, box_width, box_height])):
        raise ValueError(f"COCO bbox contains a non-finite value: {raw_box!r}.")
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"COCO bbox must have positive width and height: {raw_box!r}.")

    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"COCO bbox has no positive-area intersection with its {width}x{height} image: {raw_box!r}."
        )
    clipped = not np.allclose([x1, y1, x2, y2], [x, y, x + box_width, y + box_height])
    return [x1, y1, x2 - x1, y2 - y1], bool(clipped)


def prepare_metadata(
    data_dir: str | Path,
    annotation_file: str | Path | None = None,
    image_dir: str | Path | None = None,
    negative_category_file: str | Path | None = None,
) -> tuple[pd.DataFrame, DataPreparationReport]:
    """Create one metadata row per COCO image; pneumonia boxes alone are retained."""
    del negative_category_file  # COCO categories are the sole class source.
    ann_path = Path(annotation_file) if annotation_file else find_annotation_file(data_dir)
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = payload.get("categories")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain categories, images, and annotations arrays.")

    category_names: dict[Any, str] = {}
    for category in categories:
        if "id" not in category or "name" not in category:
            raise ValueError(f"Malformed COCO category: {category!r}.")
        category_names[category["id"]] = str(category["name"]).strip().casefold()
    pneumonia_ids = {key for key, name in category_names.items() if name == "pneumonia"}
    nopneumonia_ids = {key for key, name in category_names.items() if name == "nopneumonia"}

    image_by_id: dict[Any, dict[str, Any]] = {}
    for image in images:
        if not all(key in image for key in ("id", "file_name", "width", "height")):
            raise ValueError(f"Malformed COCO image record: {image!r}.")
        image_id = image["id"]
        if image_id in image_by_id:
            raise ValueError(f"Duplicate COCO image id: {image_id!r}.")
        image_by_id[image_id] = image

    annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    annotation_category_counts: Counter[str] = Counter()
    clipped_boxes = 0
    invalid_boxes = 0
    pneumonia_boxes: dict[Any, list[list[float]]] = defaultdict(list)
    for annotation in annotations:
        if annotation.get("image_id") not in image_by_id:
            raise ValueError(f"Annotation references unknown image id: {annotation.get('image_id')!r}.")
        category_id = annotation.get("category_id")
        if category_id not in category_names:
            raise ValueError(f"Annotation references unknown category id: {category_id!r}.")
        image = image_by_id[annotation["image_id"]]
        try:
            box, was_clipped = _validated_box(
                annotation.get("bbox"), int(image["width"]), int(image["height"])
            )
        except (TypeError, ValueError):
            invalid_boxes += 1
            raise
        clipped_boxes += int(was_clipped)
        name = category_names[category_id]
        annotation_category_counts[name] += 1
        annotations_by_image[annotation["image_id"]].append(annotation)
        if category_id in pneumonia_ids:
            pneumonia_boxes[annotation["image_id"]].append(box)

    root = Path(image_dir) if image_dir is not None else ann_path.parent
    rows: list[dict[str, Any]] = []
    missing_files = 0
    for image_id, image in image_by_id.items():
        image_path = root / str(image["file_name"])
        if not image_path.is_file():
            missing_files += 1
            raise FileNotFoundError(f"COCO image {image_id!r} is missing: {image_path}")
        names = {
            category_names[annotation["category_id"]]
            for annotation in annotations_by_image.get(image_id, [])
        }
        is_pneumonia = bool(names.intersection(
            category_names[key] for key in pneumonia_ids
        ))
        is_nopneumonia = bool(names.intersection(
            category_names[key] for key in nopneumonia_ids
        ))
        if is_pneumonia:
            target: int | None = 1
            class_name = "Pneumonia"
        elif is_nopneumonia:
            target = 0
            class_name = "No Pneumonia"
        else:
            target = None
            class_name = "Unknown"
        # Store paths relative to the project root when the dataset lives under it,
        # so split CSVs remain portable across machines.
        project_root = Path(__file__).resolve().parents[1]
        try:
            stored_path = image_path.resolve().relative_to(project_root.resolve())
            image_path_value = stored_path.as_posix()
        except ValueError:
            image_path_value = str(image_path.resolve())
        rows.append({
            "image_id": image_id,
            "image_path": image_path_value,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "target": target,
            "class_name": class_name,
            "boxes_xywh": json.dumps(pneumonia_boxes.get(image_id, [])),
            "annotation_count": len(annotations_by_image.get(image_id, [])),
            "category_names": json.dumps(sorted(names)),
            "has_both_categories": is_pneumonia and is_nopneumonia,
        })

    metadata = pd.DataFrame(rows)
    metadata["target"] = pd.array(metadata["target"], dtype="Int64")
    report = DataPreparationReport(
        annotation_file=str(ann_path.resolve()),
        image_count=len(image_by_id),
        annotation_count=len(annotations),
        pneumonia_annotations=annotation_category_counts["pneumonia"],
        nopneumonia_annotations=annotation_category_counts["nopneumonia"],
        pneumonia_images=int(metadata["target"].eq(1).sum()),
        nopneumonia_images=int(metadata["target"].eq(0).sum()),
        images_with_both_categories=int(metadata["has_both_categories"].sum()),
        unknown_images=int(metadata["target"].isna().sum()),
        clipped_boxes=clipped_boxes,
        invalid_boxes=invalid_boxes,
        missing_files=missing_files,
    )
    return metadata, report


def split_dataset(
    metadata: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.20,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a seeded stratified image-level split column."""
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Metadata must contain exactly one row per image id.")
    if metadata["target"].isna().any():
        unknown = metadata.loc[metadata["target"].isna(), "image_id"].tolist()
        raise ValueError(f"Cannot stratify images with unknown categories: {unknown[:10]}")

    test_count = round(len(metadata) * test_fraction)
    val_count = round(len(metadata) * val_fraction)
    if test_count <= 0 or val_count <= 0 or test_count + val_count >= len(metadata):
        raise ValueError("Dataset is too small for non-empty train/validation/test splits.")
    train_val, test = train_test_split(
        metadata,
        test_size=test_count,
        stratify=metadata["target"].astype(int),
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_count,
        stratify=train_val["target"].astype(int),
        random_state=seed,
    )
    out = metadata.copy()
    out["split"] = ""
    out.loc[out["image_id"].isin(train["image_id"]), "split"] = "train"
    out.loc[out["image_id"].isin(val["image_id"]), "split"] = "val"
    out.loc[out["image_id"].isin(test["image_id"]), "split"] = "test"
    if out["split"].eq("").any():
        raise RuntimeError("Some image records were not assigned to a split.")
    if len(train) + len(val) + len(test) != len(metadata):
        raise RuntimeError("Split sizes do not account for every image exactly once.")
    assert_no_leakage(out)
    return out


def assert_no_leakage(metadata: pd.DataFrame) -> None:
    if metadata["image_id"].duplicated().any():
        raise AssertionError("An image appears more than once in the split metadata.")
    sets = {
        name: set(metadata.loc[metadata["split"].eq(name), "image_id"])
        for name in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise AssertionError(f"Data leakage between {left} and {right}: {sorted(overlap)[:5]}")
    if set.union(*sets.values()) != set(metadata["image_id"]):
        raise AssertionError("The splits do not represent every metadata image exactly once.")


def parse_boxes_xywh(serialized: Any) -> list[list[float]]:
    if serialized is None or (isinstance(serialized, float) and np.isnan(serialized)):
        return []
    value = json.loads(serialized) if isinstance(serialized, str) else serialized
    return [[float(coordinate) for coordinate in box] for box in value]


def prepare_and_save(
    data_dir: str | Path,
    output_csv: str | Path,
    annotation_file: str | Path | None = None,
    image_dir: str | Path | None = None,
    negative_category_file: str | Path | None = None,
    seed: int = 42,
    split_fractions: tuple[float, float, float] = (0.70, 0.20, 0.10),
) -> DataPreparationReport:
    metadata, report = prepare_metadata(
        data_dir=data_dir,
        annotation_file=annotation_file,
        image_dir=image_dir,
        negative_category_file=negative_category_file,
    )
    metadata = split_dataset(
        metadata, *split_fractions, seed=seed
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_csv, index=False)
    for split, filename in (("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")):
        metadata.loc[metadata["split"].eq(split)].to_csv(output_csv.parent / filename, index=False)
    (output_csv.parent / "data_quality_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    metrics_report = output_csv.parent.parent.parent / "results" / "metrics" / "data_quality_report.json"
    metrics_report.parent.mkdir(parents=True, exist_ok=True)
    metrics_report.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    expected = {
        "train": round(len(metadata) * split_fractions[0]),
        "val": round(len(metadata) * split_fractions[1]),
        "test": len(metadata) - round(len(metadata) * split_fractions[0])
        - round(len(metadata) * split_fractions[1]),
    }
    split_counts = metadata["split"].value_counts().to_dict()
    if split_counts != expected:
        raise RuntimeError(f"Unexpected split sizes: {split_counts}; expected {expected}.")
    print(f"Total images: {report.image_count}")
    print(f"Pneumonia images: {report.pneumonia_images}")
    print(f"No-pneumonia images: {report.nopneumonia_images}")
    print(f"Total annotations: {report.annotation_count}")
    print(f"Pneumonia annotations: {report.pneumonia_annotations}")
    print(f"No-pneumonia annotations: {report.nopneumonia_annotations}")
    print(f"Images with both categories: {report.images_with_both_categories}")
    print(
        f"Invalid/clipped boxes: {report.clipped_boxes} clipped, "
        f"{report.invalid_boxes} invalid"
    )
    print("Exact split counts:")
    print(metadata.groupby(["split", "class_name"], observed=True).size().to_string())
    print(f"Total split rows: {len(metadata)} (expected {report.image_count})")
    return report
