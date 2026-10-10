import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import parse_boxes_xywh, prepare_metadata, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_coco_counts_and_clipped_box():
    metadata, report = prepare_metadata(PROJECT_ROOT / "train")

    assert report.image_count == 200
    assert report.annotation_count == 410
    assert report.pneumonia_annotations == 197
    assert report.nopneumonia_annotations == 213
    assert report.pneumonia_images == 120
    assert report.nopneumonia_images == 80
    assert report.images_with_both_categories == 52
    assert report.clipped_boxes == 1
    assert report.invalid_boxes == 0
    assert len(metadata) == 200
    assert sum(len(parse_boxes_xywh(value)) for value in metadata["boxes_xywh"]) == 197


def test_dynamic_category_mapping_mixed_labels_and_bbox_clipping(tmp_path):
    (tmp_path / "mixed.jpg").touch()
    (tmp_path / "negative.jpg").touch()
    (tmp_path / "unknown.jpg").touch()
    document = {
        "categories": [
            {"id": 73, "name": "pneumonia"},
            {"id": 12, "name": "nopneumonia"},
            {"id": 99, "name": "other"},
        ],
        "images": [
            {"id": "mixed", "file_name": "mixed.jpg", "width": 10, "height": 8},
            {"id": "negative", "file_name": "negative.jpg", "width": 10, "height": 8},
            {"id": "unknown", "file_name": "unknown.jpg", "width": 10, "height": 8},
        ],
        "annotations": [
            {"id": 1, "image_id": "mixed", "category_id": 73, "bbox": [9.91, 2, 1, 3]},
            {"id": 2, "image_id": "mixed", "category_id": 12, "bbox": [1, 1, 2, 2]},
            {"id": 3, "image_id": "negative", "category_id": 12, "bbox": [2, 2, 3, 3]},
        ],
    }
    annotation_path = tmp_path / "_annotations.coco.json"
    annotation_path.write_text(json.dumps(document), encoding="utf-8")

    metadata, report = prepare_metadata(tmp_path)
    mixed = metadata.set_index("image_id").loc["mixed"]
    negative = metadata.set_index("image_id").loc["negative"]
    unknown = metadata.set_index("image_id").loc["unknown"]

    assert len(metadata) == 3
    assert mixed["target"] == 1
    assert mixed["class_name"] == "Pneumonia"
    assert mixed["has_both_categories"]
    clipped_box = parse_boxes_xywh(mixed["boxes_xywh"])[0]
    assert clipped_box[:2] == [9.91, 2.0]
    assert clipped_box[2] == pytest.approx(0.09)
    assert clipped_box[3] == 3.0
    assert negative["target"] == 0
    assert parse_boxes_xywh(negative["boxes_xywh"]) == []
    assert pd.isna(unknown["target"])
    assert unknown["class_name"] == "Unknown"
    assert report.clipped_boxes == 1
    assert report.images_with_both_categories == 1


def test_stratified_split_is_disjoint_and_complete():
    metadata, _ = prepare_metadata(PROJECT_ROOT / "train")
    split = split_dataset(metadata, seed=42)

    assert split["split"].value_counts().to_dict() == {"train": 140, "val": 40, "test": 20}
    assert split["image_id"].nunique() == 200
    assert split.groupby(["split", "target"], observed=True).size().to_dict() == {
        ("test", 0): 8,
        ("test", 1): 12,
        ("train", 0): 56,
        ("train", 1): 84,
        ("val", 0): 16,
        ("val", 1): 24,
    }
