from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np


def clusters_to_boxes(clusters: Sequence[np.ndarray], min_area: int = 20) -> list[list[float]]:
    boxes: list[list[float]] = []
    for cluster in clusters:
        if cluster.size == 0:
            continue
        ys = cluster[:, 0]
        xs = cluster[:, 1]
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            boxes.append([x1, y1, x2, y2])
    return boxes


def scale_boxes_xyxy(boxes: Sequence[Sequence[float]], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[list[float]]:
    fh, fw = from_size
    th, tw = to_size
    sx, sy = tw / fw, th / fh
    return [[float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)] for x1, y1, x2, y2 in boxes]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def filter_boxes_by_area(boxes: Sequence[Sequence[float]], min_area: float = 20.0) -> list[list[float]]:
    return [list(map(float, box)) for box in boxes if box_area(box) >= min_area]


def filter_boxes_by_2std(boxes: Sequence[Sequence[float]], enabled: bool = True, z: float = 2.0) -> list[list[float]]:
    boxes = [list(map(float, box)) for box in boxes]
    if not enabled or len(boxes) <= 2:
        return boxes
    arr = np.asarray(boxes, dtype=float)
    centers = np.column_stack([(arr[:, 0] + arr[:, 2]) / 2.0, (arr[:, 1] + arr[:, 3]) / 2.0])
    areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    stats = [(centers[:, 0], centers[:, 0].mean(), centers[:, 0].std()),
             (centers[:, 1], centers[:, 1].mean(), centers[:, 1].std()),
             (areas, areas.mean(), areas.std())]
    keep = np.ones(len(boxes), dtype=bool)
    for values, mean, std in stats:
        if std > 0:
            keep &= np.abs(values - mean) <= z * std
    return [boxes[i] for i in range(len(boxes)) if keep[i]]


def merge_overlapping_boxes(boxes: Sequence[Sequence[float]], iou_threshold: float = 0.5) -> list[list[float]]:
    """Greedy box merging without external CV dependencies."""
    from .metrics import box_iou
    pending = [list(map(float, b)) for b in boxes]
    merged: list[list[float]] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            rest = []
            for other in pending:
                if box_iou(current, other) >= iou_threshold:
                    current = [min(current[0], other[0]), min(current[1], other[1]), max(current[2], other[2]), max(current[3], other[3])]
                    changed = True
                else:
                    rest.append(other)
            pending = rest
        merged.append(current)
    return merged
