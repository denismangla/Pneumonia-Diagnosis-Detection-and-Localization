import numpy as np
import torch

from src.metrics import box_iou
from src.bbox import clusters_to_boxes, scale_boxes_xyxy
from src.clustering import dfs_cluster, threshold_heatmap
from src.cnn_classifier import build_cnn
from src.cam import generate_cam, resize_heatmap


def test_iou_known_case():
    assert abs(box_iou([0, 0, 10, 10], [5, 5, 15, 15]) - (25 / 175)) < 1e-8


def test_scale_boxes():
    scaled = scale_boxes_xyxy([[10, 20, 30, 40]], (100, 100), (200, 300))
    assert scaled == [[30.0, 40.0, 90.0, 80.0]]


def test_dfs_clusters():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[4:6, 4:6] = 1
    clusters = dfs_cluster(mask)
    assert len(clusters) == 2
    boxes = clusters_to_boxes(clusters, min_area=0)
    assert len(boxes) == 2


def test_cam_shape():
    model = build_cnn(in_channels=2)
    x = torch.randn(2, 128, 128)
    cam = generate_cam(model, x, class_index=1, device=torch.device('cpu'))
    assert cam.ndim == 2
    assert 0 <= float(cam.min()) <= float(cam.max()) <= 1.000001
    resized = resize_heatmap(cam, (128, 128))
    assert resized.shape == (128, 128)
