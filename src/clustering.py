from __future__ import annotations

from collections import deque
import numpy as np


def threshold_heatmap(heatmap: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError("threshold_heatmap expects a 2-D normalized heatmap.")
    return (heatmap >= threshold).astype(np.uint8)


def dfs_cluster(binary_mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Iterative DFS that returns each connected active pixel cluster as N x 2 [y, x]."""
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be 2-D")
    active = binary_mask.astype(bool)
    visited = np.zeros_like(active, dtype=bool)
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 4:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    clusters: list[np.ndarray] = []
    h, w = active.shape
    for y in range(h):
        for x in range(w):
            if not active[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and active[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            clusters.append(np.asarray(pixels, dtype=np.int32))
    return clusters


def connected_components(binary_mask: np.ndarray) -> list[np.ndarray]:
    """Alias for DFS output retained for comparison with optimized implementations."""
    return dfs_cluster(binary_mask, connectivity=8)
