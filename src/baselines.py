from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from .preprocessing import load_xray, resize_image
from .metrics import classification_metrics


@dataclass
class BaselineConfig:
    image_size: tuple[int, int] = (128, 128)
    use_pca: bool = True
    pca_components: int = 128
    batch_size: int = 256
    rf_estimators: int = 300
    max_train_samples: int | None = None
    random_state: int = 42


def _iter_flat_batches(paths: Sequence[str | Path], image_size: tuple[int, int], batch_size: int):
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        arrays = [resize_image(load_xray(p), image_size).reshape(-1) for p in batch_paths]
        yield np.asarray(arrays, dtype=np.float32)


def fit_incremental_pca(paths: Sequence[str | Path], config: BaselineConfig) -> IncrementalPCA:
    if not config.use_pca:
        raise ValueError("fit_incremental_pca called with use_pca=False")
    pca = IncrementalPCA(n_components=config.pca_components, batch_size=config.batch_size)
    for batch in _iter_flat_batches(paths, config.image_size, config.batch_size):
        pca.partial_fit(batch)
    return pca


def transform_features(paths: Sequence[str | Path], pca: IncrementalPCA | None, config: BaselineConfig) -> np.ndarray:
    chunks = []
    for batch in _iter_flat_batches(paths, config.image_size, config.batch_size):
        chunks.append(pca.transform(batch) if pca is not None else batch)
    return np.vstack(chunks).astype(np.float32) if chunks else np.empty((0, config.image_size[0] * config.image_size[1]), dtype=np.float32)


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray, config: BaselineConfig) -> dict[str, object]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "SVM": LinearSVC(C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_baseline(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X)
    out = classification_metrics(y, pred)
    out["classification_report"] = classification_report(y, pred, output_dict=True, zero_division=0)
    return out
