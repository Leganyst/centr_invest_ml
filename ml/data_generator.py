"""Инструменты генерации данных: label propagation и балансировка."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

__all__ = [
    "run_label_propagation",
    "balance_by_oversampling",
    "BalancedTransactionDataset",
    "build_balanced_dataloader",
]


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32, order="C")


def run_label_propagation(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    *,
    confidence_threshold: float = 0.9,
    kernel: str = "rbf",
    gamma: float = 0.25,
    n_neighbors: int = 15,
    alpha: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Расширяет размеченный датасет через semi-supervised LabelSpreading.

    Возвращает X и y, включающие исходные размеченные данные,
    а также псевдо-размеченные примеры с уверенностью выше порога.
    """
    X_l = _as_float32(X_labeled)
    y_l = np.asarray(y_labeled)
    X_u = _as_float32(X_unlabeled)
    if X_u.size == 0:
        return X_l, y_l, np.empty(0, dtype=int), np.empty(0, dtype=y_l.dtype)

    if X_l.shape[1] != X_u.shape[1]:
        raise ValueError("X_labeled и X_unlabeled должны иметь одинаковое число признаков.")

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_l)
    n_labeled = X_l.shape[0]
    n_unlabeled = X_u.shape[0]

    X_all = np.vstack([X_l, X_u])
    y_all = np.concatenate([y_enc, np.full(n_unlabeled, -1, dtype=int)])

    spreading = LabelSpreading(
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors,
        alpha=alpha,
    )
    spreading.fit(X_all, y_all)

    start_idx = n_labeled
    label_dist = spreading.label_distributions_[start_idx:]
    predicted = spreading.transduction_[start_idx:]
    confidence = label_dist.max(axis=1)
    mask = confidence >= confidence_threshold

    if not mask.any():
        logger.info("Label propagation не добавил ни одной записи (порог=%.2f).", confidence_threshold)
        return X_l, y_l, np.empty(0, dtype=int), np.empty(0, dtype=label_encoder.classes_.dtype)

    pseudo_X = X_u[mask]
    pseudo_y = label_encoder.inverse_transform(predicted[mask])
    logger.info(
        "Label propagation: добавлено %d псевдо-меток из %d.",
        pseudo_X.shape[0],
        n_unlabeled,
    )
    X_final = np.vstack([X_l, pseudo_X])
    y_final = np.concatenate([y_l, pseudo_y])
    selected_indices = np.where(mask)[0]
    return _as_float32(X_final), y_final, selected_indices, pseudo_y


def balance_by_oversampling(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_per_class: int | None = None,
    random_state: int | None = None,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Выравнивает количество примеров по классам через oversampling/undersampling.
    """
    rng = np.random.default_rng(random_state)
    X_arr = _as_float32(X)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < 2:
        raise ValueError("Для балансировки требуется минимум два класса.")

    max_count = int(counts.max())
    target = max_count if max_per_class is None else min(max_per_class, max_count)
    indices: list[int] = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y_arr == cls)[0]
        if count >= target:
            chosen = rng.choice(cls_idx, size=target, replace=False)
        else:
            chosen = rng.choice(cls_idx, size=target, replace=True)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    indices = np.asarray(indices, dtype=int)
    X_bal = X_arr[indices]
    y_bal = y_arr[indices]
    if return_indices:
        return X_bal, y_bal, indices
    return X_bal, y_bal


class BalancedTransactionDataset(Dataset):
    """Обёртка над numpy-массивами для PyTorch DataLoader."""

    def __init__(self, X: np.ndarray, y_idx: np.ndarray) -> None:
        if X.shape[0] != y_idx.shape[0]:
            raise ValueError("X и y_idx имеют разное количество строк.")
        self.features = torch.from_numpy(_as_float32(X))
        self.labels = torch.from_numpy(np.asarray(y_idx, dtype=np.int64))

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def build_balanced_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 256,
    random_state: int | None = None,
    max_per_class: int | None = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, LabelEncoder]:
    """
    Готовит DataLoader для PyTorch c балансировкой классов.
    Возвращает (loader, label_encoder).
    """
    label_encoder = LabelEncoder()
    y_idx = label_encoder.fit_transform(np.asarray(y))
    X_bal, y_bal = balance_by_oversampling(
        X,
        y_idx,
        max_per_class=max_per_class,
        random_state=random_state,
    )
    dataset = BalancedTransactionDataset(X_bal, y_bal)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, label_encoder
