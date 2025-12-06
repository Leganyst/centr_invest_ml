"""Модели и sklearn-совместимая обёртка для нейросетевого классификатора."""

from __future__ import annotations

import logging
import random
from typing import Sequence, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TransactionNet(nn.Module):
    """Небольшая MLP-сеть для классификации транзакций."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-совместимый классификатор на PyTorch."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        batch_size: int = 256,
        lr: float = 1e-3,
        max_epochs: int = 30,
        weight_decay: float = 1e-4,
        random_state: int | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # ---- sklearn API ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchNNClassifier":
        X_np = self._ensure_2d_float(np.asarray(X))
        y_arr = np.asarray(y)
        if X_np.shape[0] != y_arr.shape[0]:
            raise ValueError("X и y должны иметь одинаковое количество строк.")

        self._set_random_seed()
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_arr)
        self.classes_ = self._label_encoder.classes_
        num_classes = len(self.classes_)
        if num_classes < 2:
            raise ValueError("Нейросети требуется минимум 2 класса.")

        input_dim = X_np.shape[1]
        self._model = TransactionNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

        device = self._get_device()
        self._model.to(device)
        class_weights = self._compute_class_weights(y_encoded, num_classes)
        weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_encoded.astype(np.int64)),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._loss_history: list[float] = []
        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_size_actual = batch_x.size(0)
                epoch_loss += loss.item() * batch_size_actual
                total_samples += batch_size_actual

            avg_loss = epoch_loss / max(total_samples, 1)
            self._loss_history.append(avg_loss)
            if self.verbose:
                logger.info("Epoch %d/%d, loss=%.4f", epoch + 1, self.max_epochs, avg_loss)

        # Храним модель на CPU для сериализации и инференса
        self._model.eval()
        self._model.to(torch.device("cpu"))
        self._inference_device = torch.device("cpu")
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        pred_idx = np.argmax(logits, axis=1)
        return self._label_encoder.inverse_transform(pred_idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        sums = exp_scores.sum(axis=1, keepdims=True)
        return exp_scores / np.clip(sums, a_min=1e-12, a_max=None)

    # ---- internal helpers ----
    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_model", "_label_encoder"])
        X_np = self._ensure_2d_float(np.asarray(X))
        dataset = TensorDataset(torch.from_numpy(X_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        outputs: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self._inference_device)
                logits = self._model(batch_x).cpu().numpy()
                outputs.append(logits)
        if not outputs:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        return np.vstack(outputs)

    def _compute_class_weights(self, y_encoded: np.ndarray, num_classes: int) -> np.ndarray:
        total = len(y_encoded)
        counts = np.bincount(y_encoded, minlength=num_classes)
        weights = np.zeros(num_classes, dtype=np.float64)
        for idx, count in enumerate(counts):
            if count == 0:
                weights[idx] = 0.0
            else:
                weights[idx] = total / (num_classes * count)
        return weights

    def _ensure_2d_float(self, array: np.ndarray) -> np.ndarray:
        if array.ndim != 2:
            raise ValueError("Ожидается двумерный массив признаков.")
        return array.astype(np.float32, copy=False)

    def _get_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_random_seed(self) -> None:
        if self.random_state is None:
            return
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
