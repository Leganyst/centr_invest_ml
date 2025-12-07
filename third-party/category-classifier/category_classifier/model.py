import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from pandas import DataFrame, Timestamp

from category_classifier import config
from category_classifier.schemas import Category, Transaction

logger = logging.getLogger(__name__)


class TransactionClassifier:
    """Лёгкая обёртка, предоставляющая методы предсказания для транзакций."""

    def __init__(self, model_path: Path | str | None = None):
        resolved_path = self._resolve_path(model_path, config.MODEL_PATH)
        self.model_path = resolved_path
        payload = self._load_payload(resolved_path)
        self.model = payload["model"]
        self.feature_cols = list(payload.get("feature_cols", []))
        self.input_cols = payload.get("input_cols") or [
            "Date",
            "Withdrawal",
            "Deposit",
            "Balance",
        ]
        logger.info("Transaction classifier loaded from %s", self.model_path)

    @staticmethod
    def _resolve_path(user_path: Path | str | None, default_path: Path) -> Path:
        return Path(user_path) if user_path is not None else default_path

    @staticmethod
    def _load_payload(path: Path) -> Dict[str, Any]:
        try:
            return joblib.load(path)
        except FileNotFoundError as exc:  # pragma: no cover - explicit message
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run category_classifier.train.train_model() first."
            ) from exc

    def _prepare_frame(self, tx: Transaction) -> DataFrame:
        timestamp = self._parse_date(tx.date)
        row = {
            "Date": timestamp,
            "Withdrawal": tx.withdrawal,
            "Deposit": tx.deposit,
            "Balance": tx.balance,
        }
        frame = pd.DataFrame([row])
        return frame[self.input_cols]

    @staticmethod
    def _parse_date(value: Any) -> Timestamp:
        timestamp = pd.to_datetime(value, errors="coerce")
        if not isinstance(timestamp, Timestamp) or pd.isna(timestamp):
            raise ValueError(f"Cannot parse transaction date: {value!r}")
        return timestamp

    def predict_category(self, tx: Transaction) -> Category:
        """Предсказывает наиболее вероятную категорию для транзакции."""
        features = self._prepare_frame(tx)
        pred = self.model.predict(features)[0]
        return Category(pred)

    def predict_proba(self, tx: Transaction) -> dict[Category, float]:
        """Возвращает распределение вероятностей по категориям."""
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                "Underlying model does not provide predict_proba."
            )

        features = self._prepare_frame(tx)
        proba = self.model.predict_proba(features)[0]
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise RuntimeError("Model does not expose classes_ attribute.")
        return {Category(label): float(score) for label, score in zip(classes, proba)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_tx: Transaction = Transaction(
        date="2023-01-03",
        withdrawal=100.0,
        deposit=0.0,
        balance=1500.0,
    )
    clf = TransactionClassifier()
    print("Predicted category:", clf.predict_category(sample_tx))
