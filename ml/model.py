import logging
from pathlib import Path
from typing import Any, Dict, Mapping, TypedDict

import joblib
import pandas as pd
from pandas import DataFrame, Timestamp

from . import config

logger = logging.getLogger(__name__)

__all__ = ["TransactionClassifier", "classifier"]


class TransactionPayload(TypedDict, total=False):
    """Описание структуры словаря транзакции, ожидаемой классификатором."""

    date: str
    withdrawal: float
    deposit: float
    balance: float
    refno: str


class TransactionClassifier:
    """Лёгкая обёртка, предоставляющая методы предсказания для транзакций."""

    def __init__(self, model_path: Path | str | None = None):
        resolved_path = self._resolve_path(model_path, config.MODEL_PATH)
        self.model_path = resolved_path
        payload = self._load_payload(resolved_path)
        self.model = payload["model"]
        self.feature_cols = list(payload.get("feature_cols", []))
        self.input_cols = payload.get("input_cols") or ["Date", "Withdrawal", "Deposit", "Balance", "RefNo"]
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
                f"Model artifact not found at {path}. "
                "Run ml.train.train_model() first."
            ) from exc

    def _prepare_frame(self, tx: Mapping[str, Any]) -> DataFrame:
        timestamp = self._parse_date(tx.get("date"))

        def _safe_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        row = {
            "Date": timestamp,
            "Withdrawal": _safe_float(tx.get("withdrawal")),
            "Deposit": _safe_float(tx.get("deposit")),
            "Balance": _safe_float(tx.get("balance")),
            "RefNo": str(tx.get("refno", "") or ""),
        }
        frame = pd.DataFrame([row])
        return frame[self.input_cols]

    @staticmethod
    def _parse_date(value: Any) -> Timestamp:
        timestamp = pd.to_datetime(value, errors="coerce")
        if not isinstance(timestamp, Timestamp) or pd.isna(timestamp):
            raise ValueError(f"Cannot parse transaction date: {value!r}")
        return timestamp

    def predict_category(self, tx: Mapping[str, Any]) -> str:
        """Предсказывает наиболее вероятную категорию для транзакции."""
        features = self._prepare_frame(tx)
        pred = self.model.predict(features)[0]
        return str(pred)

    def predict_proba(self, tx: Mapping[str, Any]) -> Dict[str, float]:
        """Возвращает распределение вероятностей по категориям."""
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("Underlying model does not provide predict_proba.")

        features = self._prepare_frame(tx)
        proba = self.model.predict_proba(features)[0]
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise RuntimeError("Model does not expose classes_ attribute.")
        return {str(label): float(score) for label, score in zip(classes, proba)}


classifier: TransactionClassifier | None = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_tx: TransactionPayload = {
        "date": "2023-01-03",
        "withdrawal": 100.0,
        "deposit": 0.0,
        "balance": 1500.0,
    }
    clf = TransactionClassifier()  # здесь можно, ты явно запускаешь модель после тренинга
    print("Predicted category:", clf.predict_category(sample_tx))
