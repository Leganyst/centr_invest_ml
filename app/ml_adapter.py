"""Утилиты для работы с TransactionClassifier внутри FastAPI."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from fastapi import HTTPException, status
from ml.model import TransactionClassifier
from ml.train import _parse_date_series

from .config import get_settings

LOGGER = logging.getLogger(__name__)

_classifier: TransactionClassifier | None = None
_model_error: Exception | None = None


def load_classifier() -> TransactionClassifier:
    """Загружает экземпляр классификатора по пути из настроек."""
    global _classifier, _model_error
    settings = get_settings()
    if _classifier is not None:
        return _classifier
    try:
        LOGGER.info("Загружаю TransactionClassifier из %s", settings.model_path)
        _classifier = TransactionClassifier(model_path=settings.model_path)
        _model_error = None
    except Exception as exc:  # pragma: no cover - startup error
        _model_error = exc
        LOGGER.exception("Не удалось загрузить TransactionClassifier: %s", exc)
    if _classifier is None:
        raise RuntimeError("Classifier is not loaded")
    return _classifier


def get_classifier_or_503() -> TransactionClassifier:
    """Возвращает классификатор или бросает HTTP 503, если он недоступен."""
    if _classifier is not None:
        return _classifier
    try:
        return load_classifier()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MODEL_NOT_READY: {exc}",
        )


def _normalize_label(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _normalize_label(value[0])
    if hasattr(value, "item"):
        try:
            return _normalize_label(value.item())
        except Exception:
            pass
    return str(value)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_main, _ = _parse_date_series(df.get("Date"), "Date")
    if "Date.1" in df.columns:
        date_alt, _ = _parse_date_series(df.get("Date.1"), "Date.1")
    else:
        date_alt = pd.Series(pd.NaT, index=df.index)
    df["Date"] = date_main.fillna(date_alt)
    df = df.dropna(subset=["Date"])

    for col in ["Withdrawal", "Deposit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Withdrawal", "Deposit", "Balance"])
    return df


def predict_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[dict]]:
    """Запускает модель на dataframe с колонками Date/Withdrawal/Deposit/Balance."""
    if df.empty:
        return df, [], []
    df_prepared = _prepare_dataframe(df)
    if df_prepared.empty:
        return df_prepared, [], []
    classifier = get_classifier_or_503()
    model = classifier.model
    features = df_prepared[["Date", "Withdrawal", "Deposit", "Balance"]].copy()
    raw_labels = model.predict(features)
    labels = [_normalize_label(label) for label in raw_labels]
    proba = model.predict_proba(features)
    classes = list(model.classes_)
    probabilities = [
        {str(label): float(score) for label, score in zip(classes, row)} for row in proba
    ]
    return df_prepared, list(map(str, labels)), probabilities
