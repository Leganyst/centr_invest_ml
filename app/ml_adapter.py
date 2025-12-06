"""Helpers for working with the TransactionClassifier inside FastAPI."""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import pandas as pd
from fastapi import HTTPException, status
from ml.model import TransactionClassifier

from .config import get_settings

LOGGER = logging.getLogger(__name__)

_classifier: TransactionClassifier | None = None
_model_error: Exception | None = None


def load_classifier() -> TransactionClassifier:
    """Load classifier instance using configured path."""
    global _classifier, _model_error
    settings = get_settings()
    if _classifier is not None:
        return _classifier
    try:
        LOGGER.info("Loading TransactionClassifier from %s", settings.model_path)
        _classifier = TransactionClassifier(model_path=settings.model_path)
        _model_error = None
    except Exception as exc:  # pragma: no cover - startup error
        _model_error = exc
        LOGGER.exception("Failed to load TransactionClassifier: %s", exc)
    if _classifier is None:
        raise RuntimeError("Classifier is not loaded")
    return _classifier


def get_classifier_or_503() -> TransactionClassifier:
    """Return classifier instance or raise HTTP 503 if unavailable."""
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


def predict_dataframe(df: pd.DataFrame) -> Tuple[List[str], List[dict]]:
    """Run predictions on a dataframe containing Date/Withdrawal/Deposit/Balance columns."""
    if df.empty:
        return [], []
    classifier = get_classifier_or_503()
    model = classifier.model
    features = df[["Date", "Withdrawal", "Deposit", "Balance"]].copy()
    raw_labels = model.predict(features)
    labels = [_normalize_label(label) for label in raw_labels]
    proba = model.predict_proba(features)
    classes = list(model.classes_)
    probabilities = [
        {str(label): float(score) for label, score in zip(classes, row)} for row in proba
    ]
    return list(map(str, labels)), probabilities
