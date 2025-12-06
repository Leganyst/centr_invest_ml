#!/usr/bin/env python3
"""Utility for benchmarking transaction classification models on a CSV dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ml import config
from ml import train as train_module

LOGGER = logging.getLogger("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transaction classifier on a CSV dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the CSV file (defaults to config.DATA_PATH).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["simple", "advanced", "ensemble", "catboost"],
        help="Model variant to benchmark (defaults to config.MODEL_TYPE).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Custom test size for train/test split (defaults to config.TEST_SIZE).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random state for splitting/training (defaults to config.RANDOM_STATE).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    data_path = Path(args.data) if args.data else config.DATA_PATH
    model_type = args.model_type or config.MODEL_TYPE
    test_size = args.test_size or config.TEST_SIZE
    random_state = args.random_state or config.RANDOM_STATE

    LOGGER.info("Loading dataset from %s", data_path)
    df = train_module._load_dataset(data_path)
    X = df[list(train_module.RAW_FEATURE_COLUMNS)].copy()
    y = df["Category"].astype(str)

    LOGGER.info("Running 5-fold StratifiedKFold (macro F1)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_model = train_module._build_model_pipeline(model_type)
    cv_scores = cross_val_score(
        cv_model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    LOGGER.info(
        "CV Macro F1: %.3f ± %.3f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    LOGGER.info("Training %s model on train split...", model_type)
    model = train_module._build_model_pipeline(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    LOGGER.info("=== Classification report ===\n%s", classification_report(y_test, y_pred, zero_division=0))
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    LOGGER.info("Macro F1: %.3f", macro_f1)
    LOGGER.info("Balanced Accuracy: %.3f", balanced_acc)

    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    LOGGER.info("Labels order: %s", labels_sorted)
    LOGGER.info("Confusion matrix:\n%s", cm)


if __name__ == "__main__":
    main()
