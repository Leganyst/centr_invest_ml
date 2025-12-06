"""
Обучение модели прогнозирования месячных расходов пользователя.

Задача:
    По истории транзакций одного пользователя за несколько месяцев
    построить модель, которая предсказывает суммарные расходы
    (total_expense = сумма Withdrawal) на следующий месяц.

Ожидаемый формат исходных данных (CSV):
    - Date        : дата транзакции
    - Withdrawal  : сумма расхода (>0 для списаний, 0 для пополнений)
    - Deposit     : сумма пополнения (>0 для приходов, 0 для списаний)
    - Balance     : баланс после операции
    - Category    : строковая категория (Food, Misc, ...)

Запуск из корня пакета `forecasting`:
    python -m forecasting.train \
        --data resources/data/ci_data.csv \
        --monthly resources/data/monthly_features.csv \
        --model resources/models/expense_forecast.joblib

Если аргументы не переданы, пути берутся из config.py:
    - DATA_PATH
    - MONTHLY_FEATURES_PATH
    - MODEL_PATH
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .config import DATA_PATH, MONTHLY_FEATURES_PATH, MODEL_PATH
from .features import (
    load_transactions,
    build_monthly_features,
    add_lag_features,
    train_test_split_by_month,
    build_feature_matrix,
)

logger = logging.getLogger(__name__)


# ---------- Вспомогательные функции ----------

def _mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error в процентах."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100.0)


# ---------- Основная функция обучения ----------

def train_model(
    data_path: Path | str | None = None,
    monthly_out: Path | str | None = None,
    model_path: Path | str | None = None,
    test_months: int = 3,
) -> None:
    """
    Обучает модель прогноза расходов и сохраняет артефакт на диск.

    Этапы:
      1. Читаем транзакции.
      2. Строим месячные фичи.
      3. Добавляем лаги и time-series признаки.
      4. Делим на train/test по месяцам.
      5. Считаем baseline (наивный прогноз: расход прошлого месяца).
      6. Обучаем Ridge и RandomForest.
      7. Выбираем лучшую модель по RMSE.
      8. Сохраняем joblib + JSON-отчёт рядом.
    """
    logging.basicConfig(level=logging.INFO)

    # Разруливаем пути: либо из аргументов, либо из config.py
    data_path = Path(data_path) if data_path is not None else Path(DATA_PATH)
    monthly_out = Path(monthly_out) if monthly_out is not None else Path(MONTHLY_FEATURES_PATH)
    model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)

    monthly_out.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Этап 1: загрузка и агрегация транзакций ===")
    df_tx = load_transactions(data_path)
    df_monthly = build_monthly_features(df_tx)
    df_lagged = add_lag_features(df_monthly)

    if df_lagged.empty:
        raise RuntimeError("После добавления лагов не осталось ни одной записи.")

    df_lagged.to_csv(monthly_out, index=False)
    logger.info("Месячные фичи сохранены в %s", monthly_out)

    logger.info("=== Этап 2: train/test split по месяцам ===")
    df_train, df_test = train_test_split_by_month(df_lagged, test_months=test_months)
    if df_train.empty or df_test.empty:
        raise RuntimeError("Пустой train или test после сплита, увеличьте размер данных.")

    # ---------- Baseline: наивный прогноз (как в прошлом месяце) ----------

    baseline_mask = df_test["baseline_prev_month"].notna()
    y_test_baseline = df_test.loc[baseline_mask, "total_expense"].to_numpy(dtype=float)
    y_pred_baseline = df_test.loc[baseline_mask, "baseline_prev_month"].to_numpy(dtype=float)

    baseline_mae = mean_absolute_error(y_test_baseline, y_pred_baseline)
    # В sklearn нет параметра root, чтобы взять корень — используем squared=False.
    baseline_rmse = root_mean_squared_error(y_test_baseline, y_pred_baseline)
    baseline_mape = _mape(y_test_baseline, y_pred_baseline)

    logger.info(
        "Baseline (naive prev-month): MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        baseline_mae,
        baseline_rmse,
        baseline_mape,
    )

    # ---------- Формируем X, y для моделей ----------

    logger.info("=== Этап 3: подготовка X/y ===")
    X_train, y_train, feature_cols = build_feature_matrix(df_train)
    X_test, y_test, _ = build_feature_matrix(df_test)

    logger.info("Train shape: X=%s, Test shape: X=%s", X_train.shape, X_test.shape)

    # ---------- Обучение моделей ----------

    logger.info("=== Этап 4: обучение моделей ===")
    models: Dict[str, Dict[str, Any]] = {}

    # 1. Ridge Regression (линейный baseline)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    ridge_rmse = root_mean_squared_error(y_test, y_pred_ridge)
    ridge_mape = _mape(y_test, y_pred_ridge)

    logger.info(
        "Ridge: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        ridge_mae,
        ridge_rmse,
        ridge_mape,
    )

    models["ridge"] = {
        "model": ridge,
        "mae": ridge_mae,
        "rmse": ridge_rmse,
        "mape": ridge_mape,
    }

    # 2. RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=4,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = root_mean_squared_error(y_test, y_pred_rf)
    rf_mape = _mape(y_test, y_pred_rf)

    logger.info(
        "RandomForest: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        rf_mae,
        rf_rmse,
        rf_mape,
    )

    models["random_forest"] = {
        "model": rf,
        "mae": rf_mae,
        "rmse": rf_rmse,
        "mape": rf_mape,
    }

    # (опционально позже сюда можно добавить XGBoost и др. регрессоры)

    # ---------- Выбор лучшей модели и сохранение ----------

    best_name = min(models.keys(), key=lambda k: models[k]["rmse"])
    best_entry = models[best_name]
    best_model = best_entry["model"]

    logger.info(
        "Лучшая модель: %s (RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%). Baseline RMSE=%.2f",
        best_name,
        best_entry["rmse"],
        best_entry["mae"],
        best_entry["mape"],
        baseline_rmse,
    )

    logger.info("=== Этап 5: сохранение артефакта ===")
    payload = {
        "model": best_model,
        "feature_cols": feature_cols,
        "metrics": {
            "baseline": {
                "mae": baseline_mae,
                "rmse": baseline_rmse,
                "mape": baseline_mape,
            },
            "ridge": {
                "mae": ridge_mae,
                "rmse": ridge_rmse,
                "mape": ridge_mape,
            },
            "random_forest": {
                "mae": rf_mae,
                "rmse": rf_rmse,
                "mape": rf_mape,
            },
            "best_model": best_name,
        },
    }

    joblib.dump(payload, model_path)
    logger.info("Модель прогноза расходов сохранена в %s", model_path)

    # JSON-отчёт рядом с моделью
    report_path = model_path.with_suffix(".forecast_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload["metrics"], f, ensure_ascii=False, indent=2)
    logger.info("Отчёт о модели сохранён в %s", report_path)


# ---------- CLI оболочка ----------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение модели прогноза расходов.")
    parser.add_argument(
        "--data",
        type=str,
        help=f"Путь к CSV с транзакциями (по умолчанию {DATA_PATH})",
        default=None,
    )
    parser.add_argument(
        "--monthly",
        type=str,
        help=f"Путь для сохранения monthly_features.csv (по умолчанию {MONTHLY_FEATURES_PATH})",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Путь для сохранения модели (по умолчанию {MODEL_PATH})",
        default=None,
    )
    parser.add_argument(
        "--test-months",
        type=int,
        help="Сколько последних месяцев использовать как test (по умолчанию 3).",
        default=3,
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    train_model(
        data_path=args.data,
        monthly_out=args.monthly,
        model_path=args.model,
        test_months=args.test_months,
    )


if __name__ == "__main__":
    main()
