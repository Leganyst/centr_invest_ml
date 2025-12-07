"""
Feature engineering для модуля прогноза месячных расходов.

Содержит:
- загрузку и очистку сырых транзакций;
- агрегацию по месяцам (monthly features);
- добавление лагов и time-series признаков;
- temporal train/test split по месяцам;
- подготовку матрицы признаков X, y для моделей sklearn.
"""
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from .config import DATA_PATH  # ожидается, что в config.py есть DATA_PATH

logger = logging.getLogger(__name__)


# ---------- Загрузка и подготовка сырых транзакций ----------

def load_transactions(path: Path | str | None = None) -> pd.DataFrame:
    """
    Загружает исходный CSV с транзакциями и приводит типы.

    Ожидаемые колонки:
        - Date        : дата транзакции
        - Withdrawal  : сумма расхода (>0 для списаний, 0 для пополнений)
        - Deposit     : сумма пополнения (>0 для приходов, 0 для списаний)
        - Balance     : баланс после операции
        - Category    : строковая категория (Food, Misc, ...)

    :param path: путь к CSV. Если None — берётся DATA_PATH из config.py.
    :return: очищенный DataFrame, отсортированный по Date.
    """
    path = Path(path) if path is not None else Path(DATA_PATH)
    logger.info("Загружаю транзакции из %s", path)
    df = pd.read_csv(path)

    required = {"Date", "Withdrawal", "Deposit", "Balance", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Withdrawal", "Deposit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Category"] = df["Category"].astype(str)
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info("После очистки осталось %d транзакций.", len(df))
    return df


# ---------- Агрегация по месяцам ----------

def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует транзакции по месяцам и добавляет базовые фичи.

    Выход: по одной строке на месяц, колонки:
        - year_month      : Timestamp первого дня месяца
        - total_expense   : сумма Withdrawal за месяц
        - total_deposit   : сумма Deposit за месяц
        - avg_balance     : средний Balance за месяц
        - end_balance     : Balance последней транзакции месяца
        - n_transactions  : количество транзакций
        - avg_tx_amount   : средний размер расходной транзакции
        - food_ratio      : доля транзакций с Category == "Food"
        - misc_ratio      : доля транзакций с Category == "Misc"
        - is_december     : 1, если месяц == 12
        - quarter         : квартал (1–4)
        - month_sin/cos   : сезонность по месяцу (sin/cos)

    :param df: сырые транзакции с колонками Date/Withdrawal/Deposit/Balance/Category.
    :return: DataFrame с месячными признаками.
    """
    df = df.copy()
    df["year_month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    grouped = df.groupby("year_month", sort=True)

    monthly = pd.DataFrame(index=grouped.size().index)
    monthly.index.name = "year_month"

    monthly["total_expense"] = grouped["Withdrawal"].sum()
    monthly["total_deposit"] = grouped["Deposit"].sum()
    monthly["avg_balance"] = grouped["Balance"].mean()
    monthly["end_balance"] = grouped["Balance"].last()
    monthly["n_transactions"] = grouped.size()

    # Средний размер расходной транзакции
    def _avg_expense_amount(g: pd.DataFrame) -> float:
        w = g["Withdrawal"]
        w = w[w > 0]
        if w.empty:
            return 0.0
        return float(w.mean())

    monthly["avg_tx_amount"] = grouped.apply(_avg_expense_amount)

    # Доли категорий
    def _ratio(g: pd.DataFrame, category: str) -> float:
        if g.empty:
            return 0.0
        return float((g["Category"] == category).mean())

    monthly["food_ratio"] = grouped.apply(lambda g: _ratio(g, "Food"))
    monthly["misc_ratio"] = grouped.apply(lambda g: _ratio(g, "Misc"))

    # Календарные фичи
    monthly = monthly.reset_index()
    monthly["month"] = monthly["year_month"].dt.month
    monthly["is_december"] = (monthly["month"] == 12).astype(int)
    monthly["quarter"] = monthly["year_month"].dt.quarter

    # сезонность: sin/cos по месяцу
    month_rad = 2 * np.pi * (monthly["month"] / 12.0)
    monthly["month_sin"] = np.sin(month_rad)
    monthly["month_cos"] = np.cos(month_rad)

    logger.info("Собрано месячных фичей: %d строк.", len(monthly))
    return monthly


# ---------- Лаги и time-series фичи ----------

def add_lag_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет лаги и time-series признаки к месячным фичам.

    Добавляется:
        - total_expense_lag1, lag2, lag3
        - total_expense_ma3   (скользящее среднее 3 предыдущих месяцев)
        - total_expense_growth (рост относительно прошлого месяца, %)
        - end_balance_lag1
        - baseline_prev_month (наивный прогноз = расход прошлого месяца)

    Строки без полных лагов (первые несколько месяцев) удаляются.

    :param monthly: DataFrame после build_monthly_features.
    :return: DataFrame с лага-колонками.
    """
    df = monthly.copy()
    df = df.sort_values("year_month").reset_index(drop=True)

    df["total_expense_lag1"] = df["total_expense"].shift(1)
    df["total_expense_lag2"] = df["total_expense"].shift(2)
    df["total_expense_lag3"] = df["total_expense"].shift(3)

    # скользящее среднее за 3 последних месяца, сдвинутое на 1
    df["total_expense_ma3"] = (
        df["total_expense"]
        .rolling(window=3, min_periods=3)
        .mean()
        .shift(1)
    )

    # рост vs прошлый месяц
    df["total_expense_growth"] = df["total_expense"].pct_change().shift(1)

    df["end_balance_lag1"] = df["end_balance"].shift(1)

    # наивный прогноз = расход прошлого месяца
    df["baseline_prev_month"] = df["total_expense_lag1"]

    # выкидываем строки без полных лагов
    df = df.dropna(
        subset=["total_expense_lag1", "total_expense_lag2", "total_expense_lag3"]
    ).reset_index(drop=True)

    logger.info(
        "После добавления лагов осталось %d строк для обучения/валидации.",
        len(df),
    )
    return df


# ---------- Train / test split по времени ----------

def train_test_split_by_month(
    df: pd.DataFrame,
    test_months: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит месячный датасет на train/test по календарю.

    По умолчанию:
        - test = последние `test_months` месяцев,
        - train = все предыдущие.

    Это честный temporal split для временного ряда.

    :param df: DataFrame после add_lag_features.
    :param test_months: сколько последних месяцев пустить в test.
    :return: (df_train, df_test)
    """
    df = df.sort_values("year_month").reset_index(drop=True)
    unique_months: List[pd.Timestamp] = df["year_month"].drop_duplicates().tolist()

    if len(unique_months) <= test_months:
        raise RuntimeError(
            f"Слишком мало месяцев ({len(unique_months)}) "
            f"для выделения {test_months} тестовых."
        )

    test_months_list = unique_months[-test_months:]
    mask_test = df["year_month"].isin(test_months_list)

    df_train = df[~mask_test].reset_index(drop=True)
    df_test = df[mask_test].reset_index(drop=True)

    logger.info(
        "Train months: %d, test months: %d",
        df_train["year_month"].nunique(),
        df_test["year_month"].nunique(),
    )
    return df_train, df_test


# ---------- Подготовка X, y для моделей ----------

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Разделяет месячный датасет на X (признаки) и y (target).

    Исключаем из признаков:
        - year_month         (идентификатор времени)
        - total_expense      (target)
        - baseline_prev_month (baseline, а не фича модели)
        - month              (если есть month_sin/month_cos)

    :param df: DataFrame после add_lag_features.
    :return: (X, y, feature_cols)
    """
    exclude_cols = {
        "year_month",
        "total_expense",
        "baseline_prev_month",
        "month",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["total_expense"].to_numpy(dtype=float)
    return X, y, feature_cols
