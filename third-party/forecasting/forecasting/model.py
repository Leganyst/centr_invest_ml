"""
Интерфейс для работы с моделью прогноза месячных расходов.

Идея:
    - модель и фичи живут в этом пакете;
    - бэку не нужно знать, как устроены лаги/агрегации;
    - бэк работает с одним классом: ExpenseForecastModel.

Ожидаемый вход:
    DataFrame с транзакциями одного пользователя, колонки:
        - Date       : дата транзакции (str / datetime)
        - Withdrawal : расход (>0 для списаний, 0 для пополнений)
        - Deposit    : пополнения (>0 для приходов, 0 для списаний)
        - Balance    : баланс после операции
        - Category   : строковая категория (Food, Misc, ...)

Основной сценарий использования в бэке:

    from forecasting.model import ExpenseForecastModel
    import pandas as pd

    df_user = pd.read_csv("user_transactions.csv")

    model = ExpenseForecastModel()  # подхватит путь к артефакту из config.py

    # Прогноз суммарных расходов (Withdrawal) на следующий месяц:
    next_month_expense = model.predict_next_month(df_user)

    # Если нужно — посмотреть метрики на hold-out:
    print(model.metrics)

    # Для диагностики/отладки:
    backtest_df = model.predict_series(df_user)
"""
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .features import build_monthly_features, add_lag_features


class ExpenseForecastModel:
    """
    Обёртка над joblib-артефактом модели прогноза расходов.

    Артефакт (joblib-файл), созданный forecasting.train, содержит:
        - model        : обученный регрессор (Ridge / RandomForest)
        - feature_cols : список имён признаков, ожидаемых моделью
        - metrics      : словарь с метриками (baseline, ridge, rf, best_model)

    Основные методы:
        - predict_next_month(df_tx) -> float
        - predict_series(df_tx) -> pd.DataFrame
        - metrics (property) -> dict
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        """
        :param model_path:
            Путь к joblib-файлу с моделью.
            Если None — используется MODEL_PATH из config.py.
        """
        self.model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
        payload: Dict[str, Any] = joblib.load(self.model_path)

        self._model = payload["model"]
        self._feature_cols = list(payload.get("feature_cols", []))
        self._metrics: Dict[str, Any] = payload.get("metrics", {})

    # -------- Публичные свойства --------

    @property
    def feature_cols(self) -> list[str]:
        """Список имён признаков, которые ожидает регрессионная модель."""
        return self._feature_cols

    @property
    def metrics(self) -> Dict[str, Any]:
        """
        Метрики на hold-out (те же, что посчитал train.py).

        Структура:
            {
              "baseline": {"mae": ..., "rmse": ..., "mape": ...},
              "ridge": {"mae": ..., "rmse": ..., "mape": ...},
              "random_forest": {...},
              "best_model": "random_forest" | "ridge"
            }
        """
        return self._metrics

    # -------- Вспомогательная подготовка признаков --------

    def _prepare_lagged_monthly(self, df_tx: pd.DataFrame) -> pd.DataFrame:
        """
        Внутренний пайплайн:
            сырые транзакции -> месячные агрегации -> лаги.

        Требует колонок:
            Date, Withdrawal, Deposit, Balance, Category.

        :return: DataFrame с месячными фичами и лага-колонками.
        """
        df = df_tx.copy()
        # Минимальная нормализация (на случай, если бэк уже что-то делал с df)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        for col in ["Withdrawal", "Deposit", "Balance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["Category"] = df["Category"].astype(str)

        monthly = build_monthly_features(df)
        lagged = add_lag_features(monthly)
        return lagged

    # -------- Основные методы инференса --------

    def predict_next_month(self, df_tx: pd.DataFrame) -> float:
        """
        Прогнозирует суммарные расходы пользователя (Withdrawal) на следующий месяц.

        Ожидается DataFrame с историей транзакций пользователя за несколько
        последних месяцев. Для стабильного прогноза желательно ≥ 4–6 месяцев
        истории.

        Вход:
            df_tx: pd.DataFrame с колонками:
                - 'Date'
                - 'Withdrawal'
                - 'Deposit'
                - 'Balance'
                - 'Category'

        Выход:
            float: предсказанное значение total_expense для месяца,
                   следующего за последним месяцем в df_tx (по формату обучения
                   это прогноз "по последнему набору лагов").

        Под капотом:
            - агрегируем транзакции по месяцам;
            - добавляем лаги и time-series признаки;
            - берём последнюю строку с полным набором лагов;
            - формируем вектор признаков в том же порядке, что при обучении;
            - прогоняем через сохранённый регрессор.
        """
        df_lagged = self._prepare_lagged_monthly(df_tx)
        if df_lagged.empty:
            raise ValueError(
                "Недостаточно данных для построения лагов (нужно минимум несколько месяцев истории)."
            )

        last_row = df_lagged.iloc[[-1]]  # DataFrame из одной строки

        # Проверяем, что все нужные фичи есть
        missing = [c for c in self._feature_cols if c not in last_row.columns]
