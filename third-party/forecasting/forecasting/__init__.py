# forecasting/service.py
from __future__ import annotations

from typing import List

import pandas as pd

from forecasting.model import ExpenseForecastModel
from forecasting.dto import Transaction, ForecastResponse, Category


class ExpenseForecastService:
    def __init__(self):
        self.model = ExpenseForecastModel()

    def forecasting(self, tx_list: List[Transaction]) -> ForecastResponse:
        """
        Принимает список транзакций одного пользователя
        и возвращает прогноз суммарных расходов на следующий месяц.
        """
        if not tx_list:
            raise ValueError("Для форекаста нужен непустой список транзакций.")

        rows = []
        for tx in tx_list:
            category = tx.category.value if isinstance(tx.category, Category) else str(tx.category)
            rows.append(
                {
                    "Date": tx.date,
                    "Withdrawal": tx.withdrawal,
                    "Deposit": tx.deposit,
                    "Balance": tx.balance,
                    "Category": category,
                }
            )

        df = pd.DataFrame(rows)
        predicted = self.model.predict_next_month(df)
        return ForecastResponse(predicted_expense=predicted)
