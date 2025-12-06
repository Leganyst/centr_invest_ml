from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Union

class Category(StrEnum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"


DateLike = Union[str, date, datetime]


@dataclass(slots=True)
class Transaction:
    """
    Транзакция для модуля форекаста.

    Здесь category уже входной признак, потому что
    мы считаем food_ratio/misc_ratio и т.п. при агрегации.
    """
    date: DateLike
    withdrawal: float
    deposit: float
    balance: float
    category: Category | str


@dataclass(slots=True)
class ForecastResponse:
    """
    Ответ от сервиса форекаста.

    predicted_expense — прогноз суммарных расходов (Withdrawal)
    на следующий календарный месяц.
    """
    predicted_expense: float
