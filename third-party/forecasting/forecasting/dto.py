from dataclasses import dataclass
from enum import StrEnum
from typing import Union
from datetime import date, datetime


class Category(StrEnum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"


@dataclass
class Transaction:
    date: Union[str, date, datetime]
    withdrawal: float
    deposit: float
    balance: float
    category: Category | str


@dataclass
class ForecastResponse:
    predicted_expense: float
    