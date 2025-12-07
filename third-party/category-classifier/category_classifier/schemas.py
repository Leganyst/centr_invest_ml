from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"


@dataclass
class Transaction:
    date: str
    withdrawal: float
    deposit: float
    balance: float


@dataclass
class PredictionResponse:
    category: Category
    proba: dict[Category, float]
