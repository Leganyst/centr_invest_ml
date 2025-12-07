from dataclasses import dataclass
from typing import Protocol

from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionSchema


@dataclass
class PredictionResult:
    category: TransactionCategory
    probabilities: dict[TransactionCategory, float]


class ICategoryClassifier(Protocol):
    def predict(self, transaction: TransactionSchema) -> PredictionResult: ...
