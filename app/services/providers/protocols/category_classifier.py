from typing import Protocol

from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionSchema


class ICategoryClassifier(Protocol):
    def predict(self, transaction: TransactionSchema) -> TransactionCategory: ...
