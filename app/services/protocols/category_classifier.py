from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionSchema


class ICategoryClassifier:
    def predict(self, transaction: TransactionSchema) -> TransactionCategory:
        ...