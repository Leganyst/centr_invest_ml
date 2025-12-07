from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionSchema
from category_classifier import CategoryClassifierService
from category_classifier.schemas import Transaction


class MlCategoryClassifier:
    def __init__(self):
        self.classifier = CategoryClassifierService()

    def predict(self, transaction: TransactionSchema) -> TransactionCategory:
        tx = Transaction(
            balance=transaction.balance,
            deposit=transaction.deposit,
            withdrawal=transaction.withdrawal,
            date=transaction.date.isoformat(),
        )
        response = self.classifier.predict(tx)
        return TransactionCategory(response.category)
