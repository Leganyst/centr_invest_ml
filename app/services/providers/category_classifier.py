from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionSchema
from app.services.providers.protocols.category_classifier import PredictionResult
from app.settings.ml import ModelSettings
from category_classifier import CategoryClassifierService
from category_classifier.schemas import Transaction


class MlCategoryClassifier:
    def __init__(self, settings: ModelSettings):
        self.classifier = CategoryClassifierService(model_path=settings.model_path)

    def predict(self, transaction: TransactionSchema) -> PredictionResult:
        tx = Transaction(
            balance=transaction.balance,
            deposit=transaction.deposit,
            withdrawal=transaction.withdrawal,
            date=transaction.date.isoformat(),
        )
        response = self.classifier.predict(tx)
        category = TransactionCategory(response.category)
        probabilities = {
            TransactionCategory(key): float(probability)
            for key, probability in response.proba.items()
        }
        return PredictionResult(category=category, probabilities=probabilities)
