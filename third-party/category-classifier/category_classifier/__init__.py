from pathlib import Path

from category_classifier.model import TransactionClassifier
from category_classifier.schemas import Category, PredictionResponse, Transaction


class CategoryClassifierService:
    def __init__(self, model_path: Path | str | None = None):
        self.classifier = TransactionClassifier(model_path=model_path)

    def predict(self, tx: Transaction) -> PredictionResponse:
        probes = self.classifier.predict_proba(tx)
        predicted = max(probes, key=probes.get)
        return PredictionResponse(predicted, probes)
