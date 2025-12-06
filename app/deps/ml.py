
from functools import lru_cache

from app.services.ml.model import TransactionClassifier
from app.services.ml.service import TransactionMLService
from app.settings import ml as ml_settings

@lru_cache(maxsize=1)
def get_transaction_ml_service() -> TransactionMLService:
    """
    Синглтон для ML-сервиса: модель грузится один раз на процесс.
    """
    settings = ml_settings.get_settings()  # если у тебя там Pydantic Settings
    clf = TransactionClassifier(model_path=settings.model_path)
    return TransactionMLService(classifier=clf)