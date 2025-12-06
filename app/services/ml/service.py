from datetime import date as date_type, datetime
from typing import Any, Mapping

from .model import TransactionClassifier
from . import schemas  # CategoryEnum уже там есть


class TransactionMLService:
    """
    Высокоуровневый сервис для классификации транзакций.

    Обёртка над :class:`TransactionClassifier`, которая скрывает детали работы
    с pandas/Joblib и принимает «чистые» доменные типы (datetime/float).
    Используется в бэкенде как обычный сервис/интерактор.

    На вход сервис принимает одну транзакцию (дата, списание, зачисление,
    баланс) и возвращает предсказанную категорию как Enum. При необходимости
    можно получить и распределение вероятностей по всем классам.

    **Аргументы конструктора:**
        classifier: TransactionClassifier | None
            Уже инициализированный классификатор. Если не передан, сервис
            сам создаст :class:`TransactionClassifier`, который загрузит
            модель из пути, указанного в ml-конфиге.

    **Ожидаемые поля транзакции (для вызова сервиса):**
        date: str | datetime | date
            Дата операции. Может быть datetime/date или строкой.
            Внутри приводится к строке и парсится моделью.
        withdrawal: float
            Сумма списания (расход). Если операция приходная — обычно 0.0.
        deposit: float
            Сумма зачисления (доход). Если операция расходная — обычно 0.0.
        balance: float
            Баланс счёта после операции.

    **Возвращаемые значения:**
        - при вызове экземпляра (`__call__`): :class:`schemas.CategoryEnum`
          (например, CategoryEnum.FOOD / CategoryEnum.SALARY и т.д.).
        - при вызове :meth:`with_proba`: кортеж
          ``(CategoryEnum, dict[str, float])``, где второй элемент —
          словарь вида ``{"Food": 0.78, "Misc": 0.15, ...}``.

    **Примеры использования (в чистом Python):**

    .. code-block:: python

        from datetime import datetime
        from app.services.ml.service import TransactionMLService

        ml_service = TransactionMLService()

        # Простой предикт категории
        category = ml_service(
            date=datetime(2024, 2, 10, 15, 30),
            withdrawal=499.90,
            deposit=0.0,
            balance=15000.0,
        )

        print(category)        # -> CategoryEnum.FOOD (например)
        print(category.value)  # -> "Food"

        # Предикт с распределением вероятностей
        category, proba = ml_service.with_proba(
            date="2024-03-01",
            withdrawal=0.0,
            deposit=80000.0,
            balance=120000.0,
        )

        # category -> CategoryEnum.SALARY (например)
        # proba -> {"Food": 0.01, "Salary": 0.90, "Misc": 0.05, ...}

    **Пример интеграции во FastAPI через Depends:**

    .. code-block:: python

        from fastapi import APIRouter, Depends
        from datetime import datetime
        from app.services.ml.service import TransactionMLService
        from app.deps.ml import get_transaction_ml_service
        from app.schemas.base import APIModel
        from app.services.ml.schemas import CategoryEnum

        router = APIRouter(prefix="/category", tags=["category"])

        class ClassifyRequest(APIModel):
            date: datetime
            withdrawal: float
            deposit: float
            balance: float

        class ClassifyResponse(APIModel):
            category: CategoryEnum

        @router.post("/classify", response_model=ClassifyResponse)
        def classify_transaction(
            payload: ClassifyRequest,
            ml_service: TransactionMLService = Depends(get_transaction_ml_service),
        ) -> ClassifyResponse:
            category = ml_service(
                date=payload.date,
                withdrawal=payload.withdrawal,
                deposit=payload.deposit,
                balance=payload.balance,
            )
            return ClassifyResponse(category=category)
    """
    
    def __init__(self, classifier: TransactionClassifier | None = None) -> None:
        # Можно передать готовый classifier, можно создать по умолчанию
        self._clf = classifier or TransactionClassifier()

    def __call__(
        self,
        *,
        date: str | datetime | date_type,
        withdrawal: float,
        deposit: float,
        balance: float,
    ) -> schemas.CategoryEnum:
        """
        Синхронный вызов сервиса: одна транзакция -> одна категория.

        Здесь мы превращаем доменные типы (datetime/Decimal и т.п.)
        в тот payload, который жрёт TransactionClassifier.
        """
        payload: Mapping[str, Any] = {
            "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
            "withdrawal": float(withdrawal),
            "deposit": float(deposit),
            "balance": float(balance),
        }
        category_str = self._clf.predict_category(payload)
        # Преобразуем к CategoryEnum, чтобы дальше в бэке жить в своих enum'ах
        return schemas.CategoryEnum(category_str)

    def with_proba(
        self,
        *,
        date: str | datetime | date_type,
        withdrawal: float,
        deposit: float,
        balance: float,
    ) -> tuple[schemas.CategoryEnum, dict[str, float]]:
        """Если нужно ещё и распределение вероятностей."""
        payload: Mapping[str, Any] = {
            "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
            "withdrawal": float(withdrawal),
            "deposit": float(deposit),
            "balance": float(balance),
        }
        category_str = self._clf.predict_category(payload)
        proba = self._clf.predict_proba(payload)
        return schemas.CategoryEnum(category_str), proba