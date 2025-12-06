# app/routes/category_classifier/routes.py

from datetime import datetime
from fastapi import APIRouter, Depends

from app.deps.ml import get_transaction_ml_service
from app.services.ml.service import TransactionMLService
from app.schemas.base import APIModel  # или свой базовый
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
