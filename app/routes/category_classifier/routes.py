# app/routes/category_classifier/routes.py

from datetime import datetime

from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter
from app.models.enums import TransactionCategory
from app.schemas.base import BaseSchema
from app.services.providers.protocols.category_classifier import ICategoryClassifier


router = APIRouter(prefix="/category", tags=["category"], route_class=DishkaRoute)


class ClassifyRequest(BaseSchema):
    date: datetime
    withdrawal: float
    deposit: float
    balance: float


class ClassifyResponse(BaseSchema):
    category: TransactionCategory


@router.post("/classify", response_model=ClassifyResponse)
def classify_transaction(
    payload: ClassifyRequest,
    ml_service: FromDishka[ICategoryClassifier],
) -> ClassifyResponse:
    category = ml_service.predict()
    return ClassifyResponse(category=category)
