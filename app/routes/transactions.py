import logging
from typing import Annotated

from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, UploadFile
from fastapi.params import File
from fastapi_cloud_cli.utils.pydantic_compat import TypeAdapter
from starlette import status

from app.deps.auth import CurrentUser, CurrentUserDependency
from app.models import Transaction
from app.schemas import ml as ml_schemas
from app.schemas.base import BaseSchema
from app.schemas.transactions import TransactionSchema
from app.services.filters import Paginated, PaginatedResponseSchema
from app.services.transactions import (
    TransactionAnalyticsInteractor,
    TransactionImporter,
    TransactionRetrieveInteractor,
)

router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
    route_class=DishkaRoute,
    dependencies=[CurrentUserDependency],
)
logger = logging.getLogger(__name__)


class TransactionImportResponseSchema(BaseSchema):
    count: int


@router.post("/import", status_code=status.HTTP_201_CREATED)
async def import_transactions(
    service: FromDishka[TransactionImporter],
    file: Annotated[UploadFile, File()],
) -> TransactionImportResponseSchema:
    transactions = await service(file)
    return TransactionImportResponseSchema(count=len(transactions))


@router.get("")
async def list_transactions(
    current_user: CurrentUser,
    service: FromDishka[TransactionRetrieveInteractor],
    page: Paginated,
) -> PaginatedResponseSchema[TransactionSchema]:
    transactions = await service.all(
        Transaction.user_id == current_user.id,
        page=page,
    )
    return TypeAdapter(PaginatedResponseSchema[TransactionSchema]).validate_python(
        transactions
    )


@router.get("/analytics", response_model=ml_schemas.CSVUploadResponse)
async def transactions_analytics(
    current_user: CurrentUser,
    analytics: FromDishka[TransactionAnalyticsInteractor],
) -> ml_schemas.CSVUploadResponse:
    return await analytics.for_user(current_user.id)
