import logging
from typing import Annotated

from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, UploadFile
from fastapi.params import File
from fastapi_cloud_cli.utils.pydantic_compat import TypeAdapter

from app.deps.auth import CurrentUser, CurrentUserDependency
from app.models import Transaction
from app.schemas.transactions import TransactionSchema
from app.services.filters import Paginated
from app.services.transactions import TransactionImporter, \
    TransactionRetrieveInteractor

router = APIRouter(
    prefix="/transactions", tags=["transactions"],
    route_class=DishkaRoute,
    dependencies=[CurrentUserDependency]
)
logger = logging.getLogger(__name__)


@router.post("/import", response_model=None)
async def import_transactions(
    service: FromDishka[TransactionImporter],
    file: Annotated[UploadFile, File()],
) -> None:
    await service(file)


@router.get("")
async def list_transactions(
        current_user: CurrentUser,
        service: FromDishka[TransactionRetrieveInteractor],
        page: Paginated,
) -> list[TransactionSchema]:
    transactions = await service.all(
        Transaction.user_id == current_user.id,
        page=page,
    )
    return TypeAdapter(list[TransactionSchema]).validate_python(transactions)
