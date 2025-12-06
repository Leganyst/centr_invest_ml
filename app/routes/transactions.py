import logging

from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter


router = APIRouter(prefix="/transactions", tags=["transactions"], route_class=DishkaRoute)
logger = logging.getLogger(__name__)


@router.post("/import")
async def import_transactions():
    ...


@router.get("")
async def list_transactions():
    ...


