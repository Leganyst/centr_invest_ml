from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter
from pydantic import TypeAdapter

from app.models.enums import UserRole
from app.schemas.auth import UserRetrieveSchema
from app.services.auth.permissions import permission_required
from app.services.filters import Paginated
from app.services.users import RetrieveUserInteractor

router = APIRouter(prefix="/users", tags=["users"], route_class=DishkaRoute)


@router.get("", dependencies=[permission_required(UserRole.ADMIN)])
async def list_users(retrieve_users: FromDishka[RetrieveUserInteractor],
                     paginate: Paginated) -> list[UserRetrieveSchema]:
    users = await retrieve_users.all(paginate=paginate)
    return TypeAdapter(list[UserRetrieveSchema]).validate_python(users)
