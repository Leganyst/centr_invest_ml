from fastapi import APIRouter
from dishka.integrations.fastapi import DishkaRoute

from app.schemas.health import HealthSchema

router = APIRouter(route_class=DishkaRoute)


@router.get("/health")
def health() -> HealthSchema:
    return HealthSchema(
        status="ok"
    )
