from fastapi import APIRouter
from dishka.integrations.fastapi import DishkaRoute

from app.schemas.health import HealthSchema
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.category_classifier import router as category_classifier_router


router = APIRouter(prefix="/api", route_class=DishkaRoute)
router.include_router(auth_router)
router.include_router(category_classifier_router)
router.include_router(users_router)


@router.get("/health")
def health() -> HealthSchema:
    return HealthSchema(status="ok")
