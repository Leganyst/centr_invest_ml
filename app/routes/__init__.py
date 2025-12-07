from apscheduler.schedulers.base import BaseScheduler
from dishka import FromDishka
from fastapi import APIRouter
from dishka.integrations.fastapi import DishkaRoute

from app.schemas.health import HealthSchema, JobSchema
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.transactions import router as transactions_router
from app.routes.ws import router as ws_router


router = APIRouter(route_class=DishkaRoute)
router.include_router(auth_router)
router.include_router(users_router)
router.include_router(transactions_router)
router.include_router(ws_router)


@router.get("/health")
async def health(scheduler: FromDishka[BaseScheduler]) -> HealthSchema:
    jobs = [
        JobSchema(id=job.id, next_run_time=job.next_run_time, name=job.name)
        for job in scheduler.get_jobs()
    ]
    return HealthSchema(
        status="ok",
        jobs=jobs,
    )
