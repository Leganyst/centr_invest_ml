import asyncio
import operator
from typing import Any, Awaitable, Callable
import logging

from dishka import AsyncContainer, Scope
from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka
import contextlib

from app.deps import create_container
from app.routes import router as api_router
from app.services.exception_handler import register_exception_handlers
from app.services.transactions import TransactionBackgroundClassifier
from app.settings.app import AppSettings
from apscheduler.schedulers.base import BaseScheduler


logger = logging.getLogger(__name__)


async def service_runner[T](
    container: AsyncContainer,
    target: type[T],
    action: Callable[[T], Awaitable[Any]],
) -> None:
    async with container(scope=Scope.REQUEST) as request_container:
        service = await request_container.get(target)
        try:
            await action(service)
        except Exception:
            logger.exception("Background service error")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    container: AsyncContainer = app.state.dishka_container
    scheduler = await container.get(BaseScheduler)
    scheduler.start()
    scheduler.add_job(
        service_runner,
        args=(
            container,
            TransactionBackgroundClassifier,
            operator.attrgetter("__call__"),
        ),
        trigger="interval",
        id="background-transaction-classifier",
        minutes=1,
    )
    yield
    await container.close()


def create_app() -> FastAPI:
    container = create_container()
    settings = asyncio.run(container.get(AppSettings))
    app = FastAPI(lifespan=lifespan, title=settings.app_name)
    register_exception_handlers(app)
    setup_dishka(container, app=app)
    app.include_router(api_router)
    return app
