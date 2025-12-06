import asyncio

from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka
import contextlib

from app.deps import create_container
from app.routes import router as api_router
from app.services.exception_handler import register_exception_handlers
from app.settings.app import AppSettings


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.dishka_container.close()


def create_app() -> FastAPI:
    container = create_container()
    settings = asyncio.run(container.get(AppSettings))
    app = FastAPI(lifespan=lifespan, title=settings.app_name)
    register_exception_handlers(app)
    setup_dishka(container, app=app)
    app.include_router(api_router)
    return app
