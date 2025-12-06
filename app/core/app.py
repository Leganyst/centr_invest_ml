from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka
import contextlib

from app.deps import create_container
from app.schemas import health
from app.routes import router


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.dishka_container.close()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    container = create_container()
    setup_dishka(container, app=app)
    app.include_router(router)
    return app
