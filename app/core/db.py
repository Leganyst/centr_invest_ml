
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.settings.db import DatabaseSettings


def new_engine(settings: DatabaseSettings) -> AsyncEngine:
    return create_async_engine(
        str(settings.database_url),
        pool_size=15,
        max_overflow=15,
    )


def new_session_maker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(
        engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
    )





