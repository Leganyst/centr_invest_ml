from typing import AsyncIterable

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.db import new_engine, new_session_maker
from app.settings.db import DatabaseSettings


class DbConnectionProvider(Provider):
    @provide(scope=Scope.APP)
    def get_engine(self, settings: DatabaseSettings) -> AsyncEngine:
        return new_engine(settings)

    @provide(scope=Scope.APP)
    def get_session_maker(
        self, engine: AsyncEngine
    ) -> async_sessionmaker[AsyncSession]:
        return new_session_maker(engine)

    @provide(scope=Scope.REQUEST)
    async def get_session(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> AsyncIterable[AsyncSession]:
        async with session_maker() as session:
            yield session
