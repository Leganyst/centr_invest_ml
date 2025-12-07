from typing import Iterable

from dishka import Provider, Scope, provide
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.services.filters import FilterType
from app.services.filters import PaginatedSchema, apply_pagination


class RetrieveUserInteractor:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def exists(self, query: FilterType) -> bool:
        value = await self.session.scalar(exists(User).select().where(query))
        return bool(value)

    async def all(
        self, filters: FilterType | None = None, paginate: PaginatedSchema | None = None
    ) -> Iterable[User]:
        query = select(User)
        if filters:
            query = query.where(filters)
        if paginate:
            query = apply_pagination(query, paginate, default_ordering=User.id.desc())
        return await self.session.scalars(query)

    async def get(self, query: FilterType) -> User | None:
        return await self.session.scalar(select(User).where(query).limit(1))


class UserServicesProvider(Provider):
    scope = Scope.REQUEST

    retrieve = provide(RetrieveUserInteractor)
