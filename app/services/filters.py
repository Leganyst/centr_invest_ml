from typing import Annotated, Any

from annotated_types import Ge, Le
from fastapi.params import Depends, Query
from sqlalchemy import ColumnElement, Select

from app.schemas.base import BaseSchema
from collections.abc import Sequence


type FilterType = ColumnElement[bool]


class PaginatedSchema(BaseSchema):
    limit: Annotated[int, Ge(ge=1), Le(le=100)] = 10
    offset: Annotated[int, Ge(ge=0)] = 0
    ordering: str | None = None


def get_pagination(
    limit: Annotated[int, Query()] = 10,
    offset: Annotated[int, Query()] = 0,
    ordering: str | None = None,
):
    return PaginatedSchema(
        limit=limit,
        offset=offset,
        ordering=ordering,
    )


Paginated = Annotated[PaginatedSchema, Depends(get_pagination)]


def apply_pagination[T: tuple[Any, ...]](
    query: Select[T],
    page: PaginatedSchema,
    default_ordering: ColumnElement | Sequence[ColumnElement] | None = None,
    ordering_mapping: dict[str, ColumnElement] | Sequence[ColumnElement] | None = None,
) -> Select[T]:
    query = query.offset(page.offset).limit(page.limit)
    if ordering_mapping:
        if not isinstance(ordering_mapping, dict):
            ordering_mapping = {column.key: column for column in ordering_mapping}
        query = query.order_by(ordering_mapping[page.ordering])
    elif default_ordering is not None:
        if not isinstance(default_ordering, Sequence):
            default_ordering = [default_ordering]
        query = query.order_by(*default_ordering)
    return query
