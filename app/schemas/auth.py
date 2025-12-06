from enum import StrEnum

from app.schemas.base import BaseSchema
from uuid import UUID


class UserRoles(StrEnum):
    USER = 'user'
    ADMIN = 'admin'


class UserPrincipal(BaseSchema):
    user_id: UUID
    roles: list[UserRoles]


class RegisterSchema:
    ...