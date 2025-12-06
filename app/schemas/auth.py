from app.models.enums import UserRole
from app.schemas.base import BaseSchema
from uuid import UUID


class UserPrincipal(BaseSchema):
    user_id: UUID
    role: UserRole


class RegisterSchema:
    ...