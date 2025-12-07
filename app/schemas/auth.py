from pydantic import AliasChoices, EmailStr, Field

from app.models.enums import UserRole
from app.schemas.base import BaseSchema
from uuid import UUID


class UserPrincipal(BaseSchema):
    user_id: UUID = Field(validation_alias=AliasChoices("user_id", "id"))
    role: UserRole


class UserRetrieveSchema(UserPrincipal):
    email: EmailStr
    name: str


class AuthenticationResponseSchema(BaseSchema):
    access_token: str
    user: UserRetrieveSchema


class RegisterSchema(BaseSchema):
    email: EmailStr
    name: str
    password: str


class LoginSchema(BaseSchema):
    email: EmailStr = Field(validation_alias=AliasChoices("email", "username"))
    password: str
