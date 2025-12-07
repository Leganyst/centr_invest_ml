from typing import Annotated

from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter
from fastapi.params import Body

from app.schemas.auth import (
    RegisterSchema,
    LoginSchema,
    UserRetrieveSchema,
    AuthenticationResponseSchema,
)
from app.services.auth import UserLoginInteractor, UserRegisterInteractor
from app.services.providers.protocols.token_provider import ITokenProvider
from app.deps.auth import CurrentUser

router = APIRouter(prefix="/auth", tags=["auth"], route_class=DishkaRoute)


@router.post("/register")
async def register(
    data: RegisterSchema,
    service: FromDishka[UserRegisterInteractor],
    token_encoder: FromDishka[ITokenProvider],
) -> AuthenticationResponseSchema:
    user = UserRetrieveSchema.model_validate(await service(data))
    access_token = token_encoder.encode_token(user)
    return AuthenticationResponseSchema(
        user=user,
        access_token=access_token,
    )


@router.post("/login")
async def login(
    data: Annotated[LoginSchema, Body()],
    service: FromDishka[UserLoginInteractor],
    token_encoder: FromDishka[ITokenProvider],
) -> AuthenticationResponseSchema:
    user = UserRetrieveSchema.model_validate(await service(data))
    access_token = token_encoder.encode_token(user)
    return AuthenticationResponseSchema(
        user=user,
        access_token=access_token,
    )


@router.get("/me")
async def get_me(user: CurrentUser) -> UserRetrieveSchema:
    return UserRetrieveSchema.model_validate(user)
