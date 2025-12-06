import logging

from dishka import FromDishka, Provider, Scope, provide
from dishka.integrations.fastapi import inject
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Annotated

from app.models import User

from app.services.auth import (
    LoginUserInteractor,
    RegisterUserInteractor,
)
from app.services.users import RetrieveUserInteractor
from app.services.auth.errors import AuthenticationError
from app.services.providers.protocols.token_provider import ITokenProvider


oauth2_scheme = HTTPBearer()
logger = logging.getLogger(__name__)


@inject
async def current_user(
    token: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    token_encoder: FromDishka[ITokenProvider],
    retrieve_user_interactor: FromDishka[RetrieveUserInteractor],
) -> User:
    logger.debug("Request with token %s", repr(token))
    if not token or not token.credentials:
        raise AuthenticationError()
    try:
        user = token_encoder.decode_token(token.credentials)
    except Exception:
        logger.debug("Failed to decode token", exc_info=True)
        raise AuthenticationError()
    db_user = await retrieve_user_interactor.get(User.id == user.user_id)
    if not db_user:
        logger.debug("Failed to retrieve user %s", repr(db_user))
        raise AuthenticationError()
    return db_user


CurrentUser = Annotated[User, Depends(current_user)]


class AuthServicesProvider(Provider):
    scope = Scope.REQUEST

    retrieve = provide(RetrieveUserInteractor)
    register = provide(RegisterUserInteractor)
    login = provide(LoginUserInteractor)
