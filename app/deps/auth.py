import logging

from dishka import FromDishka, Provider, Scope, provide
from dishka.integrations.fastapi import inject
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Annotated, NewType, cast

from fastapi.security.utils import get_authorization_scheme_param

from app.models import User
from app.services.auth import (
    UserLoginInteractor,
    UserRegisterInteractor,
)
from app.services.users import RetrieveUserInteractor
from app.services.auth.errors import AuthenticationError
from app.services.providers.protocols.token_provider import ITokenProvider


oauth2_scheme = HTTPBearer()
logger = logging.getLogger(__name__)


async def _current_user(
    token: HTTPAuthorizationCredentials,
    token_encoder: ITokenProvider,
    retrieve_user_interactor: RetrieveUserInteractor,
) -> User:
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


@inject
async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(oauth2_scheme)],
    token_provider: FromDishka[ITokenProvider],
    retrieve_user_interactor: FromDishka[RetrieveUserInteractor],
) -> User:
    return await _current_user(credentials, token_provider, retrieve_user_interactor)


_CurrentUser = NewType("_CurrentUser", User)
CurrentUserDependency = Depends(get_current_user)
CurrentUser = Annotated[_CurrentUser, CurrentUserDependency]


class AuthServicesProvider(Provider):
    scope = Scope.REQUEST

    register = provide(UserRegisterInteractor)
    login = provide(UserLoginInteractor)

    header_name = "Authorization"
    header_prefix = "bearer"

    @provide
    async def get_credentials(self, request: Request) -> HTTPAuthorizationCredentials:
        header = request.headers.get(self.header_name)
        if not header:
            raise AuthenticationError()
        scheme, credentials = get_authorization_scheme_param(header)
        if not scheme.lower() == self.header_prefix:
            raise AuthenticationError()
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)

    @provide
    async def get_current_user(
        self,
        token: HTTPAuthorizationCredentials,
        token_encoder: ITokenProvider,
        retrieve_user_interactor: RetrieveUserInteractor,
    ) -> _CurrentUser:
        return cast(
            _CurrentUser,
            await _current_user(
                token=token,
                token_encoder=token_encoder,
                retrieve_user_interactor=retrieve_user_interactor,
            ),
        )
