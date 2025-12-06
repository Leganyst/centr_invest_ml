import logging
import uuid

from sqlalchemy import ColumnElement
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.schemas.auth import LoginSchema, RegisterSchema
from app.services.auth.errors import UserAlreadyExists, WrongPasswordError
from app.services.providers.protocols.password_encoder import IPasswordEncoder
from app.services.users import RetrieveUserInteractor

logger = logging.getLogger(__name__)


class RegisterUserInteractor:
    def __init__(
        self,
        session: AsyncSession,
        password_encoder: IPasswordEncoder,
        retrieve_user_interactor: RetrieveUserInteractor,
    ):
        self.session = session
        self.password_encoder = password_encoder
        self.retrieve_user_interactor = retrieve_user_interactor

    async def __call__(self, data: RegisterSchema) -> User:
        user_exists = await self.retrieve_user_interactor.exists(
            User.email == data.email
        )

        if user_exists:
            raise UserAlreadyExists()

        user = User(
            id=uuid.uuid4(),
            email=str(data.email),
            name=data.name,
            password=self.password_encoder.hash_password(data.password),
        )

        try:
            self.session.add(user)
            await self.session.commit()
        except IntegrityError:
            raise UserAlreadyExists()

        logger.info("New user registered (email: %s)", user.email)

        return user


class LoginUserInteractor:
    def __init__(
        self,
        session: AsyncSession,
        password_encoder: IPasswordEncoder,
        retrieve_user_interactor: RetrieveUserInteractor,
    ):
        self.session = session
        self.password_encoder = password_encoder
        self.retrieve_user_interactor = retrieve_user_interactor

    async def __call__(self, data: LoginSchema) -> User:
        user = await self.retrieve_user_interactor.get(User.email == data.email)
        if not user:
            raise WrongPasswordError()

        if not self.password_encoder.verify(data.password, user.password):
            raise WrongPasswordError()
        logger.info("User logged in (email: %s)", user.email)
        return user
