from dishka import AsyncContainer, Provider, Scope, make_async_container
from pydantic_settings import BaseSettings

from app.deps.db import DbConnectionProvider
from app.services.category_classifier import MlCategoryClassifier
from app.services.password_encoder import BcryptPasswordEncoder
from app.services.protocols.category_classifier import ICategoryClassifier
from app.services.protocols.password_encoder import IPasswordEncoder
from app.services.protocols.token_provider import ITokenProvider
from app.services.token_provider import JwtTokenProvider
from app.settings.app import AppSettings
from app.settings.db import DatabaseSettings
from app.settings.ml import ModelSettings


class AppProvider(Provider):
    def register_settings(self, settings: type[BaseSettings]):
        self.provide(lambda: settings(), scope=Scope.APP, provides=settings)


def create_container() -> AsyncContainer:
    provider = AppProvider()
    provider.register_settings(DatabaseSettings)
    provider.register_settings(AppSettings)
    provider.register_settings(ModelSettings)

    provider.provide(BcryptPasswordEncoder, provides=IPasswordEncoder, scope=Scope.APP)
    provider.provide(JwtTokenProvider, provides=ITokenProvider, scope=Scope.APP)
    provider.provide(MlCategoryClassifier, provides=ICategoryClassifier, scope=Scope.APP)

    container = make_async_container(
        provider,
        DbConnectionProvider(),
    )
    return container
