from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.base import BaseScheduler
from dishka import AsyncContainer, Provider, Scope, make_async_container
from dishka.integrations.fastapi import FastapiProvider
from pydantic_settings import BaseSettings

from app.deps.auth import AuthServicesProvider
from app.deps.db import DbConnectionProvider
from app.services.providers.category_classifier import MlCategoryClassifier
from app.services.providers.notification_manager import AsyncioNotificationManager
from app.services.providers.password_encoder import BcryptPasswordEncoder
from app.services.providers.protocols.category_classifier import ICategoryClassifier
from app.services.providers.protocols.notification_manager import INotificationManager
from app.services.providers.protocols.password_encoder import IPasswordEncoder
from app.services.providers.protocols.token_provider import ITokenProvider
from app.services.providers.token_provider import JwtTokenProvider
from app.services.transactions import TransactionServicesProvider
from app.services.users import UserServicesProvider
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
    provider.provide(
        MlCategoryClassifier, provides=ICategoryClassifier, scope=Scope.APP
    )
    provider.provide(
        lambda: AsyncIOScheduler(), provides=BaseScheduler, scope=Scope.APP
    )
    provider.provide(
        AsyncioNotificationManager, provides=INotificationManager, scope=Scope.APP
    )

    container = make_async_container(
        provider,
        DbConnectionProvider(),
        AuthServicesProvider(),
        UserServicesProvider(),
        TransactionServicesProvider(),
        FastapiProvider(),
    )
    return container
