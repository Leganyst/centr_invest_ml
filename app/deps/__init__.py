from dishka import AsyncContainer, Provider, Scope, make_async_container
from pydantic_settings import BaseSettings

from app.deps.db import DbConnectionProvider
from app.settings.app import AppSettings
from app.settings.db import DatabaseSettings
from app.settings.ml import ModelSettings


class SettingsProvider(Provider):
    def register_settings(self, settings: type[BaseSettings]):
        self.provide(lambda: settings(), scope=Scope.APP, provides=settings)


def create_container() -> AsyncContainer:
    settings = SettingsProvider()
    settings.register_settings(DatabaseSettings)
    settings.register_settings(AppSettings)
    settings.register_settings(ModelSettings)

    container = make_async_container(
        settings,
        DbConnectionProvider(),
    )
    return container
