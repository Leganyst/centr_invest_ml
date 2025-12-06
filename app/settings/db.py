from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn

class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DB_')

    host: str
    port: int = 5432
    database: str
    username: str
    password: SecretStr

    @property
    def database_url(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.username,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            path=self.database,
        )
