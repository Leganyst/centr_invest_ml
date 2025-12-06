from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    model_type: str = "catboost"
    model_path: Path
