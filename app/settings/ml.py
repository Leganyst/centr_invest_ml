from pathlib import Path

from category_classifier import config as category_config
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_")
    model_type: str = "catboost"
    model_path: Path = category_config.MODEL_PATH
