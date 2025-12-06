"""Runtime configuration for the backend service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ml import config as ml_config


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.environ.get("MODEL_PATH", ml_config.MODEL_PATH))
    model_type: str = os.environ.get("MODEL_TYPE", ml_config.MODEL_TYPE)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
