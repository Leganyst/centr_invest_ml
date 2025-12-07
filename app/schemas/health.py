from datetime import datetime

from app.schemas.base import BaseSchema
from typing import Literal


class JobSchema(BaseSchema):
    id: str
    name: str
    next_run_time: datetime


class HealthSchema(BaseSchema):
    status: Literal["ok", "error"]
    jobs: list[JobSchema]
