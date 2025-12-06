from app.schemas.base import BaseSchema
from typing import Literal


class HealthSchema(BaseSchema):
    status: Literal["ok", "error"]
