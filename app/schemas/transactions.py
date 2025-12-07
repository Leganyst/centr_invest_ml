import contextlib
from datetime import date, datetime
from uuid import UUID

from pydantic import field_validator, model_validator

from app.models.enums import TransactionCategory
from app.schemas.base import BaseSchema


class TransactionCreateSchema(BaseSchema):
    date: date
    withdrawal: float
    deposit: float
    balance: float
    category: TransactionCategory | None = None

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, v: str):
        if isinstance(v, str):
            with contextlib.suppress(ValueError):
                return datetime.strptime(v, "%d/%m/%Y").date()
        return v



class TransactionSchema(TransactionCreateSchema):
    id: UUID
    created_at: datetime
    updated_at: datetime

