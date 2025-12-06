from datetime import datetime
from uuid import UUID

from app.schemas.base import BaseSchema


class TransactionSchema(BaseSchema):
    id: UUID
    date: datetime
    withdrawal: float
    deposit: float
    balance: float
