from datetime import date
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models import BaseModel
from app.models.enums import TransactionCategory


if TYPE_CHECKING:
    from app.models.user import User


class Transaction(BaseModel):
    __tablename__ = "transactions"

    date: Mapped[date]
    category: Mapped[TransactionCategory | None] = mapped_column(nullable=True)

    withdrawal: Mapped[float]
    deposit: Mapped[float]
    balance: Mapped[float]

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship()
