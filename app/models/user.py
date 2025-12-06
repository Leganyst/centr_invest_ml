from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.models.enums import UserRole


class User(BaseModel):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]

    password: Mapped[str]
    role: Mapped[UserRole]
