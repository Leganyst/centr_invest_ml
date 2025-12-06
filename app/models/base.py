from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from uuid import UUID


class BaseModel(DeclarativeBase):
    id: Mapped[UUID] = mapped_column(primary_key=True)
