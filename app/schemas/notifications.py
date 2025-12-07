from uuid import UUID

from app.schemas.base import BaseSchema


class NotificationSchema(BaseSchema):
    user_id: UUID
    text: str
    type: str
