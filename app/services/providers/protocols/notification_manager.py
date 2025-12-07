from typing import Protocol
from uuid import UUID

from app.schemas.notifications import NotificationSchema


class INotificationManager(Protocol):
    async def send(self, user_id: UUID, notification: NotificationSchema): ...

    async def pop(self, user_id: UUID) -> NotificationSchema: ...
