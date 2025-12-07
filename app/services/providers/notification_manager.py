from asyncio import Queue
from uuid import UUID
from collections import defaultdict
from collections.abc import Mapping
from app.schemas.notifications import NotificationSchema
from app.services.providers.protocols.notification_manager import INotificationManager

notifications_pool: Mapping[UUID, Queue[NotificationSchema]] = defaultdict(Queue)


class AsyncioNotificationManager(INotificationManager):
    async def send(self, user_id: UUID, notification: NotificationSchema):
        await notifications_pool[user_id].put(notification)

    async def pop(self, user_id: UUID) -> NotificationSchema:
        notification = await notifications_pool[user_id].get()
        return notification
