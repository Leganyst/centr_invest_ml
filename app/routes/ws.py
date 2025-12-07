from typing import Annotated

from dishka import AsyncContainer, FromDishka, Scope
from dishka.integrations.fastapi import DishkaRoute, inject
from fastapi import APIRouter, WebSocket
from fastapi.params import Query
from fastapi.security import HTTPAuthorizationCredentials

from app.deps.auth import CurrentUserFinder
from app.services.providers.protocols.notification_manager import INotificationManager

router = APIRouter(prefix="/notifications")


@router.websocket("")
@inject
async def notifications_ws(
    container: FromDishka[AsyncContainer],
    token: Annotated[str, Query()],
    socket: WebSocket,
    notifications: FromDishka[INotificationManager],
):
    async with container as request_container:
        get_user = await request_container.get(CurrentUserFinder)
        user = await get_user(
            HTTPAuthorizationCredentials(credentials=token, scheme="Bearer")
        )
        await socket.accept()
        user_id = user.id
    await socket.send_json({"detail": "authenticated", "user_id": user_id})

    while True:
        notification = await notifications.pop(user_id)
        if notification:
            await socket.send_json(notification.model_dump())
