from typing import Annotated

from dishka import AsyncContainer, FromDishka, Scope
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, WebSocket
from fastapi.params import Query
from fastapi.security import HTTPAuthorizationCredentials

from app.deps.auth import CurrentUserFinder
from app.services.providers.protocols.notification_manager import INotificationManager

router = APIRouter(route_class=DishkaRoute, prefix="/notifications")


@router.websocket("/ws")
async def notifications_ws(
    container: AsyncContainer,
    token: Annotated[str, Query()],
    socket: WebSocket,
    notifications: FromDishka[INotificationManager],
):
    async with container(scope=Scope.REQUEST) as container:
        get_user = await container.get(CurrentUserFinder)
        user = await get_user(
            HTTPAuthorizationCredentials(credentials=token, scheme="Bearer")
        )
        await socket.accept()
        user_id = user.id

    while True:
        notification = await notifications.pop(user_id)
        if notification:
            await socket.send_json(notification.model_dump())
