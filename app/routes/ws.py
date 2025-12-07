from typing import Annotated

from dishka import AsyncContainer, FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter, WebSocket
from fastapi.params import Query
from fastapi.security import HTTPAuthorizationCredentials
from starlette.websockets import WebSocketDisconnect

from app.deps.auth import CurrentUserFinder
from app.services.providers.protocols.notification_manager import INotificationManager
import logging


router = APIRouter(prefix="/notifications")
logger = logging.getLogger(__name__)


@router.websocket("")
@inject
async def notifications_ws(
    container: FromDishka[AsyncContainer],
    token: Annotated[str, Query()],
    socket: WebSocket,
    notifications: FromDishka[INotificationManager],
):
    async with container() as request_container:
        get_user = await request_container.get(CurrentUserFinder)
        user = await get_user(
            HTTPAuthorizationCredentials(credentials=token, scheme="Bearer")
        )
        await socket.accept()
        user_id = user.id

    logger.info("User connected to notifications socket (user_id=%s)", user_id)
    try:
        await socket.send_json({"detail": "authenticated", "user_id": str(user_id)})

        while True:
            notification = await notifications.pop(user_id)
            if notification:
                await socket.send_json(notification.model_dump(mode="json"))
    except WebSocketDisconnect:
        await socket.close()
