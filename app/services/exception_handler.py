from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status
from starlette.types import HTTPExceptionHandler

from app.services.errors import BaseServiceError, NotAuthenticatedError, PermissionDeniedError


class DetailJsonExceptionHandler:
    def __init__(self, status_code: int):
        self.status_code = status_code

    async def __call__(self, request: Request, exc: BaseServiceError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail},
            status_code=self.status_code
        )


def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(PermissionDeniedError, DetailJsonExceptionHandler(status.HTTP_403_FORBIDDEN))
    app.add_exception_handler(NotAuthenticatedError, DetailJsonExceptionHandler(status.HTTP_401_UNAUTHORIZED))
    app.add_exception_handler(BaseServiceError, DetailJsonExceptionHandler(status.HTTP_400_BAD_REQUEST))
