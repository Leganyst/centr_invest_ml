class BaseServiceError(Exception):
    detail: str = "Неизвестная ошибка сервиса"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail


class NotAuthenticatedError(BaseServiceError):
    detail = "Вы не авторизованы"


class PermissionDeniedError(BaseServiceError):
    detail = "У вас недостаточно прав для этого действия"
