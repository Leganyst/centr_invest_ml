from app.services.errors import BaseServiceError, NotAuthenticatedError


class UserAlreadyExists(BaseServiceError):
    detail = "Пользователь с такой почтой уже зарегистрирован"


class AuthenticationError(NotAuthenticatedError):
    detail = "Ошибка аутентификации"


class WrongPasswordError(NotAuthenticatedError):
    detail = "Проверьте правильность ввода логина и пароля."
