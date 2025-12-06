from typing import Protocol

from app.schemas.auth import UserPrincipal


class ITokenProvider(Protocol):
    def encode_token(self, user_principal: UserPrincipal) -> str:
        ...

    def decode_token(self, token: str) -> UserPrincipal:
        ...
