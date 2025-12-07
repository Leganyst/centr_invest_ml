from app.schemas.auth import UserPrincipal
from app.services.providers.protocols.token_provider import ITokenProvider
from app.settings.app import AppSettings
import jwt


class JwtTokenProvider(ITokenProvider):
    algorithm = "HS256"

    def __init__(self, settings: AppSettings) -> None:
        self.secret_key: str = settings.jwt_secret.get_secret_value()

    def encode_token(self, user_principal: UserPrincipal) -> str:
        return jwt.encode(
            user_principal.model_dump(mode="json"),
            key=self.secret_key,
            algorithm=self.algorithm,
        )

    def decode_token(self, token: str) -> UserPrincipal:
        decoded = jwt.decode(token, key=self.secret_key, algorithms=[self.algorithm])
        return UserPrincipal.model_validate(decoded)
