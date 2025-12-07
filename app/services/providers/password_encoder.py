from app.services.providers.protocols.password_encoder import IPasswordEncoder
import bcrypt


class BcryptPasswordEncoder(IPasswordEncoder):
    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify(self, password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed_password.encode())
