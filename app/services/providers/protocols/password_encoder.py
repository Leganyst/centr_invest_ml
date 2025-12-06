from typing import Protocol


class IPasswordEncoder(Protocol):
    def hash_password(self, password: str) -> str: ...

    def verify(self, password: str, hashed_password: str) -> bool: ...
