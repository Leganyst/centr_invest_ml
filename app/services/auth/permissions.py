from fastapi.params import Depends

from app.models.enums import UserRole
from app.deps.auth import CurrentUser
from app.services.errors import PermissionDeniedError


class PermissionRequired:
    def __init__(self, role: UserRole):
        self.role = role

    async def __call__(self, current_user: CurrentUser):
        if not current_user.role == self.role:
            raise PermissionDeniedError


def permission_required(item: UserRole):
    return Depends(PermissionRequired(item))
