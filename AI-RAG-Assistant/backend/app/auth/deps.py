import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database.deps import get_user_repository
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.services.jwt_service import InvalidTokenError, decode_access_token

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    user_repository: UserRepository = Depends(get_user_repository),
) -> User | None:
    if credentials is None:
        return None

    try:
        user_id = decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = await user_repository.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")

    return user


async def get_current_user(
    current_user: User | None = Depends(get_optional_current_user),
) -> User:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
