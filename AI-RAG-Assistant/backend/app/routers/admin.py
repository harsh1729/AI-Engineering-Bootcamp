import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_admin_user
from app.database.deps import get_user_repository
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.models.auth import AdminUserSummary, AdminUsersResponse
from app.services.admin_user_service import (
    AdminUserService,
    UserNotFoundError,
    UserNotManageableError,
)

router = APIRouter(prefix="/admin", tags=["admin"])


async def get_admin_user_service(
    user_repository: UserRepository = Depends(get_user_repository),
) -> AdminUserService:
    return AdminUserService(user_repository)


@router.get("/users", response_model=AdminUsersResponse)
async def list_users(
    _: User = Depends(get_current_admin_user),
    admin_user_service: AdminUserService = Depends(get_admin_user_service),
) -> AdminUsersResponse:
    return await admin_user_service.list_users()


@router.post("/users/{user_id}/approve", response_model=AdminUserSummary)
async def approve_user(
    user_id: uuid.UUID,
    current_admin: User = Depends(get_current_admin_user),
    admin_user_service: AdminUserService = Depends(get_admin_user_service),
) -> AdminUserSummary:
    try:
        return await admin_user_service.approve_user(
            user_id,
            acting_admin_id=current_admin.id,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UserNotManageableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/users/{user_id}/disable", response_model=AdminUserSummary)
async def disable_user(
    user_id: uuid.UUID,
    current_admin: User = Depends(get_current_admin_user),
    admin_user_service: AdminUserService = Depends(get_admin_user_service),
) -> AdminUserSummary:
    try:
        return await admin_user_service.disable_user(
            user_id,
            acting_admin_id=current_admin.id,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UserNotManageableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/users/{user_id}/enable", response_model=AdminUserSummary)
async def enable_user(
    user_id: uuid.UUID,
    current_admin: User = Depends(get_current_admin_user),
    admin_user_service: AdminUserService = Depends(get_admin_user_service),
) -> AdminUserSummary:
    try:
        return await admin_user_service.enable_user(
            user_id,
            acting_admin_id=current_admin.id,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UserNotManageableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
