from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user
from app.database.deps import get_user_repository
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.models.auth import (
    AuthResponse,
    LoginRequest,
    MeResponse,
    RegisterRequest,
    RegisterResponse,
    UserPublic,
)
from app.services.auth_service import (
    AccountDisabledError,
    AccountNotApprovedError,
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
)

router = APIRouter(prefix="/auth", tags=["auth"])


async def get_auth_service(
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(user_repository)


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RegisterResponse:
    try:
        message = await auth_service.register(request)
        return RegisterResponse(message=message)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    try:
        return await auth_service.login(request)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AccountNotApprovedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AccountDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/me", response_model=MeResponse)
async def me(current_user: User = Depends(get_current_user)) -> MeResponse:
    return MeResponse(
        user=UserPublic(
            id=current_user.id,
            name=current_user.name,
            email=current_user.email,
            is_admin=current_user.is_admin,
        ),
        last_login_at=current_user.last_login_at,
    )
