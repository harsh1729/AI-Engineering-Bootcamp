import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserPublic(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    is_admin: bool = False


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class RegisterResponse(BaseModel):
    message: str


class MeResponse(BaseModel):
    user: UserPublic
    last_login_at: datetime | None = None


class AdminUserSummary(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    is_enabled: bool
    created_at: datetime


class AdminUsersResponse(BaseModel):
    pending: list[AdminUserSummary]
    approved: list[AdminUserSummary]
