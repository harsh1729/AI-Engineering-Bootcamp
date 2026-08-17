import uuid
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt

from app.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET_KEY


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be decoded or is missing required claims."""


def create_access_token(user_id: uuid.UUID) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> uuid.UUID:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise InvalidTokenError("Token subject is missing.")
        return uuid.UUID(subject)
    except (JWTError, ValueError) as exc:
        raise InvalidTokenError("Invalid or expired token.") from exc
