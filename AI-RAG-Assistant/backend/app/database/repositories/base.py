from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Shared async SQLAlchemy session wrapper for repository classes."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
