import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.config import LOGGED_IN_USER_USAGE_LIMIT_MESSAGE
from app.database.models.user_usage import UserUsage
from app.database.repositories.base import BaseRepository
from app.services.usage_exceptions import UsageLimitExceeded


class UserUsageRepository(BaseRepository):
    async def record_interaction(self, user_id: uuid.UUID, max_interactions: int) -> int:
        """Atomically count one logged-in user interaction across all LLM providers."""
        await self._session.execute(
            insert(UserUsage)
            .values(user_id=user_id, interaction_count=0)
            .on_conflict_do_nothing(index_elements=["user_id"])
        )

        result = await self._session.execute(
            select(UserUsage)
            .where(UserUsage.user_id == user_id)
            .with_for_update()
        )
        row = result.scalar_one()

        if row.interaction_count >= max_interactions:
            raise UsageLimitExceeded(LOGGED_IN_USER_USAGE_LIMIT_MESSAGE)

        row.interaction_count += 1
        await self._session.flush()
        return row.interaction_count
