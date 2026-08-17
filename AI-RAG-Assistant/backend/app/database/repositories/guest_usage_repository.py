from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.config import GUEST_USAGE_LIMIT_MESSAGE
from app.database.models.guest_usage import GuestUsage
from app.database.repositories.base import BaseRepository
from app.services.usage_exceptions import UsageLimitExceeded


class GuestUsageRepository(BaseRepository):
    async def record_interaction(self, guest_id: str, max_interactions: int) -> int:
        """Atomically count one guest interaction across all LLM providers."""
        await self._session.execute(
            insert(GuestUsage)
            .values(guest_id=guest_id, interaction_count=0)
            .on_conflict_do_nothing(index_elements=["guest_id"])
        )

        result = await self._session.execute(
            select(GuestUsage)
            .where(GuestUsage.guest_id == guest_id)
            .with_for_update()
        )
        row = result.scalar_one()

        if row.interaction_count >= max_interactions:
            raise UsageLimitExceeded(GUEST_USAGE_LIMIT_MESSAGE)

        row.interaction_count += 1
        await self._session.flush()
        return row.interaction_count
