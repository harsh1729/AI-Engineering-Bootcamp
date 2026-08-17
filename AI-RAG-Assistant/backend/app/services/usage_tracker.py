import uuid

from app.config import LOGGED_IN_USER_INTERACTIONS, MAX_DEMO_INTERACTIONS
from app.database.repositories.guest_usage_repository import GuestUsageRepository
from app.database.repositories.user_usage_repository import UserUsageRepository


class UsageTrackerService:
    """Persists guest and logged-in user interaction counts in Postgres."""

    def __init__(
        self,
        guest_usage_repository: GuestUsageRepository,
        user_usage_repository: UserUsageRepository,
        max_guest_interactions: int = MAX_DEMO_INTERACTIONS,
        max_user_interactions: int = LOGGED_IN_USER_INTERACTIONS,
    ) -> None:
        self._guest_usage_repository = guest_usage_repository
        self._user_usage_repository = user_usage_repository
        self._max_guest_interactions = max_guest_interactions
        self._max_user_interactions = max_user_interactions

    async def record_guest_interaction(self, guest_id: str) -> int:
        """Counts one guest interaction. Raises UsageLimitExceeded at the guest cap."""
        return await self._guest_usage_repository.record_interaction(
            guest_id,
            self._max_guest_interactions,
        )

    async def record_user_interaction(self, user_id: uuid.UUID) -> int:
        """Counts one logged-in user interaction. Raises UsageLimitExceeded at the user cap."""
        return await self._user_usage_repository.record_interaction(
            user_id,
            self._max_user_interactions,
        )
