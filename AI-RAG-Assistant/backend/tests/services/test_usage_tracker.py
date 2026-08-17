from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import LOGGED_IN_USER_INTERACTIONS, MAX_DEMO_INTERACTIONS
from app.services.usage_exceptions import UsageLimitExceeded
from app.services.usage_tracker import UsageTrackerService


@pytest.mark.asyncio
async def test_usage_tracker_service_delegates_guest_interaction() -> None:
    guest_repository = MagicMock()
    guest_repository.record_interaction = AsyncMock(return_value=3)
    user_repository = MagicMock()
    service = UsageTrackerService(guest_repository, user_repository)

    count = await service.record_guest_interaction("guest-1")

    assert count == 3
    guest_repository.record_interaction.assert_awaited_once_with(
        "guest-1",
        MAX_DEMO_INTERACTIONS,
    )
    user_repository.record_interaction.assert_not_called()


@pytest.mark.asyncio
async def test_usage_tracker_service_delegates_user_interaction() -> None:
    import uuid

    user_id = uuid.uuid4()
    guest_repository = MagicMock()
    user_repository = MagicMock()
    user_repository.record_interaction = AsyncMock(return_value=42)
    service = UsageTrackerService(guest_repository, user_repository)

    count = await service.record_user_interaction(user_id)

    assert count == 42
    user_repository.record_interaction.assert_awaited_once_with(
        user_id,
        LOGGED_IN_USER_INTERACTIONS,
    )
    guest_repository.record_interaction.assert_not_called()


@pytest.mark.asyncio
async def test_usage_tracker_service_propagates_limit_error() -> None:
    guest_repository = MagicMock()
    guest_repository.record_interaction = AsyncMock(
        side_effect=UsageLimitExceeded("limit reached"),
    )
    service = UsageTrackerService(guest_repository, MagicMock())

    with pytest.raises(UsageLimitExceeded, match="limit reached"):
        await service.record_guest_interaction("guest-1")
