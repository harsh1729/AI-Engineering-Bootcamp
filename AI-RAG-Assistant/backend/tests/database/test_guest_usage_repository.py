from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import GUEST_USAGE_LIMIT_MESSAGE, MAX_DEMO_INTERACTIONS
from app.database.models.guest_usage import GuestUsage
from app.database.repositories.guest_usage_repository import GuestUsageRepository
from app.services.usage_exceptions import UsageLimitExceeded


@pytest.fixture
def session() -> MagicMock:
    mock = MagicMock()
    mock.add = MagicMock()
    mock.flush = AsyncMock()
    mock.execute = AsyncMock()
    return mock


def _scalar_result(row: GuestUsage | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = row
    return result


@pytest.mark.asyncio
async def test_guest_usage_repository_records_first_interaction(session: MagicMock) -> None:
    row = GuestUsage(guest_id="guest-1", interaction_count=0)
    session.execute = AsyncMock(side_effect=[MagicMock(), _scalar_result(row)])

    repo = GuestUsageRepository(session)
    count = await repo.record_interaction("guest-1", MAX_DEMO_INTERACTIONS)

    assert count == 1
    assert row.interaction_count == 1
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_guest_usage_repository_raises_when_limit_reached(session: MagicMock) -> None:
    row = GuestUsage(guest_id="guest-1", interaction_count=MAX_DEMO_INTERACTIONS)
    session.execute = AsyncMock(side_effect=[MagicMock(), _scalar_result(row)])

    repo = GuestUsageRepository(session)

    with pytest.raises(UsageLimitExceeded, match=GUEST_USAGE_LIMIT_MESSAGE):
        await repo.record_interaction("guest-1", MAX_DEMO_INTERACTIONS)

    assert row.interaction_count == MAX_DEMO_INTERACTIONS
    session.flush.assert_not_awaited()
