import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import LOGGED_IN_USER_INTERACTIONS, LOGGED_IN_USER_USAGE_LIMIT_MESSAGE
from app.database.models.user_usage import UserUsage
from app.database.repositories.user_usage_repository import UserUsageRepository
from app.services.usage_exceptions import UsageLimitExceeded

USER_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


@pytest.fixture
def session() -> MagicMock:
    mock = MagicMock()
    mock.add = MagicMock()
    mock.flush = AsyncMock()
    mock.execute = AsyncMock()
    return mock


def _scalar_result(row: UserUsage | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = row
    return result


@pytest.mark.asyncio
async def test_user_usage_repository_records_first_interaction(session: MagicMock) -> None:
    row = UserUsage(user_id=USER_ID, interaction_count=0)
    session.execute = AsyncMock(side_effect=[MagicMock(), _scalar_result(row)])

    repo = UserUsageRepository(session)
    count = await repo.record_interaction(USER_ID, LOGGED_IN_USER_INTERACTIONS)

    assert count == 1
    assert row.interaction_count == 1
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_usage_repository_raises_when_limit_reached(session: MagicMock) -> None:
    row = UserUsage(user_id=USER_ID, interaction_count=LOGGED_IN_USER_INTERACTIONS)
    session.execute = AsyncMock(side_effect=[MagicMock(), _scalar_result(row)])

    repo = UserUsageRepository(session)

    with pytest.raises(UsageLimitExceeded, match=LOGGED_IN_USER_USAGE_LIMIT_MESSAGE):
        await repo.record_interaction(USER_ID, LOGGED_IN_USER_INTERACTIONS)

    assert row.interaction_count == LOGGED_IN_USER_INTERACTIONS
    session.flush.assert_not_awaited()
