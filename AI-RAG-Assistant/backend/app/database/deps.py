from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db_session
from app.database.repositories.chat_repository import ChatRepository
from app.database.repositories.document_repository import DocumentRepository
from app.database.repositories.guest_usage_repository import GuestUsageRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.repositories.user_repository import UserRepository
from app.database.repositories.user_usage_repository import UserUsageRepository
from app.services.chat_persistence_service import ChatPersistenceService
from app.services.document_access_service import DocumentAccessService
from app.services.rag_options_resolver import RagOptionsResolverService
from app.services.usage_tracker import UsageTrackerService


async def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[UserRepository, None]:
    yield UserRepository(session)


async def get_chat_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[ChatRepository, None]:
    yield ChatRepository(session)


async def get_message_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[MessageRepository, None]:
    yield MessageRepository(session)


async def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[DocumentRepository, None]:
    yield DocumentRepository(session)


async def get_guest_usage_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[GuestUsageRepository, None]:
    yield GuestUsageRepository(session)


async def get_user_usage_repository(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[UserUsageRepository, None]:
    yield UserUsageRepository(session)


async def get_usage_tracker_service(
    guest_usage_repository: GuestUsageRepository = Depends(get_guest_usage_repository),
    user_usage_repository: UserUsageRepository = Depends(get_user_usage_repository),
) -> AsyncGenerator[UsageTrackerService, None]:
    yield UsageTrackerService(guest_usage_repository, user_usage_repository)


async def get_chat_persistence_service(
    chat_repository: ChatRepository = Depends(get_chat_repository),
    message_repository: MessageRepository = Depends(get_message_repository),
) -> AsyncGenerator[ChatPersistenceService, None]:
    yield ChatPersistenceService(chat_repository, message_repository)


async def get_document_access_service(
    document_repository: DocumentRepository = Depends(get_document_repository),
) -> AsyncGenerator[DocumentAccessService, None]:
    yield DocumentAccessService(document_repository)


async def get_rag_options_resolver_service(
    document_repository: DocumentRepository = Depends(get_document_repository),
) -> AsyncGenerator[RagOptionsResolverService, None]:
    yield RagOptionsResolverService(document_repository)
