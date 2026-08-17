from app.database.base import Base
from app.database import models  # noqa: F401


def test_metadata_includes_core_tables() -> None:
    table_names = set(Base.metadata.tables.keys())

    assert table_names == {
        "users",
        "chats",
        "messages",
        "documents",
        "guest_usage",
        "user_usage",
    }


def test_user_chat_message_relationships() -> None:
    from app.database.models.chat_session import Chat
    from app.database.models.message import Message
    from app.database.models.user import User

    assert User.chats.property.back_populates == "user"
    assert Chat.user.property.back_populates == "chats"
    assert Chat.messages.property.back_populates == "chat"
    assert Message.chat.property.back_populates == "messages"


def test_user_has_enabled_flag() -> None:
    from app.database.models.user import User

    enabled_column = User.__table__.c.is_enabled
    assert enabled_column.nullable is False


def test_chat_supports_guest_or_user_owner() -> None:
    from app.database.models.chat_session import Chat

    guest_id_column = Chat.__table__.c.guest_id
    user_id_column = Chat.__table__.c.user_id

    assert guest_id_column.nullable is True
    assert user_id_column.nullable is True


def test_document_supports_guest_or_user_owner() -> None:
    from app.database.models.stored_document import Document

    guest_id_column = Document.__table__.c.guest_id
    user_id_column = Document.__table__.c.user_id

    assert guest_id_column.nullable is True
    assert user_id_column.nullable is True


def test_user_document_relationship() -> None:
    from app.database.models.stored_document import Document
    from app.database.models.user import User

    assert User.documents.property.back_populates == "user"
    assert Document.user.property.back_populates == "documents"
