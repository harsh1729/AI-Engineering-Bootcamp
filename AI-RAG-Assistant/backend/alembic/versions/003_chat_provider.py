"""Add provider to chats for per-LLM guest conversation history."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003_chat_provider"
down_revision: Union[str, None] = "002_chat_guest_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column("provider", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_chats_provider", "chats", ["provider"], unique=False)
    op.create_index(
        "ix_chats_guest_id_provider",
        "chats",
        ["guest_id", "provider"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chats_guest_id_provider", table_name="chats")
    op.drop_index("ix_chats_provider", table_name="chats")
    op.drop_column("chats", "provider")
