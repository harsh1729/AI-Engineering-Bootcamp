"""Add guest_id to chats and make user_id nullable for anonymous sessions."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002_chat_guest_id"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("chats", "user_id", existing_type=sa.UUID(), nullable=True)

    op.add_column(
        "chats",
        sa.Column("guest_id", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_chats_guest_id", "chats", ["guest_id"], unique=False)

    op.create_check_constraint(
        "ck_chats_user_or_guest",
        "chats",
        "user_id IS NOT NULL OR guest_id IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint("ck_chats_user_or_guest", "chats", type_="check")
    op.drop_index("ix_chats_guest_id", table_name="chats")
    op.drop_column("chats", "guest_id")
    op.alter_column("chats", "user_id", existing_type=sa.UUID(), nullable=False)
