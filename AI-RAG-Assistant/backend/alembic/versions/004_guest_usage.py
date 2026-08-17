"""Add guest_usage table for persistent demo interaction counts."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004_guest_usage"
down_revision: Union[str, None] = "003_chat_provider"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "guest_usage",
        sa.Column("guest_id", sa.String(length=64), nullable=False),
        sa.Column("interaction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("guest_id"),
    )


def downgrade() -> None:
    op.drop_table("guest_usage")
