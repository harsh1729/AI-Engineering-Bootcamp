"""Add user_usage table for persistent logged-in user interaction counts."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005_user_usage"
down_revision: Union[str, None] = "004_guest_usage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_usage",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("interaction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_usage")
