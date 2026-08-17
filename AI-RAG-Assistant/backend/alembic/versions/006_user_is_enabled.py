"""Add is_enabled to users for admin disable without deletion."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006_user_is_enabled"
down_revision: Union[str, None] = "005_user_usage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.alter_column("users", "is_enabled", server_default=None)


def downgrade() -> None:
    op.drop_column("users", "is_enabled")
