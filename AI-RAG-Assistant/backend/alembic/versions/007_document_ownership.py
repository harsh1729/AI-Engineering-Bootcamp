"""Add user_id and guest_id to documents for caller-scoped ownership."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007_document_ownership"
down_revision: Union[str, None] = "006_user_is_enabled"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("user_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("guest_id", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    op.create_index("ix_documents_guest_id", "documents", ["guest_id"], unique=False)
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")
    op.drop_index("ix_documents_guest_id", table_name="documents")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_column("documents", "guest_id")
    op.drop_column("documents", "user_id")
