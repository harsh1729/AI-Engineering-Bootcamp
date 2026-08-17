"""Add content_type to documents for text vs OCR-indexed image uploads."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "008_document_content_type"
down_revision: Union[str, None] = "007_document_ownership"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "content_type",
            sa.String(length=16),
            nullable=False,
            server_default="text",
        ),
    )
    op.alter_column("documents", "content_type", server_default=None)


def downgrade() -> None:
    op.drop_column("documents", "content_type")
