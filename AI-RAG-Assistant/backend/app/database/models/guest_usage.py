from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.models._types import utcnow


class GuestUsage(Base):
    __tablename__ = "guest_usage"

    guest_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )
