from app.config import MAX_DEMO_INTERACTIONS


class UsageLimitExceeded(Exception):
    """Raised when a guest has exhausted their demo interaction budget."""


class InMemoryUsageTracker:
    """Tracks per-guest interaction counts in a process-local dict.

    This is a demo-only implementation: state is lost on restart and isn't
    shared across processes. Migrating to a persistent store later (e.g.
    Postgres) only requires a class with the same `record_interaction`
    method - callers never touch the underlying storage directly.
    """

    def __init__(self, max_interactions: int = MAX_DEMO_INTERACTIONS) -> None:
        self._max_interactions = max_interactions
        self._usage: dict[str, int] = {}

    def record_interaction(self, guest_id: str) -> int:
        """Counts one user interaction for `guest_id`. Raises
        UsageLimitExceeded (without recording it) if the guest has already
        reached their limit, so blocked requests never reach the LLM."""
        current = self._usage.get(guest_id, 0)

        if current >= self._max_interactions:
            raise UsageLimitExceeded(
                f"Demo usage limit of {self._max_interactions} messages reached for this session."
            )

        self._usage[guest_id] = current + 1
        return self._usage[guest_id]


usage_tracker = InMemoryUsageTracker()
