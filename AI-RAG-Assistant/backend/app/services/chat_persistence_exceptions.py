class ChatNotFoundError(Exception):
    """Raised when a chat_id does not exist."""


class ChatAccessDeniedError(Exception):
    """Raised when a chat does not belong to the requesting guest."""
