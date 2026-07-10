import logging

from logs.config import configure_logging

configure_logging()

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ai_sdk.{name}")