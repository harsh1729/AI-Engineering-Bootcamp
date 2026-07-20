import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIRECTORY = PROJECT_ROOT / "runtime_logs"
LOG_DIRECTORY.mkdir(exist_ok=True)
LOG_FILE = LOG_DIRECTORY / "ai_sdk.log"

LOG_LEVEL = logging.INFO

MAX_LOG_FILE_SIZE_MB = 5
LOG_BACKUP_COUNT = 3

ENABLE_CONSOLE_LOGGING = False
ENABLE_FILE_LOGGING = True



def configure_logging() -> None:
    logger = logging.getLogger("ai_sdk")

    # Prevent duplicate handlers if configure_logging() is called multiple times
    if logger.handlers:
        return

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    if ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if ENABLE_FILE_LOGGING:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False