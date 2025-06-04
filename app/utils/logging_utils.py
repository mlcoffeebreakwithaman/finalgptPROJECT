import logging
from logging import Logger

def get_logger(name: str = "studygpt", level: int = logging.INFO) -> Logger:
    """
    Returns a configured logger instance.
    Logs to console with format: [timestamp] [LEVEL] message
    Avoids adding duplicate handlers if re-imported.
    
    Args:
        name (str): Logger name (default: "studygpt").
        level (int): Logging level (default: logging.INFO).
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Format: [2025-06-01 10:00:00] [INFO] Message
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False  # Prevent double logging if root handler exists

    return logger

# ✅ Global logger used throughout the app
logger = get_logger()