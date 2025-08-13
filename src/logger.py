# src/logger.py
import logging
import os
from dotenv import load_dotenv

# load .env when running locally or in container (harmless if env vars already set)
load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger that writes to stdout and optionally to a file.
    Ensures the log directory exists before creating FileHandler.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    # If relative, make it relative to repository root (/app when running in container)
    if not os.path.isabs(log_file):
        # __file__ is src/logger.py -> go up two levels to project root /app
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_file = os.path.join(root, log_file)

    # Normalize and ensure directory exists
    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # If directory creation fails, fallback to stdout-only logging
        log_file = None

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Avoid adding duplicate handlers on repeated imports
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level, logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setLevel(getattr(logging, log_level, logging.INFO))
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception:
                # If file handler creation fails, keep console handler only
                logger.exception("Unable to create file handler for logs; continuing with console only.")

    return logger
