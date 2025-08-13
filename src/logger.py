# src/logger.py
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """
    Return configured logger. Creates log dir if needed and falls back to stdout-only if creation fails.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    # Make relative paths absolute under project root (/app in container)
    if not os.path.isabs(log_file):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_file = os.path.join(root, log_file)

    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file)

    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # Directory creation failed (permission); we'll continue with console-only logging
        log_file = None

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level, logging.INFO))
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Optional file handler
        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setLevel(getattr(logging, log_level, logging.INFO))
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                logger.exception("Failed to create file handler; continuing with console-only logging.")

    return logger
