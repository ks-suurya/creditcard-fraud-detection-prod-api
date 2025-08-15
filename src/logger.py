# src/logger.py
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """
    Robust logger: console + optional file handler. Creates log directory if missing.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    if not os.path.isabs(log_file):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_file = os.path.join(root, log_file)

    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file)

    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # If cannot create logs directory, skip file logging.
        log_file = None

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level, logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setLevel(getattr(logging, log_level, logging.INFO))
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception:
                logger.exception("Unable to create FileHandler; continuing with console only.")

    return logger
