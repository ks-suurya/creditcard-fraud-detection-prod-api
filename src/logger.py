# src/logger.py
import logging
import sys
import os
from pathlib import Path
from src.utils import get_env_var

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Optionally include environment name for context
ENV = os.getenv("ENV", "dev")

def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger.
    Logs go to stdout and optionally to a file.
    """
    log_level = get_env_var("LOG_LEVEL", "INFO").upper()

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Optional file logging with directory creation
        log_file = get_env_var("LOG_FILE", None, required=False)
        if log_file:
            try:
                # Create directory if it doesn't exist
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create file handler
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(console_format)
                logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # If file logging fails, log the error to console and continue
                print(f"Warning: Could not set up file logging to {log_file}: {e}", file=sys.stderr)
                print("Continuing with console logging only...", file=sys.stderr)

    return logger

def log_info(message: str, **kwargs):
    logger.info(f"[{ENV}] {message} | {kwargs}")

def log_error(message: str, **kwargs):
    logger.error(f"[{ENV}] {message} | {kwargs}")